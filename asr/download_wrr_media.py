import re
import csv
import json
import time
import argparse
import threading
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.elararchive.org/uncategorized/SO_a1fb087d-aa27-44f3-a3c6-bce1da6bdd7b/"
WRR_PATTERN = re.compile(r"(?<![A-Za-z0-9])wrr0\d{3,4}", re.IGNORECASE)

AUDIO_KEYWORDS = {
    "wav", "wave", "mp3", "mpeg-audio", "audio", "flac", "aac", "ogg", "m4a", "x-wav"
}

VIDEO_KEYWORDS = {
    "mp4", "mpeg-video", "video", "mov", "avi", "mkv", "webm", "quicktime"
}


@dataclass
class ResultItem:
    page_url: str
    title: str
    href: str
    file_format: str
    access: str


@dataclass
class ParentPage:
    title: str
    href: str
    source_page: str


@dataclass
class CandidateMediaFile:
    wrr_id: str
    parent_title: str
    parent_url: str
    file_title: str
    file_page_url: str
    file_format: str
    access: str
    is_audio: bool
    is_video: bool


@dataclass
class DownloadResult:
    wrr_id: str
    status: str
    selected_type: str
    source_page: str
    file_page_url: str
    download_url: str
    output_wav: str
    error: str


print_lock = threading.Lock()


def safe_print(*args):
    with print_lock:
        print(*args, flush=True)


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Please install it first: brew install ffmpeg")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    })
    return session


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str:
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.strip().split())


def extract_href_from_item(item, base_url: str) -> str:
    a = item.select_one(".archive_name a[href]")
    if a and a.get("href"):
        return urljoin(base_url, a["href"])

    onclick = item.get("onclick", "")
    match = re.search(r"window\.location\s*=\s*['\"]([^'\"]+)['\"]", onclick)
    if match:
        return urljoin(base_url, match.group(1))

    return ""


def parse_result_items(html: str, page_url: str) -> List[ResultItem]:
    soup = BeautifulSoup(html, "html.parser")
    result_list = soup.select_one("div#search-results")

    if not result_list:
        return []

    items = []

    for item in result_list.select("div.result-item"):
        a = item.select_one(".archive_name a")
        title = normalize_text(a.get_text()) if a else normalize_text(item.get("title", ""))
        href = extract_href_from_item(item, page_url)

        fmt_el = item.select_one(".archive_description p")
        file_format = normalize_text(fmt_el.get_text() if fmt_el else "")

        access_el = item.select_one(".archive_data_content p")
        access = normalize_text(access_el.get_text() if access_el else "")

        if title or href:
            items.append(ResultItem(page_url, title, href, file_format, access))

    return items


def scan_local_wrr_ids(txt_dir: Path) -> Set[str]:
    wrr_ids = set()

    for txt_file in txt_dir.rglob("*.txt"):
        match = WRR_PATTERN.search(txt_file.name)
        if match:
            wrr_ids.add(match.group(0).lower())

    return wrr_ids


def detect_wrr_ids(text: str, target_ids: Set[str]) -> Set[str]:
    found = set()

    for match in WRR_PATTERN.finditer(text or ""):
        wrr_id = match.group(0).lower()
        if wrr_id in target_ids:
            found.add(wrr_id)

    return found


def is_audio_format(file_format: str, title: str, href: str) -> bool:
    text = f"{file_format} {title} {href}".lower()
    return any(k in text for k in AUDIO_KEYWORDS)


def is_video_format(file_format: str, title: str, href: str) -> bool:
    text = f"{file_format} {title} {href}".lower()
    return any(k in text for k in VIDEO_KEYWORDS)


def crawl_parent_pages(base_url: str, max_page: int, delay: float) -> List[ParentPage]:
    session = make_session()
    pages = []

    for pg in range(1, max_page + 1):
        page_url = f"{base_url}?pg={pg}"
        safe_print(f"[Main list] Fetching page {pg}/{max_page}: {page_url}")

        try:
            html = fetch_html(session, page_url)
            items = parse_result_items(html, page_url)
            safe_print(f"[Main list] Page {pg} contains {len(items)} items")
        except Exception as e:
            safe_print(f"[Warning] Failed to fetch main list page: {page_url} | {e}")
            continue

        for item in items:
            if item.href:
                pages.append(ParentPage(item.title, item.href, page_url))

        time.sleep(delay)

    seen = set()
    unique = []

    for p in pages:
        if p.href not in seen:
            seen.add(p.href)
            unique.append(p)

    return unique


def analyze_parent_page(
    parent: ParentPage,
    target_wrr_ids: Set[str],
    delay: float,
) -> Dict[str, List[CandidateMediaFile]]:
    session = make_session()
    matches: Dict[str, List[CandidateMediaFile]] = {}

    try:
        html = fetch_html(session, parent.href)
        items = parse_result_items(html, parent.href)
    except Exception as e:
        safe_print(f"[Warning] Failed to fetch child page: {parent.href} | {e}")
        return matches

    for item in items:
        combined = f"{item.title} {item.href} {item.file_format}"
        wrr_ids = detect_wrr_ids(combined, target_wrr_ids)

        if not wrr_ids:
            continue

        is_audio = is_audio_format(item.file_format, item.title, item.href)
        is_video = is_video_format(item.file_format, item.title, item.href)

        if not is_audio and not is_video:
            continue

        for wrr_id in wrr_ids:
            matches.setdefault(wrr_id, []).append(
                CandidateMediaFile(
                    wrr_id=wrr_id,
                    parent_title=parent.title,
                    parent_url=parent.href,
                    file_title=item.title,
                    file_page_url=item.href,
                    file_format=item.file_format,
                    access=item.access,
                    is_audio=is_audio,
                    is_video=is_video,
                )
            )

    if delay > 0:
        time.sleep(delay)

    return matches


def choose_candidate(candidates: List[CandidateMediaFile]) -> Optional[CandidateMediaFile]:
    audio = [c for c in candidates if c.is_audio]
    if audio:
        return audio[0]

    video = [c for c in candidates if c.is_video]
    if video:
        return video[0]

    return None


def extract_download_url(session: requests.Session, file_page_url: str) -> str:
    html = fetch_html(session, file_page_url)
    soup = BeautifulSoup(html, "html.parser")

    a = soup.select_one('a.fa-download[href*="/download/file/"]')
    if a and a.get("href"):
        return urljoin(file_page_url, a["href"])

    a = soup.select_one('a[href*="/download/file/"]')
    if a and a.get("href"):
        return urljoin(file_page_url, a["href"])

    return ""


def download_raw_file(session: requests.Session, url: str, raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    with session.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()

        tmp = raw_path.with_suffix(raw_path.suffix + ".part")

        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        tmp.replace(raw_path)


def convert_to_wav(input_path: Path, output_wav: Path, sample_rate: int, overwrite: bool) -> None:
    if output_wav.exists() and not overwrite:
        return

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_wav),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])


def download_one_wrr(
    wrr_id: str,
    candidates: List[CandidateMediaFile],
    output_dir: Path,
    sample_rate: int,
    overwrite: bool,
) -> DownloadResult:
    candidate = choose_candidate(candidates)

    if not candidate:
        return DownloadResult(
            wrr_id=wrr_id,
            status="no_media_found",
            selected_type="",
            source_page="",
            file_page_url="",
            download_url="",
            output_wav="",
            error="",
        )

    selected_type = "audio" if candidate.is_audio else "video"
    session = make_session()

    try:
        download_url = extract_download_url(session, candidate.file_page_url)

        if not download_url:
            raise RuntimeError("No download link was found")

        wrr_dir = output_dir / wrr_id
        wrr_dir.mkdir(parents=True, exist_ok=True)

        raw_path = wrr_dir / f"{wrr_id}.download"
        output_wav = wrr_dir / f"{wrr_id}.wav"

        if not output_wav.exists() or overwrite:
            download_raw_file(session, download_url, raw_path)
            convert_to_wav(raw_path, output_wav, sample_rate, overwrite=True)

        try:
            raw_path.unlink()
        except Exception:
            pass

        return DownloadResult(
            wrr_id=wrr_id,
            status="ok",
            selected_type=selected_type,
            source_page=candidate.parent_url,
            file_page_url=candidate.file_page_url,
            download_url=download_url,
            output_wav=str(output_wav),
            error="",
        )

    except Exception as e:
        return DownloadResult(
            wrr_id=wrr_id,
            status="error",
            selected_type=selected_type,
            source_page=candidate.parent_url,
            file_page_url=candidate.file_page_url,
            download_url="",
            output_wav="",
            error=str(e),
        )


def save_results(results: List[DownloadResult], output_dir: Path) -> None:
    csv_path = output_dir / "download_summary.csv"
    json_path = output_dir / "download_summary.json"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wrr_id", "status", "selected_type", "source_page",
                "file_page_url", "download_url", "output_wav", "error"
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--txt-dir", required=True)
    parser.add_argument("--output-dir", default="./data/wave_files")
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--max-page", type=int, default=16)

    parser.add_argument("--scan-workers", type=int, default=20)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    check_ffmpeg()

    txt_dir = Path(args.txt_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT directory does not exist: {txt_dir}")

    target_wrr_ids = scan_local_wrr_ids(txt_dir)

    safe_print("=" * 80)
    safe_print(f"Local WRR ID count: {len(target_wrr_ids)}")
    safe_print(f"Download output directory: {output_dir}")
    safe_print("=" * 80)

    if not target_wrr_ids:
        safe_print("No local WRR IDs were found. Exiting.")
        return

    parent_pages = crawl_parent_pages(args.base_url, args.max_page, args.delay)

    safe_print(f"Unique SO page count: {len(parent_pages)}")
    safe_print("=" * 80)

    candidates_by_wrr: Dict[str, List[CandidateMediaFile]] = {}

    with ThreadPoolExecutor(max_workers=args.scan_workers) as executor:
        futures = {
            executor.submit(analyze_parent_page, p, target_wrr_ids, args.delay): p
            for p in parent_pages
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            page_matches = future.result()

            for wrr_id, items in page_matches.items():
                candidates_by_wrr.setdefault(wrr_id, []).extend(items)

            if idx % 10 == 0 or idx == len(futures):
                safe_print(f"[Scan progress] {idx}/{len(futures)}")

    results = []

    with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
        futures = {
            executor.submit(
                download_one_wrr,
                wrr_id,
                candidates_by_wrr.get(wrr_id, []),
                output_dir,
                args.sample_rate,
                args.overwrite,
            ): wrr_id
            for wrr_id in sorted(target_wrr_ids)
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)

            if result.status == "ok":
                safe_print(f"[OK] {result.wrr_id} -> {result.output_wav}")
            elif result.status == "no_media_found":
                safe_print(f"[NO MEDIA] {result.wrr_id}")
            else:
                safe_print(f"[ERROR] {result.wrr_id}: {result.error}")

            if idx % 10 == 0 or idx == len(futures):
                safe_print(f"[Download progress] {idx}/{len(futures)}")

    results.sort(key=lambda x: x.wrr_id)
    save_results(results, output_dir)

    ok = sum(1 for r in results if r.status == "ok")
    no_media = sum(1 for r in results if r.status == "no_media_found")
    err = sum(1 for r in results if r.status == "error")

    safe_print("=" * 80)
    safe_print("Download completed")
    safe_print(f"OK: {ok}")
    safe_print(f"No media found: {no_media}")
    safe_print(f"Errors: {err}")
    safe_print("=" * 80)


if __name__ == "__main__":
    main()