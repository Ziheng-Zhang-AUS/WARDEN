import re
import csv
import json
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


WRR_PATTERN = re.compile(r"wrr0\d{3,4}", re.IGNORECASE)


@dataclass
class AnnotationSegment:
    start: float
    end: float
    duration: float
    text: str


@dataclass
class PairedSegment:
    wrr_id: str
    start: float
    end: float
    duration: float
    tc_text: str
    ts_text: str


@dataclass
class OutputChunk:
    wrr_id: str
    split: str
    audio: str
    output_audio: str
    text: str
    duration: float
    segment_count: int


@dataclass
class SourceResult:
    wrr_id: str
    split: str
    status: str
    error: str
    tc_file: str
    ts_file: str
    source_audio: str
    paired_segment_count: int
    output_chunk_count: int
    total_duration: float
    chunks: List[OutputChunk]


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found. Please install it first: brew install ffmpeg")


def run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-3000:])


def parse_float(x: str) -> Optional[float]:
    try:
        return float(x.strip())
    except Exception:
        return None


def parse_annotation_file(path: Path) -> List[AnnotationSegment]:
    segments = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) < 7:
                continue

            start = parse_float(parts[1])
            end = parse_float(parts[3])
            duration = parse_float(parts[5])
            text = "\t".join(parts[6:]).strip()

            if start is None or end is None:
                continue

            if end <= start:
                continue

            if not text:
                continue

            if duration is None:
                duration = end - start

            segments.append(AnnotationSegment(start, end, duration, text))

    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def scan_txt_pairs(txt_dir: Path) -> Dict[str, Dict[str, Path]]:
    pairs: Dict[str, Dict[str, Path]] = {}

    for txt in txt_dir.rglob("*.txt"):
        name = txt.name.lower()
        match = WRR_PATTERN.search(name)

        if not match:
            continue

        wrr_id = match.group(0).lower()

        if name.endswith("_tc.txt"):
            pairs.setdefault(wrr_id, {})["tc"] = txt
        elif name.endswith("_ts.txt"):
            pairs.setdefault(wrr_id, {})["ts"] = txt

    return dict(sorted(pairs.items()))


def find_source_audio(wave_dir: Path, wrr_id: str) -> Optional[Path]:
    expected = wave_dir / wrr_id / f"{wrr_id}.wav"

    if expected.exists():
        return expected

    candidates = sorted(wave_dir.rglob(f"{wrr_id}.wav"))
    if candidates:
        return candidates[0]

    candidates = sorted(wave_dir.rglob(f"*{wrr_id}*.wav"))
    if candidates:
        return candidates[0]

    return None


def get_split(wrr_id: str) -> str:
    match = re.search(r"wrr0(\d{3,4})", wrr_id.lower())

    if not match:
        return "test"

    n = int(match.group(1))

    if n <= 357:
        return "train"

    if 358 <= n <= 370:
        return "val"

    return "test"


def get_audio_folder(split: str) -> str:
    if split == "val":
        return "validation"
    return split


def intersect_tc_ts(
    wrr_id: str,
    tc: List[AnnotationSegment],
    ts: List[AnnotationSegment],
    min_overlap: float,
) -> List[PairedSegment]:
    paired = []

    i = 0
    j = 0

    while i < len(tc) and j < len(ts):
        a = tc[i]
        b = ts[j]

        start = max(a.start, b.start)
        end = min(a.end, b.end)
        duration = end - start

        if duration >= min_overlap:
            paired.append(
                PairedSegment(
                    wrr_id=wrr_id,
                    start=start,
                    end=end,
                    duration=duration,
                    tc_text=a.text,
                    ts_text=b.text,
                )
            )

        if a.end <= b.end:
            i += 1
        else:
            j += 1

    return paired


def pack_segments(
    segments: List[PairedSegment],
    max_duration: float,
) -> List[List[PairedSegment]]:
    chunks = []
    current = []
    current_duration = 0.0

    for seg in segments:
        if seg.duration > max_duration:
            if current:
                chunks.append(current)
                current = []
                current_duration = 0.0

            start = seg.start

            while start < seg.end:
                end = min(start + max_duration, seg.end)

                sub = PairedSegment(
                    wrr_id=seg.wrr_id,
                    start=start,
                    end=end,
                    duration=end - start,
                    tc_text=seg.tc_text,
                    ts_text=seg.ts_text,
                )

                chunks.append([sub])
                start = end

            continue

        if current_duration + seg.duration <= max_duration:
            current.append(seg)
            current_duration += seg.duration
        else:
            if current:
                chunks.append(current)

            current = [seg]
            current_duration = seg.duration

    if current:
        chunks.append(current)

    return chunks


def cut_piece(
    input_audio: Path,
    output_audio: Path,
    start: float,
    duration: float,
    sample_rate: int,
) -> None:
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(input_audio),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_audio),
    ]

    run_cmd(cmd)


def concat_pieces(piece_files: List[Path], output_audio: Path) -> None:
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = Path(f.name)

        for p in piece_files:
            escaped = str(p).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_audio),
        ]
        run_cmd(cmd)
    finally:
        try:
            list_path.unlink()
        except Exception:
            pass


def build_text(chunk: List[PairedSegment], text_source: str) -> str:
    if text_source == "ts":
        return " ".join(seg.ts_text.strip() for seg in chunk if seg.ts_text.strip()).strip()

    return " ".join(seg.tc_text.strip() for seg in chunk if seg.tc_text.strip()).strip()


def create_chunk_audio(
    source_audio: Path,
    chunk: List[PairedSegment],
    output_audio: Path,
    tmp_dir: Path,
    sample_rate: int,
) -> float:
    pieces = []
    duration = 0.0

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for i, seg in enumerate(chunk, start=1):
        piece = tmp_dir / f"piece_{i:03d}.wav"
        cut_piece(source_audio, piece, seg.start, seg.duration, sample_rate)
        pieces.append(piece)
        duration += seg.duration

    if len(pieces) == 1:
        shutil.move(str(pieces[0]), str(output_audio))
    else:
        concat_pieces(pieces, output_audio)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return duration


def process_one_wrr(
    wrr_id: str,
    pair: Dict[str, Path],
    wave_dir: Path,
    output_dir: Path,
    max_duration: float,
    min_overlap: float,
    sample_rate: int,
    text_source: str,
) -> SourceResult:
    split = get_split(wrr_id)
    audio_folder = get_audio_folder(split)

    tc_file = pair.get("tc")
    ts_file = pair.get("ts")

    try:
        if not tc_file or not ts_file:
            raise RuntimeError("Missing _tc.txt or _ts.txt")

        source_audio = find_source_audio(wave_dir, wrr_id)

        if not source_audio:
            raise RuntimeError(f"Source audio was not found for {wrr_id}")

        tc_segments = parse_annotation_file(tc_file)
        ts_segments = parse_annotation_file(ts_file)

        paired = intersect_tc_ts(wrr_id, tc_segments, ts_segments, min_overlap)
        chunks = pack_segments(paired, max_duration)

        output_chunks = []

        audio_out_dir = output_dir / audio_folder
        tmp_root = output_dir / "_tmp" / wrr_id

        audio_out_dir.mkdir(parents=True, exist_ok=True)
        tmp_root.mkdir(parents=True, exist_ok=True)

        chunk_index = 1

        for chunk in chunks:
            text = build_text(chunk, text_source)

            if not text:
                continue

            filename = f"{wrr_id}_{chunk_index:02d}.wav"
            output_audio = audio_out_dir / filename
            audio_rel = f"{audio_folder}/{filename}"
            tmp_dir = tmp_root / f"{chunk_index:02d}"

            duration = create_chunk_audio(
                source_audio=source_audio,
                chunk=chunk,
                output_audio=output_audio,
                tmp_dir=tmp_dir,
                sample_rate=sample_rate,
            )

            output_chunks.append(
                OutputChunk(
                    wrr_id=wrr_id,
                    split=split,
                    audio=audio_rel,
                    output_audio=str(output_audio),
                    text=text,
                    duration=round(duration, 3),
                    segment_count=len(chunk),
                )
            )

            chunk_index += 1

        shutil.rmtree(tmp_root, ignore_errors=True)

        return SourceResult(
            wrr_id=wrr_id,
            split=split,
            status="ok",
            error="",
            tc_file=str(tc_file),
            ts_file=str(ts_file),
            source_audio=str(source_audio),
            paired_segment_count=len(paired),
            output_chunk_count=len(output_chunks),
            total_duration=round(sum(c.duration for c in output_chunks), 3),
            chunks=output_chunks,
        )

    except Exception as e:
        return SourceResult(
            wrr_id=wrr_id,
            split=split,
            status="error",
            error=str(e),
            tc_file=str(tc_file) if tc_file else "",
            ts_file=str(ts_file) if ts_file else "",
            source_audio="",
            paired_segment_count=0,
            output_chunk_count=0,
            total_duration=0.0,
            chunks=[],
        )


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder in ["train", "validation", "test"]:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)


def save_jsonl(results: List[SourceResult], output_dir: Path) -> None:
    records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for r in results:
        if r.status != "ok":
            continue

        if r.split not in records:
            continue

        for c in r.chunks:
            records[r.split].append({
                "audio": c.audio,
                "text": c.text,
            })

    file_map = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test": output_dir / "test.jsonl",
    }

    for split, path in file_map.items():
        with open(path, "w", encoding="utf-8") as f:
            for item in records[split]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_summary(results: List[SourceResult], output_dir: Path) -> None:
    summary_csv = output_dir / "summary_by_source.csv"
    chunks_csv = output_dir / "chunks.csv"
    full_json = output_dir / "summary_full.json"

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wrr_id", "split", "status", "error",
                "tc_file", "ts_file", "source_audio",
                "paired_segment_count", "output_chunk_count", "total_duration",
            ],
        )
        writer.writeheader()

        for r in results:
            writer.writerow({
                "wrr_id": r.wrr_id,
                "split": r.split,
                "status": r.status,
                "error": r.error,
                "tc_file": r.tc_file,
                "ts_file": r.ts_file,
                "source_audio": r.source_audio,
                "paired_segment_count": r.paired_segment_count,
                "output_chunk_count": r.output_chunk_count,
                "total_duration": r.total_duration,
            })

    with open(chunks_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "wrr_id", "split", "audio", "output_audio",
                "text", "duration", "segment_count",
            ],
        )
        writer.writeheader()

        for r in results:
            for c in r.chunks:
                writer.writerow(asdict(c))

    with open(full_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--txt-dir", required=True)
    parser.add_argument("--wave-dir", default="./data/wave_files")
    parser.add_argument("--output-dir", default="./data/transcribe")

    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--min-overlap", type=float, default=0.05)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=6)

    parser.add_argument(
        "--text-source",
        choices=["tc", "ts"],
        default="tc",
    )

    args = parser.parse_args()

    check_ffmpeg()

    txt_dir = Path(args.txt_dir).expanduser().resolve()
    wave_dir = Path(args.wave_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not txt_dir.exists():
        raise FileNotFoundError(f"TXT directory does not exist: {txt_dir}")

    if not wave_dir.exists():
        raise FileNotFoundError(f"Wave directory does not exist: {wave_dir}")

    ensure_output_dirs(output_dir)

    pairs = scan_txt_pairs(txt_dir)

    complete_pairs = {
        wrr_id: pair
        for wrr_id, pair in pairs.items()
        if "tc" in pair and "ts" in pair
    }

    print("=" * 80)
    print(f"TXT directory: {txt_dir}")
    print(f"Wave directory: {wave_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Detected WRR IDs: {len(pairs)}")
    print(f"Complete tc/ts pairs: {len(complete_pairs)}")
    print("=" * 80)

    results = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_one_wrr,
                wrr_id,
                pair,
                wave_dir,
                output_dir,
                args.max_duration,
                args.min_overlap,
                args.sample_rate,
                args.text_source,
            ): wrr_id
            for wrr_id, pair in complete_pairs.items()
        }

        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)

            if result.status == "ok":
                print(
                    f"[OK] {result.wrr_id} | {result.split} | "
                    f"chunks={result.output_chunk_count}"
                )
            else:
                print(f"[ERROR] {result.wrr_id} | {result.error}")

            if idx % 10 == 0 or idx == len(futures):
                print(f"[Progress] {idx}/{len(futures)}")

    results.sort(key=lambda r: r.wrr_id)

    save_jsonl(results, output_dir)
    save_summary(results, output_dir)

    ok = [r for r in results if r.status == "ok"]
    err = [r for r in results if r.status != "ok"]

    split_stats = {
        "train": {"sources": 0, "chunks": 0, "duration": 0.0},
        "val": {"sources": 0, "chunks": 0, "duration": 0.0},
        "test": {"sources": 0, "chunks": 0, "duration": 0.0},
    }

    for r in ok:
        split_stats[r.split]["sources"] += 1
        split_stats[r.split]["chunks"] += r.output_chunk_count
        split_stats[r.split]["duration"] += r.total_duration

    print("=" * 80)
    print("Build completed")
    print(f"Successful sources: {len(ok)}")
    print(f"Failed sources: {len(err)}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    for split, stat in split_stats.items():
        print(
            f"{split}: "
            f"sources={stat['sources']}, "
            f"chunks={stat['chunks']}, "
            f"duration={stat['duration']:.2f}s"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()