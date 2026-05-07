# WARDEN

This repository implements a two-stage pipeline:

1. **ASR fine-tuning** using Whisper for Wardaman transcription
2. **Translation fine-tuning** using Qwen with LoRA via LLaMAFactory

---

For data privacy reasons, the complete Wardaman data is available [here](http://hdl.handle.net/2196/884f9353-ea4c-4686-b83c-18cdb828193z).

# **1. ASR (Whisper Fine-tuning)**

## **Environment Setup**

```
conda create -n whisper_release python=3.10
conda activate whisper_release
pip install -r asr/requirements.txt
```

## **Data Format**

The directory structure must be:

```
data/transcribe/
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── train/
├── validation/
└── test/
```

Each JSON line file should follow this format:

```
{ "audio": "filename.wav", "text": "transcription" }
```

## Training

```bash
python asr/train_whisper.py \
  --data_dir data/transcribe \
  --model_name openai/whisper-medium \
  --language su \
  --output_dir results/whisper_medium \
  --max_steps 300
```

## **Output**

Fine-tuned Whisper checkpoints are saved to:

```
result/
```

## **Checkpoint**

The fine-tuned checkpoint for ASR can be found [here](https://huggingface.co/ZihengZhang/WARDEN-Whisper).

# **2. Lexicon Retrieval & Injection**

This module augments ASR transcription using a Wardaman-English lexicon. The original PDF of the Wardaman-English dictionary can be found at [link](https://www.elararchive.org/uncategorized/IO_d388196b-78ca-48ce-8eb9-86e31597bba0/).

It performs:

- Exact match
- CER-based fuzzy retrieval
- Optional affix matching
- Top-K filtering
- Structured injection formatting

## **Dictionary Format**

A cleaned lexicon file is provided at:

```text

data/lexicon/lexicon.csv

```

This file is used for information injection during the transcription-translation stage. More than 2,000 lexicon entries were manually cleaned and some of the content was transcribed into natural language to make it easier for large models to understand their meaning more intuitively.

Lexicon must be a CSV file, required columns:

```

lexical_unit, variant, pos, gloss

```

## **Injection Input Format**

Input JSONL (from ASR output):

```

{ "text": "transcription sentence" }

```

## **Run Lexicon Injection**

```

python lexicon/lexinject.py \
 --input data/transcribe/train.jsonl \
 --output data/translate/train_with_lexicon.jsonl \
 --dict cleaned_lexicon.csv \
 --top_k 2 \
 --cer_threshold 0.2 \
 --output_mode flat_json

```

## **Output**

Each sample will include an additional field:

```

{
"text": "...",
"lexicon": {
"word1": ["gloss1", "gloss2"],
"word2": ["gloss3"]
}
}

```

## **Statistics**

During injection, the script reports:

- Average entries per word (before top-k)
- Average entries per word (after top-k)
- Exact match ratio

These metrics are useful for:

- Ablation studies
- Zero-shot experiments
- Injection density control

# **3. Translation (Qwen + LoRA via LLaMAFactory)**

## **Environment Setup**

```python
conda create -n llama_release python=3.10
conda activate llama_release
pip install -r translation/requirements.txt
```

## **Dataset Format**

Place translation data in:

```
data/translate/
```

A required dataset_info.json example:

```
{
  "demo": {
    "file_name": "demo.json",
    "formatting": "sharegpt",
    "columns": { "messages": "messages" },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }
  }
}
```

Each training example should follow:

```
{
 "messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
 ]
}
```

## **Training**

```
llamafactory-cli train translation/configs/qwen_sft.yaml
```

## **Checkpoint**

The fine-tuned checkpoint for translating can be found [here](https://huggingface.co/ZihengZhang/WARDEN-Qwen3).
