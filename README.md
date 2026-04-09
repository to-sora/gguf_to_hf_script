# GGUF to HF Scripts

Utilities for converting selected multimodal GGUF checkpoints into Hugging Face
repositories saved as `safetensors`.

The repo currently contains:

- `gguf_to_hf_qwen3_5_9B.py`: convert a Qwen3.5-9B GGUF checkpoint, with
  optional vision `mmproj`, into a `Qwen3_5ForConditionalGeneration` repo.
- `verify_hf_qwen.py`: load the converted Qwen repo and run text-generation
  sanity checks.
- `gguf_to_hf_gemma4E2B.py`: convert a Gemma-4-E2B GGUF checkpoint, with
  optional `mmproj`, into a `Gemma4ForConditionalGeneration` repo.
- `verify_hf_gemma4E2B.py`: load the converted Gemma repo and run text and
  optional image inference.

## Requirements

Install the Python packages used by the scripts:

```bash
pip install numpy torch safetensors transformers gguf Pillow
```

Notes:

- Use a Torch build that matches your CUDA setup if you want GPU conversion or
  verification.
- The scripts automatically fall back to CPU if CUDA is unavailable.
- The Qwen and Gemma verification scripts require a `transformers` build that
  includes `Qwen3_5ForConditionalGeneration` and
  `Gemma4ForConditionalGeneration`.

## Input Files

You will usually need:

- A text GGUF checkpoint: `--gguf`
- An `mmproj` GGUF checkpoint for multimodal weights: `--mmproj`
- A source Hugging Face model directory: `--src`

The source HF directory is used for config, tokenizer, processor, and related
metadata files. For Gemma, it is also used to copy missing buffers from
`model.safetensors`.

## Usage

### Qwen3.5-9B conversion

Use explicit paths. The script has local-machine defaults baked in, so relying
on its default arguments is usually wrong outside the original environment.

```bash
python gguf_to_hf_qwen3_5_9B.py \
  --gguf /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --src /path/to/source_hf_repo \
  --out /path/to/output_hf_repo \
  --shard-gb 4 \
  --device cuda
```

Text-only conversion:

```bash
python gguf_to_hf_qwen3_5_9B.py \
  --gguf /path/to/model.gguf \
  --mmproj "" \
  --src /path/to/source_hf_repo \
  --out /path/to/output_hf_repo \
  --device cpu
```

What it writes:

- Sharded `model-xxxxx-of-xxxxx.safetensors`
- `model.safetensors.index.json`
- `config.json`
- `generation_config.json`
- Tokenizer and processor files copied from `--src`

### Qwen3.5-9B verification

```bash
python verify_hf_qwen.py \
  --model /path/to/output_hf_repo \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-new-tokens 256
```

Useful flags:

- `--temperature 0.7` to enable sampling
- `--enable-thinking` to allow `<think>` output
- `--trust-remote-code` if your source repo requires it

### Gemma-4-E2B conversion

Full multimodal conversion:

```bash
python gguf_to_hf_gemma4E2B.py \
  --gguf /path/to/model.gguf \
  --mmproj /path/to/mmproj.gguf \
  --src /path/to/source_hf_repo \
  --out /path/to/output_hf_repo \
  --shard-gb 4 \
  --device cuda
```

Language-only conversion:

```bash
python gguf_to_hf_gemma4E2B.py \
  --gguf /path/to/model.gguf \
  --src /path/to/source_hf_repo \
  --out /path/to/output_hf_repo \
  --skip-mmproj \
  --device cpu
```

Notes:

- `--mmproj` is optional, but required if you want the vision/audio towers
  restored from GGUF.
- `--src` should contain the reference HF repo files, including
  `model.safetensors`, because the script merges buffers that are not present in
  GGUF.

### Gemma-4-E2B verification

Text-only:

```bash
python verify_hf_gemma4E2B.py \
  --model /path/to/output_hf_repo \
  --device cuda:0 \
  --dtype bfloat16
```

Text + image captioning:

```bash
python verify_hf_gemma4E2B.py \
  --model /path/to/output_hf_repo \
  --images /path/to/image1.jpg /path/to/image2.png \
  --text-prompt "where is Hong Kong" \
  --image-prompt "A short caption for this image:" \
  --device cuda:0
```

## Output Loading

Qwen:

```python
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

model_dir = "/path/to/output_hf_repo"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

Gemma:

```python
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration

model_dir = "/path/to/output_hf_repo"
processor = AutoProcessor.from_pretrained(model_dir)
model = Gemma4ForConditionalGeneration.from_pretrained(
    model_dir,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

## Practical Notes

- Large checkpoints may need substantial RAM, VRAM, and disk space during
  conversion.
- `--shard-gb` controls the maximum size of each written safetensors shard.
- The verification scripts are the fastest way to catch missing tensors,
  tokenizer/config mismatches, or broken generation after conversion.
