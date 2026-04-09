#!/usr/bin/env python3
"""Load the converted Gemma-4-E2B repo and run text + image inference.

Gemma-4-E2B is a base (pre-trained) model and ships with no chat template,
so this script uses plain continuation prompts — the standard HF pattern for
verifying a multimodal base model.  The image placeholder is the tokenizer's
own ``image_token`` (``<|image|>``), which ``Gemma4Processor`` expands into
the per-image soft-token span during ``proc(text=..., images=...)``.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to converted Gemma-4-E2B HF model directory",
    )
    ap.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="One or more image paths for captioning",
    )
    ap.add_argument(
        "--text-prompt",
        default="where is Hong Kong",
        help="Prompt for text-only generation",
    )
    ap.add_argument(
        "--image-prompt",
        default="A short caption for this image:",
        help="Prompt suffix used after the image token for image captioning",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device or device_map target, e.g. cuda:0 or cpu",
    )
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype",
    )
    ap.add_argument(
        "--max-new-tokens-text",
        type=int,
        default=60,
        help="Max new tokens for text-only generation",
    )
    ap.add_argument(
        "--max-new-tokens-image",
        type=int,
        default=80,
        help="Max new tokens for image generation/captioning",
    )
    ap.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling during generation",
    )
    ap.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty for generation",
    )
    return ap.parse_args()


def resolve_dtype(dtype_name: str):
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    torch_dtype = resolve_dtype(args.dtype)

    print("[1] Loading model + processor...")
    proc = AutoProcessor.from_pretrained(args.model)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch_dtype,
        device_map=args.device,
    ).eval()
    first_param = next(model.parameters())
    print(f"    Model loaded. dtype={model.dtype}, device={first_param.device}")

    image_token = proc.tokenizer.image_token

    print("\n[2] Text-only generation:")
    text_inputs = proc(text=[args.text_prompt], return_tensors="pt").to(args.device)
    with torch.no_grad():
        text_out = model.generate(
            **text_inputs,
            max_new_tokens=args.max_new_tokens_text,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
        )
    text_resp = proc.tokenizer.decode(
        text_out[0][text_inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    print(f"    {text_resp.strip()}")

    if not args.images:
        print("\n[3] Image captioning skipped (no --images provided)")
        return

    print("\n[3] Image captioning:")
    for img_path in args.images:
        prompt = f"{image_token} {args.image_prompt}"
        image_inputs = proc(text=[prompt], images=[img_path], return_tensors="pt").to(args.device)
        with torch.no_grad():
            image_out = model.generate(
                **image_inputs,
                max_new_tokens=args.max_new_tokens_image,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
            )
        image_resp = proc.tokenizer.decode(
            image_out[0][image_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"\n  === {img_path} ===")
        print(f"  {image_resp.strip()}")


if __name__ == "__main__":
    main()
