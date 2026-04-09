#!/usr/bin/env python3
"""
verify_hf_qwen.py  — Verify and test the converted Qwen3.5-9B HF model

Tests performed:
  1. Weight sanity  (norms, A_log, shapes) — checked on CPU, no extra VRAM
  2. Forward pass with full chat template
  3. Generation with correct eos_token_ids
  4. Logit diagnostic on canned prompts

Usage:
  python verify_hf.py --model PATH [--device DEV] [--dtype DTYPE]
                      [--max-new-tokens N] [--temperature T]
                      [--enable-thinking]
"""

import argparse
import re

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to converted HF model")
    ap.add_argument("--device", default="cuda:0", help="Torch device / device_map target")
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model load dtype",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for chat generation tests",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature; omit for greedy decoding",
    )
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow the model to emit <think> reasoning instead of suppressing it",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loader",
    )
    ap.add_argument(
        "--raw-prompt",
        default="Hello",
        help="Prompt for raw next-token diagnostic",
    )
    ap.add_argument(
        "--chat-user-prompt",
        default="Hello!",
        help="User prompt for chat-prefix next-token diagnostic",
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
    model_dtype = resolve_dtype(args.dtype)
    model_path = args.model
    device = args.device

    print("=" * 60)
    print(f"[1] Loading tokenizer & model from:\n    {model_path}")
    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=model_dtype,
        device_map=device,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print("    Loaded OK.")

    print("\n[2] Tokenizer / config sanity:")
    tc = model.config.text_config
    print(f"  eos_token      : {tok.eos_token!r}  id={tok.eos_token_id}")
    print(f"  pad_token      : {tok.pad_token!r}  id={tok.pad_token_id}")
    print(f"  config eos_id  : {tc.eos_token_id}")
    print(f"  vocab_size     : {tc.vocab_size}")
    print(f"  num_layers     : {tc.num_hidden_layers}")
    print(f"  layer_types[:8]: {tc.layer_types[:8]}")

    print("\n[3] Weight sanity spot-check (CPU):")

    LM = "model.language_model"
    checks = {
        f"{LM}.layers.0.input_layernorm.weight":     ("range", (-0.5,  0.5)),
        f"{LM}.norm.weight":                         ("range", (-0.5,  2.5)),
        f"{LM}.layers.0.linear_attn.norm.weight":    ("range", ( 0.3,  1.5)),
        f"{LM}.layers.0.linear_attn.A_log":          ("all_neg", None),
        f"{LM}.layers.0.linear_attn.conv1d.weight":  ("shape", (8192, 1, 4)),
        f"{LM}.layers.11.self_attn.q_norm.weight":   ("range", (-1.0,  1.5)),
        f"{LM}.layers.11.self_attn.k_norm.weight":   ("range", (-1.0,  1.5)),
        f"{LM}.embed_tokens.weight":                 ("shape", (248320, 4096)),
        "lm_head.weight":                            ("shape", (248320, 4096)),
    }

    all_pass = True
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    for key, (check_type, expected) in checks.items():
        param = named_params.get(key) if key in named_params else named_buffers.get(key)
        if param is None:
            sd_val = model.state_dict().get(key)
            if sd_val is None:
                print(f"  MISSING  {key}")
                all_pass = False
                continue
            t = sd_val.cpu().float()
        else:
            t = param.detach().cpu().float()

        if check_type == "shape":
            ok = tuple(t.shape) == expected
            print(f"  {'OK' if ok else 'FAIL':4s}  {key}: shape={tuple(t.shape)}")
        elif check_type == "range":
            lo, hi = expected
            m = t.mean().item()
            ok = lo < m < hi
            print(f"  {'OK' if ok else 'FAIL':4s}  {key}: mean={m:.4f}  (expect {lo}..{hi})")
        elif check_type == "all_neg":
            ok = bool((t < 0).all())
            print(f"  {'OK' if ok else 'FAIL':4s}  {key}: all_negative={ok}  min={t.min():.4f}  max={t.max():.4f}")
        if not ok:
            all_pass = False

    print(f"\n  Weight sanity: {'PASS ✓' if all_pass else 'FAIL ✗'}")

    EOS_IDS = sorted({tok.eos_token_id, tc.eos_token_id, 248046, 248044} - {None})
    print(f"\n[4] eos_token_ids for generation: {EOS_IDS}")

    def chat(messages, max_new_tokens=1024, temperature=None, enable_thinking=False):
        """
        enable_thinking=False  → appends <think>\n\n</think> prefix so the model
        skips the reasoning chain and goes straight to the answer.
        enable_thinking=True   → lets the model reason (needs ~2000+ tokens).
        """
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not enable_thinking:
            text += "\n</think>\n\n"
        inputs = tok(text, return_tensors="pt").to(device)
        n_prompt = inputs["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=EOS_IDS,
            pad_token_id=tok.pad_token_id or EOS_IDS[0],
            do_sample=temperature is not None,
            repetition_penalty=1.1,
        )
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        decoded = tok.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
        if enable_thinking:
            decoded = re.sub(r"<think>.*?</think>\s*", "", decoded, flags=re.DOTALL).strip()
        return decoded

    print("\n[5] Chat generation tests:")
    print(f"    thinking={'enabled' if args.enable_thinking else 'suppressed'}")
    print("-" * 40)
    prompts = [
        [{"role": "user", "content": "What is 2 + 2?"}],
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "Write one sentence describing the sky."}],
    ]
    for msgs in prompts:
        q = msgs[0]["content"]
        print(f"  Q: {q}")
        resp = chat(
            msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            enable_thinking=args.enable_thinking,
        )
        print(f"  A: {resp}")
        print()

    print(f"[6] Top-10 next tokens — raw {args.raw_prompt!r} (no template):")
    with torch.no_grad():
        logits = model(**tok(args.raw_prompt, return_tensors="pt").to(device)).logits[0, -1]
    probs = logits.softmax(-1)
    for p, idx in zip(*torch.topk(probs, 10)):
        print(f"    {idx.item():7d}  {tok.decode([idx.item()])!r:25s}  {p.item()*100:.2f}%")

    print(f"\n[7] Top-10 next tokens — chat prefix for user prompt {args.chat_user_prompt!r}:")
    msg = [{"role": "user", "content": args.chat_user_prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    with torch.no_grad():
        logits2 = model(**tok(text, return_tensors="pt").to(device)).logits[0, -1]
    probs2 = logits2.softmax(-1)
    for p, idx in zip(*torch.topk(probs2, 10)):
        print(f"    {idx.item():7d}  {tok.decode([idx.item()])!r:25s}  {p.item()*100:.2f}%")

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
