#!/usr/bin/env python3
"""
verify_hf_qwen.py  — Verify and test the converted Qwen3.5-9B HF model

Tests performed:
  1. Weight sanity  (norms, A_log, shapes) — checked on CPU, no extra VRAM
  2. Forward pass with full chat template
  3. Generation with correct eos_token_ids
  4. Logit diagnostic on canned prompts

Usage:
  python verify_hf.py [--model PATH]
"""

import argparse, torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

DEFAULT_MODEL = r"/mnt/DATA9/LLM_model/converted/Qwen3.5-9B-Uncensored-HF"
DEVICE = "cuda:0"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
args = ap.parse_args()
MODEL_PATH = args.model

# ── 1. load ────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"[1] Loading tokenizer & model from:\n    {MODEL_PATH}")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()
print("    Loaded OK.")

# ── 2. tokenizer / config sanity ───────────────────────────────────────────────
print("\n[2] Tokenizer / config sanity:")
tc = model.config.text_config  # nested text config on Qwen3_5Config
print(f"  eos_token      : {tok.eos_token!r}  id={tok.eos_token_id}")
print(f"  pad_token      : {tok.pad_token!r}  id={tok.pad_token_id}")
print(f"  config eos_id  : {tc.eos_token_id}")
print(f"  vocab_size     : {tc.vocab_size}")
print(f"  num_layers     : {tc.num_hidden_layers}")
print(f"  layer_types[:8]: {tc.layer_types[:8]}")

# ── 3. weight sanity (on CPU to avoid OOM) ────────────────────────────────────
print("\n[3] Weight sanity spot-check (CPU):")

# Expected: after correct conversion, Qwen3_5RMSNorm stores OFFSET from 1.
# forward: output * (1 + weight).  Init: weight = zeros → (1+0)=1.0 scale.
# After training: weight drifts.  Range for input_layernorm ≈ (-0.5, 0.5).
# q_norm / k_norm drift further but still should be roughly (-1.0, 1.5).
# model.norm (output norm) can drift further still — wider range allowed.
# linear_attn.norm uses Qwen3_5RMSNormGated: forward = weight * norm (no +1).
#   → weight initialized to 1.0, trained value ≈ (0.4, 1.5).

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
named_params  = dict(model.named_parameters())
named_buffers = dict(model.named_buffers())
for key, (check_type, expected) in checks.items():
    param = named_params.get(key) if key in named_params else named_buffers.get(key)
    if param is None:
        # try state_dict (loads to CPU view)
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

# ── 4. eos ids ─────────────────────────────────────────────────────────────────
# Chat models must stop at <|im_end|> (248046) and <|endoftext|> (248044)
EOS_IDS = sorted({tok.eos_token_id, tc.eos_token_id, 248046, 248044} - {None})
print(f"\n[4] eos_token_ids for generation: {EOS_IDS}")

# ── 5. chat generation ─────────────────────────────────────────────────────────
def chat(messages, max_new_tokens=1024, temperature=None, enable_thinking=False):
    """
    enable_thinking=False  → appends <think>\n\n</think> prefix so the model
    skips the reasoning chain and goes straight to the answer.
    enable_thinking=True   → lets the model reason (needs ~2000+ tokens).
    """
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not enable_thinking:
        # The chat template ends with "<think>\n"; close it immediately to suppress reasoning.
        # Do NOT prepend another <think>: that creates a nested unclosed tag.
        text += "\n</think>\n\n"
    inputs = tok(text, return_tensors="pt").to(DEVICE)
    n_prompt = inputs["input_ids"].shape[1]
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        eos_token_id=EOS_IDS,
        pad_token_id=tok.pad_token_id or EOS_IDS[0],
        do_sample=temperature is not None,
        repetition_penalty=1.1,   # prevents loops when torch fallback is used
    )
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tok.decode(out[0][n_prompt:], skip_special_tokens=True).strip()
    # Strip residual <think>...</think> blocks if thinking was enabled
    if enable_thinking:
        import re
        decoded = re.sub(r"<think>.*?</think>\s*", "", decoded, flags=re.DOTALL).strip()
    return decoded

print("\n[5] Chat generation tests (thinking suppressed, greedy):")
print("    Note: install flash-linear-attention + causal-conv1d for best quality")
print("-" * 40)
prompts = [
    [{"role": "user", "content": "What is 2 + 2?"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "Write one sentence describing the sky."}],
]
for msgs in prompts:
    q = msgs[0]["content"]
    print(f"  Q: {q}")
    resp = chat(msgs, max_new_tokens=256, enable_thinking=False)
    print(f"  A: {resp}")
    print()

# ── 6. logit diagnostic ────────────────────────────────────────────────────────
print("[6] Top-10 next tokens — raw 'Hello' (no template):")
with torch.no_grad():
    logits = model(**tok("Hello", return_tensors="pt").to(DEVICE)).logits[0, -1]
probs = logits.softmax(-1)
for p, idx in zip(*torch.topk(probs, 10)):
    print(f"    {idx.item():7d}  {tok.decode([idx.item()])!r:25s}  {p.item()*100:.2f}%")

print("\n[7] Top-10 next tokens — chat prefix '<|im_start|>assistant\\n':")
msg = [{"role": "user", "content": "Hello!"}]
text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
with torch.no_grad():
    logits2 = model(**tok(text, return_tensors="pt").to(DEVICE)).logits[0, -1]
probs2 = logits2.softmax(-1)
for p, idx in zip(*torch.topk(probs2, 10)):
    print(f"    {idx.item():7d}  {tok.decode([idx.item()])!r:25s}  {p.item()*100:.2f}%")

print("\n" + "=" * 60)
print("Verification complete.")
print("=" * 60)
