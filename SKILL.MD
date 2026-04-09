---
name: gguf-to-hf
description: Convert a GGUF model file back to HuggingFace safetensors format for fine-tuning. Use when the user wants to turn a .gguf checkpoint into a loadable HF repo, especially for hybrid/SSM/linear-attention architectures (Qwen3.5, Qwen3Next, Mamba, Jamba, etc.) where naive name-mapping is not enough.
---

# GGUF → HF Conversion

Converting GGUF back to HF is **not** symmetric to HF→GGUF. llama.cpp's `convert_hf_to_gguf.py` applies several lossy-looking but reversible transforms, and some of those transforms are **not self-inverse**. Getting this wrong produces a model whose weights look numerically plausible and pass shape checks, but whose generation loops or produces gibberish.

**Different architectures need different reverse transforms. Never copy a previous conversion script blindly — always re-derive from the current llama.cpp source.**

## Mandatory workflow

Before writing any conversion code for a new architecture, read these three sources in order. Do not skip any.

1. **`llama.cpp/convert_hf_to_gguf.py`** — find the `ModelBase.register(...)` decorator for the target architecture, then read its `modify_tensors()` method and any parent class chain. Every transform you see here must be reversed.

2. **`llama.cpp/src/llama-model.cpp`** — find the tensor loading block for the architecture (search for the arch enum, e.g. `LLM_ARCH_QWEN35`). This tells you the GGUF tensor shapes in `ne = {...}` order. Remember: **GGUFReader reverses `ne`**, so `ne = {A, B}` becomes numpy shape `(B, A)`.

3. **`llama.cpp/src/models/<arch>.cpp`** — the inference graph. This is where subtle layout assumptions live (Q/gate interleaving, tiled vs interleaved head expansion, conv kernel direction, norm formulas). If the HF forward pass does something structurally different, a raw weight copy will be wrong.

4. **HF `modeling_<arch>.py`** — read the matching module's `__init__` (weight names and shapes) and `forward` (how each weight is actually used). Compare against the llama.cpp inference graph.

## Known-nasty transforms (check every one for your arch)

These are transforms I have personally been burned by. Each one is a candidate bug on any hybrid/SSM architecture.

### 1. V-head reorder (`_LinearAttentionVReorderBase`) — **NOT self-inverse**

For models where `num_k_heads != num_v_heads` (Qwen3.5, Qwen3Next), the HF→GGUF converter reorders V heads from grouped `[K0v0,K0v1,K1v0,K1v1,...]` to tiled `[K0v0,K1v0,...,K15v0,K0v1,...]`.

The forward reshape is `(num_k_heads, num_v_per_k, head_dim)` then swap axes. **The inverse must reshape with `(num_v_per_k, num_k_heads, head_dim)`** — applying the same function twice does **not** return to the original when `num_k_heads != num_v_per_k`.

```python
def reorder_v_heads_inverse(tensor, dim, head_dim, num_k_heads, num_v_per_k):
    """GGUF tiled → HF grouped. NOT the same as forward."""
    shape = list(tensor.shape)
    if dim < 0: dim += len(shape)
    new_shape = shape[:dim] + [num_v_per_k, num_k_heads, head_dim] + shape[dim+1:]
    t = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim+1] = perm[dim+1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)
```

Applies to: `in_proj_qkv` (V rows only), `in_proj_z`, `in_proj_b`, `in_proj_a`, `A_log`, `dt_bias`, `conv1d` (V channels only), `out_proj` (columns).

### 2. RMSNorm `+1` encoding

Most Qwen3.5/Qwen3Next RMSNorm layers are stored in GGUF as `hf_weight + 1` because HF uses `output = x * (1 + weight)` with `weight` initialized to zeros. Reverse: subtract 1.

**Exceptions:** `linear_attn.norm.weight` (Qwen3_5RMSNormGated) is stored as-is because HF initializes it to `ones` and computes `output = weight * RMSNorm(x) * silu(gate)` directly. The llama.cpp converter explicitly excludes `linear_attn.norm.weight` from the `+1` rule — your reverse must exclude it too.

### 3. `A_log` → `-exp(A_log)`

GGUF stores `ssm_a = -exp(A_log)`. Reverse: `A_log = log(-ssm_a)`. Sanity check: the result must be finite and all `ssm_a` values must be strictly negative.

### 4. `conv1d` squeezed

HF shape is `(channels, 1, kernel_size)`. GGUF stores `(channels, kernel_size)`. Reverse: `tensor.unsqueeze(1)`.

Also verify kernel direction: in both `ggml_ssm_conv` and torch `F.conv1d` with `cat([conv_state, hidden_states])`, index 0 = oldest, index K-1 = newest. Usually matches without extra flip, but **verify** for any new architecture.

### 5. `dt_bias` rename

GGUF converter renames `.dt_bias` → `.dt_proj.bias` internally. The final GGUF tensor name may be `ssm_dt.bias` or just `ssm_dt`. Handle both.

### 6. Combined QKV / QKVZ projections

Some versions (Qwen3Next) store `in_proj_qkvz.weight` as a single tensor with per-k-head interleaving `[K0_Q, K0_K, K0_V, K0_Z, K1_Q, ...]`. The HF→GGUF converter splits it via a non-trivial `permute → view → split → view → cat → permute`. Qwen3.5 already uses separate `in_proj_qkv` and `in_proj_z`, but other architectures may not — **read the converter code for your specific arch**.

## Architecture class choice (HF side)

This matters for generation even if weights are perfect.

- **`*ForCausalLM`** — text-only wrapper. Default `_prepare_position_ids_for_generation` produces 2D position_ids.
- **`*ForConditionalGeneration`** — multimodal wrapper. Often overrides `_prepare_position_ids_for_generation` to produce **3D MROPE position_ids** for Qwen3.5/Qwen3Next/Qwen2.5-VL.

If the model uses MROPE (check `rope_parameters.mrope_section`), using the CausalLM class will produce correct **first-token** logits but subsequent generation steps get wrong position IDs and produce garbage. Match the native repo's architecture class exactly.

When using `*ForConditionalGeneration`:
- Prefix language weights with `model.language_model.*` instead of `model.*`
- Keep `lm_head.weight` at the top level
- Emit a nested config (`text_config` + `vision_config`) matching the native repo
- Vision weights will be randomly initialized — fine for text-only inference as long as user never passes `pixel_values`

## Tensor name/shape reference (general pattern)

Always grep the GGUF file first to get the actual tensor names:

```python
import gguf
reader = gguf.GGUFReader(path)
for t in reader.tensors[:50]:
    print(t.name, [d for d in t.shape])
```

Common GGUF → HF name mapping (Qwen3.5 family — **adapt for other archs**):

| GGUF                         | HF (text-only)                       |
|------------------------------|---------------------------------------|
| `token_embd.weight`          | `model.embed_tokens.weight`           |
| `output.weight`              | `lm_head.weight`                      |
| `output_norm.weight`         | `model.norm.weight` (− 1)             |
| `blk.{i}.attn_norm.weight`   | `...input_layernorm.weight` (− 1)     |
| `blk.{i}.post_attention_norm.weight` | `...post_attention_layernorm.weight` (− 1) |
| `blk.{i}.ffn_gate/up/down`   | `...mlp.{gate,up,down}_proj.weight`   |
| `blk.{i}.attn_qkv.weight`    | `...linear_attn.in_proj_qkv.weight` (V reorder) |
| `blk.{i}.attn_gate.weight`   | `...linear_attn.in_proj_z.weight` (V reorder) |
| `blk.{i}.ssm_beta.weight`    | `...linear_attn.in_proj_b.weight` (V reorder) |
| `blk.{i}.ssm_alpha.weight`   | `...linear_attn.in_proj_a.weight` (V reorder) |
| `blk.{i}.ssm_conv1d.weight`  | `...linear_attn.conv1d.weight` (unsqueeze + V reorder on V part) |
| `blk.{i}.ssm_dt` or `ssm_dt.bias` | `...linear_attn.dt_bias` (V reorder) |
| `blk.{i}.ssm_a`              | `...linear_attn.A_log` (V reorder, then `log(-x)`) |
| `blk.{i}.ssm_norm.weight`    | `...linear_attn.norm.weight` (no offset) |
| `blk.{i}.ssm_out.weight`     | `...linear_attn.out_proj.weight` (V reorder on columns) |
| `blk.{i}.attn_q/k/v/output`  | `...self_attn.{q,k,v,o}_proj.weight` |
| `blk.{i}.attn_q_norm/k_norm` | `...self_attn.{q,k}_norm.weight` (− 1) |

Skip `mtp.*` tensors (multi-token prediction head, usually unused).

## GGUF dtype handling

BF16 is stored as raw `uint8` bytes with the last numpy dim doubled:
```python
if raw.dtype == np.uint8:
    buf = np.frombuffer(raw.tobytes(), dtype=np.uint16)
    t = torch.from_numpy(buf.copy()).view(torch.bfloat16)
    t = t.reshape(raw.shape[:-1] + (raw.shape[-1] // 2,))
```
Float32 tensors (norms, `ssm_a`, `ssm_dt`) come through as normal numpy arrays.

## Verification protocol — do ALL of these

Weight sanity checks (shapes, ranges) are **not sufficient**. The V-head reorder bug passes every static check. You must run actual generation.

1. **Shape spot check** — every expected key present with expected shape.
2. **Norm value ranges** — `input_layernorm` mean ≈ 0, `ssm_norm` mean ≈ 1, `A_log` all finite and stored as `log(-x)` (so recovered is negative finite).
3. **Single-token logits** — tokenize "Hello", take raw forward, check top-10 tokens are sensible (e.g. `,`, ` world`). This verifies the prefill path.
4. **Multi-token greedy generation** — this is the real test:
   ```python
   inputs = tok("write me a hello world", return_tensors="pt").to("cuda:0")
   out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
   print(tok.decode(out[0], skip_special_tokens=True))
   ```
   If this loops (`"me a me a me a..."`) or produces unrelated text while single-token prediction was fine, you have a **state/cache bug**, almost certainly in a recurrent-state-affecting weight (V-head reorder, A_log sign, conv1d direction, `dt_bias`).
5. **Cross-check against llama.cpp** — run the same prompt through `llama-cli` on the original GGUF. Output should be similar (not identical, because sampling differs, but structurally plausible).

## Common failure modes and their fingerprints

| Symptom | Likely cause |
|---------|-------------|
| Shapes pass, first-token logits plausible, generation loops immediately | **V-head reorder wrong direction** (non-self-inverse bug) |
| Generation coherent for 3-10 tokens then diverges to loops | MROPE position_ids bug → using `*ForCausalLM` when native is `*ForConditionalGeneration` |
| All norms look too big or too small uniformly | `+1` offset applied wrong (missed exception for `linear_attn.norm.weight`, or forgot to subtract) |
| `NaN`/`Inf` in A_log | Forgot `log(-x)`, or `ssm_a` had non-negative values (ordering bug upstream) |
| First forward produces garbage logits | Conv1d squeeze/unsqueeze wrong, or Q/gate interleaving misunderstood for full-attention layers |
| `RuntimeError: shape mismatch` in `in_proj_qkvz` | Architecture uses combined QKVZ; you need the split-and-reshape that matches llama.cpp's forward |

## Reference implementation

A working converter for Qwen3.5-9B lives at `/home/AI_trainer/Desktop/QWEN_project/gguf_to_hf.py` and the matching verifier at `verify_hf.py`. Use these as a **template**, not a copy-paste: every new architecture needs fresh reading of `convert_hf_to_gguf.py` and `modeling_*.py`.

## Minimum checklist before declaring conversion done

- [ ] Read `modify_tensors()` for the arch in `convert_hf_to_gguf.py` and reversed every transform
- [ ] Read `src/models/<arch>.cpp` inference graph and cross-checked layout assumptions vs HF `forward`
- [ ] Matched the native repo's architecture class (`ForCausalLM` vs `ForConditionalGeneration`)
- [ ] Emitted a config with the same structure (flat vs nested `text_config`) as the native repo
- [ ] Copied tokenizer files from the source HF repo
- [ ] Multi-token `model.generate(..., do_sample=False, max_new_tokens=200)` produces coherent output on a trivial prompt
- [ ] Output is structurally similar to `llama-cli` on the same GGUF
