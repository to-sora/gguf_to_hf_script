#!/usr/bin/env python3
"""
gguf_to_hf_qwen3_5_9b.py
Convert Qwen3.5-9B GGUF → HuggingFace safetensors (Qwen3_5ForConditionalGeneration)

Architecture: hybrid SSM+Attention (Mamba-style linear attn every 3/4 layers,
full attention on every 4th layer starting at index 3).

Reverse transforms applied (from llama.cpp convert_hf_to_gguf.py + _LinearAttentionVReorderBase):
  - norm weights stored as (hf + 1) in GGUF → subtract 1 (except linear_attn.norm)
  - ssm_a stored as -exp(A_log) → A_log = log(-ssm_a)
  - conv1d stored squeezed (8192,4) → unsqueeze(1) → (8192,1,4)
  - dt_bias renamed to dt_proj.bias in intermediate step → restore as dt_bias param
  - V-head reorder applied to: in_proj_qkv (V rows), in_proj_z (all rows),
    in_proj_b/a (all rows), out_proj (columns), conv1d (V channel), A_log/dt_bias (1-D)
  - V-head reorder is self-inverse (apply same function to reverse)

Usage:
  python gguf_to_hf.py [--gguf PATH] [--out DIR] [--shard-gb SIZE]
"""

import sys, os, json, shutil, argparse, struct
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/home/User/Desktop/QWEN_project")
DEFAULT_GGUF  = BASE / "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
DEFAULT_MMPROJ = BASE / "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf"
DEFAULT_SRC   = BASE / "Qwen3.5-9B"        # tokenizer + reference config
DEFAULT_OUT   = r'/mnt/LLM_model/converted/Qwen3.5-9B-Uncensored-HF'

# ── architecture constants (from config.json text_config) ──────────────────────
NUM_LAYERS          = 32
FULL_ATTN_INTERVAL  = 4           # full attention at layers 3,7,11,...
HIDDEN_SIZE         = 4096
NUM_K_HEADS_LIN     = 16          # linear_num_key_heads
NUM_V_HEADS_LIN     = 32          # linear_num_value_heads
HEAD_K_DIM          = 128         # linear_key_head_dim
HEAD_V_DIM          = 128         # linear_value_head_dim
NUM_V_PER_K         = NUM_V_HEADS_LIN // NUM_K_HEADS_LIN   # = 2

LAYER_TYPES = [
    "full_attention" if (i + 1) % FULL_ATTN_INTERVAL == 0 else "linear_attention"
    for i in range(NUM_LAYERS)
]

# ── helpers ────────────────────────────────────────────────────────────────────

def gguf_bf16_to_torch(data: np.ndarray) -> torch.Tensor:
    """Convert raw uint8 BF16 bytes (numpy) → torch.bfloat16 tensor."""
    if data.dtype != np.uint8:
        # Already float32 (e.g. conv1d, ssm_a, ssm_dt, norm weights)
        return torch.from_numpy(data.copy()).to(torch.float32)
    buf = np.frombuffer(data.tobytes(), dtype=np.uint16)
    t = torch.from_numpy(buf.copy()).view(torch.bfloat16)
    return t.reshape(data.shape[:-1] + (data.shape[-1] // 2,))


def load_tensor(t) -> torch.Tensor:
    """Read a GGUFReader tensor → torch tensor in HF-compatible layout."""
    raw = np.array(t.data, copy=False)
    if raw.dtype == np.uint8:
        # BF16 packed: last dim holds 2 bytes per element → halve it
        return gguf_bf16_to_torch(raw)
    return torch.from_numpy(raw.copy())


def reorder_v_heads(tensor: torch.Tensor, dim: int, head_dim: int) -> torch.Tensor:
    """
    Inverse of the GGUF converter's V-head reorder, i.e. convert tiled (GGUF)
    layout back to grouped (HF) layout along `dim`.

    GGUF tiled:   [K0_v0, K1_v0, ..., K15_v0, K0_v1, K1_v1, ..., K15_v1]
    HF  grouped:  [K0_v0, K0_v1, K1_v0, K1_v1, ..., K15_v0, K15_v1]

    NOT self-inverse when num_v_per_k != num_k_heads — the reverse direction
    must reshape with (num_v_per_k, num_k_heads) ordering, then swap axes.
    """
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    # Reverse direction: reshape with NUM_V_PER_K outer, NUM_K_HEADS_LIN inner
    new_shape = shape[:dim] + [NUM_V_PER_K, NUM_K_HEADS_LIN, head_dim] + shape[dim + 1:]
    tensor = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return tensor.permute(*perm).contiguous().reshape(*shape)


# ── GGUF → HF tensor name mapping ─────────────────────────────────────────────

def convert_tensor(name: str, tensor: torch.Tensor, layer_idx: int | None):
    """
    Yield (hf_name, hf_tensor) pairs after applying reverse transforms.
    May yield 0 or more items (0 for ignored tensors).
    """
    # ── global tensors ──
    # NOTE: weights use `model.language_model.*` prefix to match
    # Qwen3_5ForConditionalGeneration (native repo architecture).
    if name == "token_embd.weight":
        yield "model.language_model.embed_tokens.weight", tensor
        return

    if name == "output.weight":
        yield "lm_head.weight", tensor
        return

    if name == "output_norm.weight":
        # norm stored as (hf + 1) in GGUF → subtract 1
        yield "model.language_model.norm.weight", tensor - 1.0
        return

    # ── skip mtp and any non-blk tensors ──
    if not name.startswith("blk."):
        return

    # ── per-block tensors ──
    i = layer_idx
    is_linear = (LAYER_TYPES[i] == "linear_attention")
    pfx = f"model.language_model.layers.{i}"

    # ── shared (all layers) ──
    if name == f"blk.{i}.attn_norm.weight":
        yield f"{pfx}.input_layernorm.weight", tensor - 1.0
        return

    if name == f"blk.{i}.post_attention_norm.weight":
        yield f"{pfx}.post_attention_layernorm.weight", tensor - 1.0
        return

    # ── MLP (all layers) ──
    if name == f"blk.{i}.ffn_gate.weight":
        yield f"{pfx}.mlp.gate_proj.weight", tensor
        return
    if name == f"blk.{i}.ffn_up.weight":
        yield f"{pfx}.mlp.up_proj.weight", tensor
        return
    if name == f"blk.{i}.ffn_down.weight":
        yield f"{pfx}.mlp.down_proj.weight", tensor
        return

    # ── linear attention layers ──
    if is_linear:
        if name == f"blk.{i}.attn_qkv.weight":
            # V rows are reorder-shuffled; reverse: apply reorder again
            q_dim = HEAD_K_DIM * NUM_K_HEADS_LIN      # 2048
            k_dim = HEAD_K_DIM * NUM_K_HEADS_LIN      # 2048
            q = tensor[:q_dim]
            k = tensor[q_dim:q_dim + k_dim]
            v = tensor[q_dim + k_dim:]
            v = reorder_v_heads(v, 0, HEAD_V_DIM)     # reverse (self-inverse)
            yield f"{pfx}.linear_attn.in_proj_qkv.weight", torch.cat([q, k, v], dim=0)
            return

        if name == f"blk.{i}.attn_gate.weight":
            # in_proj_z: all rows reordered
            t2 = reorder_v_heads(tensor, 0, HEAD_V_DIM)
            yield f"{pfx}.linear_attn.in_proj_z.weight", t2
            return

        if name == f"blk.{i}.ssm_beta.weight":
            # in_proj_b: rows reordered (head_dim=1 per V slot)
            t2 = reorder_v_heads(tensor, 0, 1)
            yield f"{pfx}.linear_attn.in_proj_b.weight", t2
            return

        if name == f"blk.{i}.ssm_alpha.weight":
            # in_proj_a: same as beta
            t2 = reorder_v_heads(tensor, 0, 1)
            yield f"{pfx}.linear_attn.in_proj_a.weight", t2
            return

        if name == f"blk.{i}.ssm_conv1d.weight":
            # stored squeezed (8192, 4); HF needs (8192, 1, 4)
            # V-channel portion was reordered → reverse
            qk_ch = HEAD_K_DIM * NUM_K_HEADS_LIN * 2   # 4096
            qk_part = tensor[:qk_ch]                   # (4096, 4)
            v_part  = tensor[qk_ch:]                   # (4096, 4)
            v_part  = reorder_v_heads(v_part, 0, HEAD_V_DIM)
            merged  = torch.cat([qk_part, v_part], dim=0)  # (8192, 4)
            yield f"{pfx}.linear_attn.conv1d.weight", merged.unsqueeze(1)
            return

        # dt_bias is stored as ssm_dt.bias (via dt_proj.bias rename in llama.cpp)
        if name in (f"blk.{i}.ssm_dt", f"blk.{i}.ssm_dt.bias"):
            # dt_bias: 1-D, V-head reordered (head_dim=1)
            t2 = reorder_v_heads(tensor.unsqueeze(-1), 0, 1).squeeze(-1)
            yield f"{pfx}.linear_attn.dt_bias", t2
            return

        if name == f"blk.{i}.ssm_a":
            # stored as -exp(A_log), also V-head reordered
            # step 1: reverse reorder
            t2 = reorder_v_heads(tensor.unsqueeze(-1), 0, 1).squeeze(-1)
            # step 2: recover A_log = log(-t2)
            A_log = torch.log(-t2)
            yield f"{pfx}.linear_attn.A_log", A_log
            return

        if name == f"blk.{i}.ssm_norm.weight":
            # NOT offset-stored (excluded from +1 rule)
            yield f"{pfx}.linear_attn.norm.weight", tensor
            return

        if name == f"blk.{i}.ssm_out.weight":
            # out_proj: columns (input dim) are reordered
            t2 = reorder_v_heads(tensor, 1, HEAD_V_DIM)
            yield f"{pfx}.linear_attn.out_proj.weight", t2
            return

    # ── full attention layers ──
    else:
        if name == f"blk.{i}.attn_q.weight":
            yield f"{pfx}.self_attn.q_proj.weight", tensor
            return
        if name == f"blk.{i}.attn_k.weight":
            yield f"{pfx}.self_attn.k_proj.weight", tensor
            return
        if name == f"blk.{i}.attn_v.weight":
            yield f"{pfx}.self_attn.v_proj.weight", tensor
            return
        if name == f"blk.{i}.attn_output.weight":
            yield f"{pfx}.self_attn.o_proj.weight", tensor
            return
        if name == f"blk.{i}.attn_q_norm.weight":
            # same Qwen3_5RMSNorm forward: (1+weight)*norm → +1 was applied in HF→GGUF
            yield f"{pfx}.self_attn.q_norm.weight", tensor - 1.0
            return
        if name == f"blk.{i}.attn_k_norm.weight":
            yield f"{pfx}.self_attn.k_norm.weight", tensor - 1.0
            return


# ── vision (mmproj) conversion ────────────────────────────────────────────────
#
# All vision weight transforms are the inverse of llama.cpp's vision exporter:
#   class Qwen3VLVisionModel.modify_tensors(...)         in convert_hf_to_gguf.py
#   (registered with @ModelBase.register("Qwen3_5ForConditionalGeneration"))
#
# The HF target shapes/names come from:
#   class Qwen3_5VisionPatchEmbed.__init__               (modeling_qwen3_5.py)
#   class Qwen3_5VisionAttention.__init__/.forward
#   class Qwen3_5VisionMLP.__init__/.forward
#   class Qwen3_5VisionBlock.__init__/.forward
#   class Qwen3_5VisionPatchMerger.__init__/.forward
#   class Qwen3_5VisionModel.__init__
#
# (Locating by class/function signature, not line number, because line numbers
# drift between transformers releases.)

def convert_vision_tensor(name: str, tensor: torch.Tensor):
    """
    Yield (hf_name, hf_tensor) pairs for one mmproj GGUF tensor.

    GGUF linear-weight tensors are already stored in HF (out_features, in_features)
    layout (the BF16-uint8 packing only halves the last dim *byte-count*, so the
    logical shape after gguf_bf16_to_torch matches what HF expects directly —
    no transposes needed for vision linears).
    """
    # ── 1. Patch embed ────────────────────────────────────────────────────────
    # llama.cpp Qwen3VLVisionModel.modify_tensors splits the HF Conv3D weight
    # along the temporal axis:
    #     data[:, :, 0, :, :]  →  v.patch_embd.weight
    #     data[:, :, 1, :, :]  →  v.patch_embd.weight.1
    # The HF Conv3D kernel is (out=hidden, in=3, kt=temporal_patch_size=2,
    #                          kh=patch_size, kw=patch_size).
    # Inverse: torch.stack([w0, w1], dim=2) along the temporal axis.
    if name == "v.patch_embd.weight":
        return  # handled together with .1 below (we yield from the .1 branch)
    if name == "v.patch_embd.weight.1":
        return
    if name == "v.patch_embd.bias":
        # Qwen3_5VisionPatchEmbed.proj.bias — direct copy.
        yield "model.visual.patch_embed.proj.bias", tensor
        return

    # ── 2. Position embedding ─────────────────────────────────────────────────
    # llama.cpp keeps `visual.pos_embed.weight` as-is and renames it
    # `v.position_embd.weight`.  HF stores it as nn.Embedding
    # (num_position_embeddings, hidden_size) — same shape, just rename.
    # Reference: Qwen3_5VisionModel.__init__ → self.pos_embed = nn.Embedding(...)
    if name == "v.position_embd.weight":
        yield "model.visual.pos_embed.weight", tensor
        return

    # ── 3. Per-block weights ──────────────────────────────────────────────────
    # llama.cpp tensor template strings (gguf-py/gguf/constants.py):
    #     V_ENC_INPUT_NORM      = "v.blk.{bid}.ln1"
    #     V_ENC_POST_ATTN_NORM  = "v.blk.{bid}.ln2"
    #     V_ENC_ATTN_QKV        = "v.blk.{bid}.attn_qkv"
    #     V_ENC_ATTN_O          = "v.blk.{bid}.attn_out"
    #     V_ENC_FFN_UP          = "v.blk.{bid}.ffn_up"
    #     V_ENC_FFN_DOWN        = "v.blk.{bid}.ffn_down"
    # HF block layout: Qwen3_5VisionBlock.__init__:
    #     self.norm1 / self.norm2 (LayerNorm)
    #     self.attn  = Qwen3_5VisionAttention()  → self.qkv, self.proj
    #     self.mlp   = Qwen3_5VisionMLP()        → self.linear_fc1, self.linear_fc2
    if name.startswith("v.blk."):
        parts = name.split(".")           # ['v','blk','{i}','<sub>','weight'/'bias']
        bid = int(parts[2])
        sub = ".".join(parts[3:-1])       # e.g. 'attn_qkv', 'ffn_up'
        suffix = parts[-1]                # 'weight' or 'bias'
        pfx = f"model.visual.blocks.{bid}"

        gguf_to_hf = {
            "ln1":      f"{pfx}.norm1",
            "ln2":      f"{pfx}.norm2",
            "attn_qkv": f"{pfx}.attn.qkv",
            "attn_out": f"{pfx}.attn.proj",
            "ffn_up":   f"{pfx}.mlp.linear_fc1",
            "ffn_down": f"{pfx}.mlp.linear_fc2",
        }
        if sub in gguf_to_hf:
            yield f"{gguf_to_hf[sub]}.{suffix}", tensor
            return

    # ── 4. Merger (patch merger MLP) ──────────────────────────────────────────
    # llama.cpp Qwen3VLVisionModel.modify_tensors maps:
    #     visual.merger.linear_fc1.{w,b}   →  mm.0.{w,b}
    #     visual.merger.linear_fc2.{w,b}   →  mm.2.{w,b}
    #     visual.merger.norm.{w,b}         →  v.post_ln.{w,b}   (note: NOT a
    #         "post-encoder norm" — despite the GGUF name, it is the LayerNorm
    #         INSIDE Qwen3_5VisionPatchMerger.__init__: self.norm = nn.LayerNorm)
    # HF reference: Qwen3_5VisionPatchMerger.__init__/.forward
    if name == "mm.0.weight":
        yield "model.visual.merger.linear_fc1.weight", tensor
        return
    if name == "mm.0.bias":
        yield "model.visual.merger.linear_fc1.bias", tensor
        return
    if name == "mm.2.weight":
        yield "model.visual.merger.linear_fc2.weight", tensor
        return
    if name == "mm.2.bias":
        yield "model.visual.merger.linear_fc2.bias", tensor
        return
    if name == "v.post_ln.weight":
        yield "model.visual.merger.norm.weight", tensor
        return
    if name == "v.post_ln.bias":
        yield "model.visual.merger.norm.bias", tensor
        return


def vision_pass(reader, device: torch.device) -> dict[str, torch.Tensor]:
    """Read all mmproj tensors and return an HF state dict for model.visual.*."""
    out: dict[str, torch.Tensor] = {}
    patch_halves: dict[int, torch.Tensor] = {}

    for t in reader.tensors:
        name = t.name
        raw = load_tensor(t).to(device)

        # Patch embed Conv3D reassembly: stack the two temporal halves.
        # llama.cpp split direction (Qwen3VLVisionModel.modify_tensors):
        #     w0 = data[:, :, 0, :, :]   → "v.patch_embd.weight"
        #     w1 = data[:, :, 1, :, :]   → "v.patch_embd.weight.1"
        # so we restore with stack(dim=2): order [w0, w1] preserves temporal index.
        if name == "v.patch_embd.weight":
            patch_halves[0] = raw
            continue
        if name == "v.patch_embd.weight.1":
            patch_halves[1] = raw
            continue

        for hf_name, hf_tensor in convert_vision_tensor(name, raw):
            out[hf_name] = hf_tensor

    if 0 in patch_halves and 1 in patch_halves:
        w0 = patch_halves[0]                 # (1152, 3, 16, 16)
        w1 = patch_halves[1]                 # (1152, 3, 16, 16)
        # HF Conv3D weight shape: (out, in, kt, kh, kw) — kt=2 here.
        proj = torch.stack([w0, w1], dim=2)  # (1152, 3, 2, 16, 16)
        out["model.visual.patch_embed.proj.weight"] = proj

    return out


def parse_layer_idx(name: str) -> int | None:
    """Extract block index from 'blk.{i}.*' name."""
    if not name.startswith("blk."):
        return None
    try:
        return int(name.split(".")[1])
    except (IndexError, ValueError):
        return None


# ── sharding ───────────────────────────────────────────────────────────────────

def shard_and_save(tensors: dict[str, torch.Tensor], out_dir: Path, max_shard_gb: float):
    """Save tensors as sharded safetensors + index json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_shard_gb * 1024**3)

    shards: list[dict[str, torch.Tensor]] = []
    cur_shard: dict[str, torch.Tensor] = {}
    cur_bytes = 0

    for k, v in tensors.items():
        nb = v.numel() * v.element_size()
        if cur_bytes + nb > max_bytes and cur_shard:
            shards.append(cur_shard)
            cur_shard = {}
            cur_bytes = 0
        cur_shard[k] = v
        cur_bytes += nb

    if cur_shard:
        shards.append(cur_shard)

    n = len(shards)
    weight_map: dict[str, str] = {}

    for idx, shard in enumerate(shards, 1):
        fname = f"model-{idx:05d}-of-{n:05d}.safetensors"
        fpath = out_dir / fname
        print(f"  Saving shard {idx}/{n}: {fname}  ({sum(v.numel()*v.element_size() for v in shard.values())/1e9:.2f} GB)")
        save_file(shard, str(fpath))
        for k in shard:
            weight_map[k] = fname

    index = {"metadata": {"total_size": sum(v.numel()*v.element_size() for v in tensors.values())},
             "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"  Wrote index with {len(weight_map)} tensors across {n} shards.")


# ── config builder ─────────────────────────────────────────────────────────────

def build_text_config(src_dir: Path) -> dict:
    """
    Build a Qwen3_5Config dict matching the native multimodal repo.

    We emit architecture = Qwen3_5ForConditionalGeneration (not ForCausalLM)
    because only the multimodal class overrides _prepare_position_ids_for_generation
    to produce the 3D MROPE position_ids the model requires during generation.
    Using ForCausalLM breaks multi-token generation even with correct weights.

    Vision weights will be randomly initialized — fine for text-only inference
    as long as the user never passes pixel_values.
    """
    cfg = json.loads((src_dir / "config.json").read_text())
    # Pass through the original config verbatim so the loader sees exactly the
    # same structure as the native repo. We only override transformers_version.
    cfg["transformers_version"] = "4.57.0.dev0"
    return cfg


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf",     type=Path, required=True)
    ap.add_argument("--mmproj",   type=Path, default=None, help="Vision mmproj GGUF (set to '' to skip)")
    ap.add_argument("--src",      type=Path, required=True, help="Source HF repo for tokenizer/config")
    ap.add_argument("--out",      type=Path, required=True)
    ap.add_argument("--shard-gb", type=float, default=4.0, help="Max shard size in GB")
    ap.add_argument("--device",   default="cuda", help="Device for tensor ops (cuda/cpu)")
    args = ap.parse_args()

    import gguf as gguf_lib
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 1. read GGUF ──────────────────────────────────────────────────────────
    print(f"\n[1] Reading GGUF: {args.gguf}")
    reader = gguf_lib.GGUFReader(str(args.gguf))
    tensor_names = {t.name for t in reader.tensors}
    print(f"    Total tensors in GGUF: {len(reader.tensors)}")

    # ── 2. convert tensors ────────────────────────────────────────────────────
    print("\n[2] Converting tensors...")
    hf_tensors: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for t in reader.tensors:
        name = t.name
        layer_idx = parse_layer_idx(name)

        raw = load_tensor(t)

        converted = list(convert_tensor(name, raw.to(device), layer_idx))
        if not converted:
            skipped.append(name)
        for hf_name, hf_tensor in converted:
            hf_tensors[hf_name] = hf_tensor.contiguous().to(torch.bfloat16)

    # ── 2b. vision mmproj ─────────────────────────────────────────────────────
    mmproj_path = args.mmproj
    if mmproj_path is not None and mmproj_path.exists() and mmproj_path.is_file():
        print(f"\n[2b] Reading vision mmproj GGUF: {mmproj_path}")
        v_reader = gguf_lib.GGUFReader(str(mmproj_path))
        print(f"     Vision tensors in GGUF: {len(v_reader.tensors)}")
        v_tensors = vision_pass(v_reader, device)
        print(f"     Converted vision tensors: {len(v_tensors)}")
        for k, v in v_tensors.items():
            hf_tensors[k] = v.contiguous().to(torch.bfloat16)
    else:
        print("\n[2b] Skipping vision mmproj (not provided / not found)")

    print(f"    Converted: {len(hf_tensors)} HF tensors")
    if skipped:
        print(f"    Skipped {len(skipped)} tensors: {skipped[:10]}{'...' if len(skipped)>10 else ''}")

    # ── 3. sanity-check expected keys ─────────────────────────────────────────
    print("\n[3] Checking expected keys...")
    missing = []
    for i in range(NUM_LAYERS):
        pfx = f"model.language_model.layers.{i}"
        required = [
            f"{pfx}.input_layernorm.weight",
            f"{pfx}.post_attention_layernorm.weight",
            f"{pfx}.mlp.gate_proj.weight",
            f"{pfx}.mlp.up_proj.weight",
            f"{pfx}.mlp.down_proj.weight",
        ]
        if LAYER_TYPES[i] == "linear_attention":
            required += [
                f"{pfx}.linear_attn.in_proj_qkv.weight",
                f"{pfx}.linear_attn.in_proj_z.weight",
                f"{pfx}.linear_attn.in_proj_b.weight",
                f"{pfx}.linear_attn.in_proj_a.weight",
                f"{pfx}.linear_attn.conv1d.weight",
                f"{pfx}.linear_attn.dt_bias",
                f"{pfx}.linear_attn.A_log",
                f"{pfx}.linear_attn.norm.weight",
                f"{pfx}.linear_attn.out_proj.weight",
            ]
        else:
            required += [
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx}.self_attn.o_proj.weight",
                f"{pfx}.self_attn.q_norm.weight",
                f"{pfx}.self_attn.k_norm.weight",
            ]
        for k in required:
            if k not in hf_tensors:
                missing.append(k)

    for g in ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]:
        if g not in hf_tensors:
            missing.append(g)

    if missing:
        print(f"    WARNING: {len(missing)} expected keys missing:")
        for m in missing[:20]:
            print(f"      {m}")
    else:
        print("    All expected keys present.")

    # ── 4. shapes spot-check ──────────────────────────────────────────────────
    print("\n[4] Shape spot-check:")
    LM = "model.language_model"
    spot = {
        f"{LM}.embed_tokens.weight":                  (248320, 4096),
        "lm_head.weight":                             (248320, 4096),
        f"{LM}.norm.weight":                          (4096,),
        f"{LM}.layers.0.input_layernorm.weight":      (4096,),
        f"{LM}.layers.0.linear_attn.in_proj_qkv.weight": (8192, 4096),
        f"{LM}.layers.0.linear_attn.in_proj_z.weight":   (4096, 4096),
        f"{LM}.layers.0.linear_attn.in_proj_b.weight":   (32, 4096),
        f"{LM}.layers.0.linear_attn.in_proj_a.weight":   (32, 4096),
        f"{LM}.layers.0.linear_attn.conv1d.weight":      (8192, 1, 4),
        f"{LM}.layers.0.linear_attn.dt_bias":             (32,),
        f"{LM}.layers.0.linear_attn.A_log":               (32,),
        f"{LM}.layers.0.linear_attn.norm.weight":         (128,),
        f"{LM}.layers.0.linear_attn.out_proj.weight":     (4096, 4096),
        f"{LM}.layers.11.self_attn.q_proj.weight":        (8192, 4096),
        f"{LM}.layers.11.self_attn.k_proj.weight":        (1024, 4096),
        f"{LM}.layers.11.self_attn.v_proj.weight":        (1024, 4096),
        f"{LM}.layers.11.self_attn.o_proj.weight":        (4096, 4096),
        f"{LM}.layers.11.self_attn.q_norm.weight":        (256,),
        f"{LM}.layers.11.self_attn.k_norm.weight":        (256,),
    }
    all_ok = True
    for k, expected in spot.items():
        if k not in hf_tensors:
            print(f"  MISSING: {k}")
            all_ok = False
        else:
            actual = tuple(hf_tensors[k].shape)
            ok = actual == expected
            status = "OK" if ok else "MISMATCH"
            if not ok:
                all_ok = False
            print(f"  {status:8s} {k}: {actual} (expected {expected})")
    if all_ok:
        print("  All shapes correct!")

    # ── 5. norm value sanity ──────────────────────────────────────────────────
    print("\n[5] Norm value sanity:")
    for k in [f"{LM}.layers.0.input_layernorm.weight",
              f"{LM}.layers.0.linear_attn.norm.weight",
              f"{LM}.norm.weight"]:
        if k in hf_tensors:
            v = hf_tensors[k].float()
            print(f"  {k}: min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")

    print("\n[6] A_log sanity (should be valid log values):")
    alog = hf_tensors.get(f"{LM}.layers.0.linear_attn.A_log")
    if alog is not None:
        print(f"  A_log[0]: min={alog.min():.4f} max={alog.max():.4f} (should be negative finite values)")

    # ── 6. save safetensors ───────────────────────────────────────────────────
    print(f"\n[7] Saving safetensors to: {args.out}")
    # Move all tensors to CPU before saving
    hf_tensors_cpu = {k: v.cpu() for k, v in hf_tensors.items()}
    shard_and_save(hf_tensors_cpu, args.out, args.shard_gb)

    # ── 7. write config ───────────────────────────────────────────────────────
    print("\n[8] Writing config.json...")
    cfg = build_text_config(args.src)
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))

    # ── 8. copy tokenizer files ───────────────────────────────────────────────
    print("\n[9] Copying tokenizer files from source repo...")
    tok_files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json", "chat_template.jinja",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "processor_config.json",
    ]
    for fname in tok_files:
        src_f = args.src / fname
        if src_f.exists():
            shutil.copy2(src_f, args.out / fname)
            print(f"    Copied {fname}")
        else:
            print(f"    Not found (skipping): {fname}")

    # ── 9. write generation_config ────────────────────────────────────────────
    print("\n[10] Writing generation_config.json...")
    gen_cfg = {
        "_from_model_config": True,
        "bos_token_id": None,
        # stop at <|endoftext|> (248044) AND <|im_end|> (248046) for chat
        "eos_token_id": [248044, 248046],
        "pad_token_id": 248044,
        "use_cache": True,
        "transformers_version": "4.57.0.dev0",
    }
    (args.out / "generation_config.json").write_text(json.dumps(gen_cfg, indent=2))
    print("    Written generation_config.json")

    # ── 10. final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Done. Output: {args.out}")
    print(f"  Tensors saved:  {len(hf_tensors)}")
    out_files = sorted(args.out.glob("*.safetensors"))
    total_gb = sum(f.stat().st_size for f in out_files) / 1e9
    print(f"  Safetensors:    {len(out_files)} files, {total_gb:.2f} GB total")
    print(f"\nTo load for fine-tuning:")
    print(f"  from transformers import Qwen3_5ForConditionalGeneration")
    print(f"  model = Qwen3_5ForConditionalGeneration.from_pretrained('{args.out}', dtype=torch.bfloat16, device_map='auto')")
    print(f"  # (vision weights will be randomly initialized — fine for text-only use)")
    print(f"\nTo verify:")
    print(f"  python verify_hf_qwen.py --model '{args.out}'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
