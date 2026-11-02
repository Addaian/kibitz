#!/usr/bin/env python3
"""
Convert a peft LoRA adapter (from scripts/train_cuda.py) into a GGUF LoRA for llama.cpp.

    python3 scripts/convert_gguf.py adapters/cuda-v2 adapters/kibitz-lora.gguf \
        --base ~/.cache/.../Mistral-7B-Instruct-v0.3-Q4_K_M.gguf

llama.cpp runs on Mac, Windows and Linux, CPU or GPU, so this is the format
Kibitz ships in: a stock base GGUF plus this ~80 MB adapter, applied at load time.

The matrices are peft's own: lora_a is (r, in), lora_b is (out, r), and alpha is
written raw, since llama.cpp derives scale = alpha / r from lora_b itself. One
transformation is needed. llama.cpp's HF converter reorders the rows of q_proj
and k_proj for its rotary layout, so the base GGUF holds permuted weights, and
the adapter's output rows (lora_b) must be permuted the same way to line up.

llama.cpp, like mlx-lm, would apply an adapter whose tensors match nothing in
the base without complaint. So every tensor is checked against the base GGUF's
names and shapes before anything is written.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_PEFT_KEY = re.compile(r"^base_model\.model\.model\.layers\.(\d+)\.(\w+)\.(\w+)\.lora_([AB])\.weight$")
_GGUF_NAME = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def permute_rows(w: np.ndarray, n_head: int) -> np.ndarray:
    """The row order llama.cpp's converter gives q_proj / k_proj, applied to lora_b."""
    return (w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:])
            .swapaxes(1, 2)
            .reshape(w.shape))


def read_base(base_gguf: Path) -> tuple[str, int, int, dict[str, tuple[int, ...]]]:
    """(architecture, n_head, n_head_kv, {tensor name: (out, in)}) of a base GGUF."""
    from gguf import GGUFReader

    reader = GGUFReader(base_gguf)

    def field(key):
        f = reader.fields.get(key)
        if f is None:
            sys.exit(f"{base_gguf} has no {key}")
        return f.contents()

    arch = field("general.architecture")
    if arch != "llama":
        sys.exit(f"base architecture is {arch!r}; only llama-layout models (Llama, Mistral) are supported")
    # GGUF lists dimensions innermost first; reverse to torch's (out, in).
    shapes = {t.name: tuple(int(d) for d in reversed(t.shape)) for t in reader.tensors}
    return arch, int(field(f"{arch}.attention.head_count")), int(field(f"{arch}.attention.head_count_kv")), shapes


def convert(peft_dir: Path, out_path: Path, base_gguf: Path) -> int:
    from gguf import GGUFType, GGUFWriter, Keys
    from safetensors.numpy import load_file

    cfg = json.loads((peft_dir / "adapter_config.json").read_text())
    for flag in ("use_dora", "use_rslora"):
        if cfg.get(flag):
            sys.exit(f"{flag} adapters are not supported")
    if cfg.get("bias", "none") != "none":
        sys.exit("adapters with trained biases are not supported")

    arch, n_head, n_head_kv, base_shapes = read_base(base_gguf)
    weights = load_file(str(peft_dir / "adapter_model.safetensors"))

    out = {}
    for key, value in weights.items():
        m = _PEFT_KEY.match(key)
        if not m or f"{m[2]}.{m[3]}" not in _GGUF_NAME:
            sys.exit(f"unexpected key in adapter: {key}")
        layer, block, name, ab = m.groups()
        module = _GGUF_NAME[f"{block}.{name}"]
        value = value.astype(np.float32)
        if ab == "B" and module == "attn_q":
            value = permute_rows(value, n_head)
        elif ab == "B" and module == "attn_k":
            value = permute_rows(value, n_head_kv)
        out[f"blk.{layer}.{module}.weight.lora_{ab.lower()}"] = value

    for name in sorted({n.rsplit(".lora_", 1)[0] for n in out}):
        a, b = out.get(f"{name}.lora_a"), out.get(f"{name}.lora_b")
        if a is None or b is None:
            sys.exit(f"{name} has only one of lora_a / lora_b")
        if name not in base_shapes:
            sys.exit(f"{name} is not a tensor of the base model")
        if (b.shape[0], a.shape[1]) != base_shapes[name] or a.shape[0] != b.shape[1]:
            sys.exit(f"{name}: adapter {(b.shape[0], a.shape[1])} does not fit base {base_shapes[name]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(out_path, arch)
    writer.add_type(GGUFType.ADAPTER)
    writer.add_string(Keys.Adapter.TYPE, "lora")
    writer.add_float32(Keys.Adapter.LORA_ALPHA, float(cfg["lora_alpha"]))
    for name in sorted(out):
        writer.add_tensor(name, out[name])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("peft_dir", type=Path, help="directory with adapter_model.safetensors")
    ap.add_argument("out_path", type=Path, help="the .gguf adapter to write")
    ap.add_argument("--base", type=Path, required=True,
                    help="the base model GGUF the adapter will be applied to")
    args = ap.parse_args()

    if args.out_path.exists():
        sys.exit(f"{args.out_path} already exists; choose another path")
    n = convert(args.peft_dir, args.out_path, args.base)
    print(f"wrote {n} tensors to {args.out_path} (for {args.base.name})")


if __name__ == "__main__":
    main()
