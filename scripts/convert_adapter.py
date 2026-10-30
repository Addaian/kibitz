#!/usr/bin/env python3
"""
Convert a peft LoRA adapter (from scripts/train_cuda.py) into mlx-lm's format.

    python3 scripts/convert_adapter.py adapters/cuda adapters/mistral7b-cuda
    python3 scripts/evaluate.py --adapter adapters/mistral7b-cuda

The result loads onto the 4-bit MLX base like any adapter trained here, so the
CLI, evaluation and KibitzCommentator need no changes. The two formats hold the
same matrices: peft computes `(alpha/r) * B(A x)` with A of shape (r, in);
mlx-lm computes `scale * (x a) b` with a of shape (in, r). So a = A^T, b = B^T
and scale = alpha / r.

mlx-lm loads adapter weights non-strictly, so a misnamed key would be ignored
without a word. This script therefore checks that it wrote exactly the keys the
config describes.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
_PEFT_KEY = re.compile(r"^base_model\.model\.(model\.layers\.(\d+)\.(\w+)\.(\w+))\.lora_([AB])\.weight$")


def convert(peft_dir: Path, out_dir: Path, mlx_base: str) -> int:
    import mlx.core as mx

    cfg = json.loads((peft_dir / "adapter_config.json").read_text())
    for flag in ("use_dora", "use_rslora"):
        if cfg.get(flag):
            sys.exit(f"{flag} adapters are not supported")
    if cfg.get("bias", "none") != "none":
        sys.exit("adapters with trained biases are not supported")
    rank, alpha = cfg["r"], cfg["lora_alpha"]

    weights = mx.load(str(peft_dir / "adapter_model.safetensors"))
    out, layers, modules = {}, set(), set()
    for key, value in weights.items():
        m = _PEFT_KEY.match(key)
        if not m:
            sys.exit(f"unexpected key in adapter: {key}")
        path, layer, block, name, ab = m.groups()
        layers.add(int(layer))
        modules.add(f"{block}.{name}")
        out[f"{path}.lora_{ab.lower()}"] = value.astype(mx.float32).T

    # mlx-lm adds LoRA to the last `num_layers` blocks, so they must be contiguous
    # and reach the top; the keys then name exactly which modules in each.
    expected = {f"model.layers.{l}.{mod}.lora_{ab}"
                for l in layers for mod in modules for ab in "ab"}
    if set(out) != expected:
        sys.exit(f"adapter is not a full grid of layers x modules: "
                 f"{len(out)} keys, expected {len(expected)}")
    if sorted(layers) != list(range(min(layers), max(layers) + 1)):
        sys.exit(f"adapter layers are not contiguous: {sorted(layers)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), out)
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora",
        "model": mlx_base,
        "converted_from": str(peft_dir),
        "hf_base_model": cfg.get("base_model_name_or_path"),
        # Counted from the top. Assumes the adapter reaches the last block, which
        # train_cuda.py guarantees; check for adapters from anywhere else.
        "num_layers": len(layers),
        "lora_parameters": {"rank": rank, "scale": alpha / rank, "dropout": 0.0,
                            "keys": sorted(modules)},
    }, indent=2))
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("peft_dir", type=Path, help="directory with adapter_model.safetensors")
    ap.add_argument("out_dir", type=Path, help="where to write the MLX adapter")
    ap.add_argument("--mlx-base", help="MLX base model it will be loaded onto "
                                       "(default: configs/model.yaml base_model)")
    args = ap.parse_args()

    base = args.mlx_base or yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())["base_model"]
    if (args.out_dir / "adapters.safetensors").exists():
        sys.exit(f"{args.out_dir} already holds an adapter; choose another directory")
    n = convert(args.peft_dir, args.out_dir, base)
    print(f"wrote {n} tensors to {args.out_dir} (for {base})")


if __name__ == "__main__":
    main()
