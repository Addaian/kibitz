#!/usr/bin/env python3
"""
LoRA fine-tuning, via MLX.

    python3 scripts/train.py --measure           # token lengths, no training
    python3 scripts/train.py                     # train with the configs as written
    python3 scripts/train.py --iters 200 --warmup 5 --model mlx-community/Llama-3.2-3B-Instruct-4bit

Reads configs/model.yaml and configs/training.yaml and hands the merged result to
mlx-lm. Command-line flags override the files, so the configs stay the record of
what a real run used.

The plan (§7.2, §7.3) specifies peft + bitsandbytes, which is CUDA-only and cannot
run on this machine; see Decision Log row 11.
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "data" / "datasets"
MLX_DATA = DATASETS / "mlx"
CONFIGS = ROOT / "configs"


def load_configs() -> dict:
    """Merge mlx-lm's defaults, then model.yaml, then training.yaml."""
    from mlx_lm.lora import CONFIG_DEFAULTS

    config = dict(CONFIG_DEFAULTS)
    model_cfg = yaml.safe_load((CONFIGS / "model.yaml").read_text())
    train_cfg = yaml.safe_load((CONFIGS / "training.yaml").read_text())

    # our key is `base_model`; mlx-lm calls it `model`
    model_cfg["model"] = model_cfg.pop("base_model")
    config.update(model_cfg)
    config.update(train_cfg)
    return config


def prepare_data_dir() -> Path:
    """
    mlx-lm looks for train/valid/test.jsonl; §4.4 of the plan names the middle
    split `val.jsonl`. Symlink rather than copy or rename, so there is exactly
    one source of truth for the data.
    """
    missing = [n for n in ("train", "val", "test") if not (DATASETS / f"{n}.jsonl").exists()]
    if missing:
        sys.exit(f"missing splits {missing} — run: python3 scripts/build_dataset.py")

    MLX_DATA.mkdir(parents=True, exist_ok=True)
    for ours, theirs in (("train", "train"), ("val", "valid"), ("test", "test")):
        link = MLX_DATA / f"{theirs}.jsonl"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / f"{ours}.jsonl")
    return MLX_DATA


def measure(config) -> None:
    """Token-length distribution, so max_seq_length is a measurement not a guess."""
    from mlx_lm import load

    print(f"loading tokenizer from {config['model']} ...")
    _, tokenizer = load(config["model"])

    for split in ("train", "val", "test"):
        path = DATASETS / f"{split}.jsonl"
        prompt_len, total_len = [], []
        for line in path.open():
            msgs = json.loads(line)["messages"]
            p = tokenizer.apply_chat_template(msgs[:1], tokenize=True, add_generation_prompt=True)
            f = tokenizer.apply_chat_template(msgs, tokenize=True)
            prompt_len.append(len(p))
            total_len.append(len(f))
        prompt_len.sort()
        total_len.sort()
        q = lambda xs, pct: xs[int(len(xs) * pct)]
        over = sum(1 for t in total_len if t > config["max_seq_length"])
        print(f"\n{split}: {len(total_len):,} pairs")
        print(f"  prompt tokens  median {q(prompt_len,.5):4d}   p95 {q(prompt_len,.95):4d}   max {prompt_len[-1]:4d}")
        print(f"  total  tokens  median {q(total_len,.5):4d}   p95 {q(total_len,.95):4d}   max {total_len[-1]:4d}")
        print(f"  over max_seq_length={config['max_seq_length']}: {over:,} ({100*over/len(total_len):.2f}%)")


class StopOnNaN:
    """
    Abort when a reported loss stops being finite. mlx-lm trains on regardless,
    and a diverged run once spent ten hours writing NaN checkpoints.
    """

    def on_train_loss_report(self, info: dict):
        self._check("train", info["train_loss"], info["iteration"])

    def on_val_loss_report(self, info: dict):
        self._check("val", info["val_loss"], info["iteration"])

    @staticmethod
    def _check(split, loss, iteration):
        if not math.isfinite(loss):
            sys.exit(f"{split} loss is {loss} at iter {iteration}; stopping. "
                     f"Checkpoints saved before this point are in the adapter directory.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true",
                    help="report token lengths and exit, without training")
    ap.add_argument("--model", help="override configs/model.yaml base_model")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--batch-size", type=int, dest="batch_size")
    ap.add_argument("--learning-rate", type=float, dest="learning_rate")
    ap.add_argument("--max-seq-length", type=int, dest="max_seq_length")
    ap.add_argument("--adapter-path", dest="adapter_path")
    ap.add_argument("--warmup", type=int,
                    help="LR warmup, in optimizer updates; the config default suits a full run, "
                         "not a short one")
    ap.add_argument("--resume", dest="resume_adapter_file",
                    help="continue from an existing adapter file")
    args = ap.parse_args()

    config = load_configs()
    for key in ("model", "iters", "batch_size", "learning_rate",
                "max_seq_length", "adapter_path", "resume_adapter_file"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value

    if args.measure:
        return measure(config)

    config["data"] = str(prepare_data_dir())
    config["train"] = True
    config["adapter_path"] = str(ROOT / config["adapter_path"])

    # A cosine schedule written for the default iters would decay against the
    # wrong horizon if --iters was passed. mlx-lm counts `iters` in micro-batches
    # but steps the schedule once per optimizer update, so the horizon is
    # iters / grad_accumulation_steps — using iters itself leaves the LR barely
    # decayed when the run ends.
    updates = config["iters"] // config.get("grad_accumulation_steps", 1)
    schedule = config.get("lr_schedule")
    if schedule and schedule.get("arguments") and len(schedule["arguments"]) >= 2:
        schedule["arguments"][0] = config["learning_rate"]
        schedule["arguments"][1] = updates
        if args.warmup is not None:
            schedule["warmup"] = args.warmup
        print(f"lr         {config['learning_rate']:.1e} cosine over "
              f"{updates} updates ({config['iters']} iters), {schedule['warmup']} warmup updates")

    print(f"model      {config['model']}")
    print(f"data       {config['data']}")
    print(f"adapters   {config['adapter_path']}")
    print(f"iters      {config['iters']}  batch {config['batch_size']}"
          f" x {config['grad_accumulation_steps']} accum"
          f"  = effective {config['batch_size'] * config['grad_accumulation_steps']}")
    print(f"lora       rank {config['lora_parameters']['rank']}"
          f"  scale {config['lora_parameters']['scale']}"
          f"  layers {config['num_layers']}")
    print(f"mask_prompt {config['mask_prompt']}\n")

    from mlx_lm.lora import run
    run(types.SimpleNamespace(**config), training_callback=StopOnNaN())


if __name__ == "__main__":
    main()
