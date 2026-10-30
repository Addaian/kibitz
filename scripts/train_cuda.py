#!/usr/bin/env python3
"""
LoRA fine-tuning on an NVIDIA GPU (a Lambda instance), via transformers + peft + trl.

    python3 scripts/train_cuda.py                     # the configs as written
    python3 scripts/train_cuda.py --micro-batch 16    # same effective batch, no accumulation
    python3 scripts/train_cuda.py --model <hf-id> --max-steps 3 --out /tmp/smoke

The MLX run (scripts/train.py) and this one are meant to be interchangeable, so
both read configs/model.yaml and configs/training.yaml and this script
translates rather than restates them:

* mlx-lm puts LoRA on every linear layer (q k v o gate up down) of the top
  `num_layers` blocks and applies `scale * BA`. peft's equivalent is
  `lora_alpha = scale * rank` on the same modules, so the learning rate carries
  over unchanged. (A peft-style alpha/r of 2 at LR 1e-5 would train 10x weaker.)
* `iters` counts micro-batches; transformers counts optimizer updates, so
  max_steps = iters / grad_accumulation_steps.
* mask_prompt = loss on the comment only: the data becomes prompt/completion
  pairs, which trl trains with completion_only_loss.

The base is the original bf16 Mistral, not the 4-bit MLX copy (a gated model:
accept its licence on Hugging Face and set HF_TOKEN). The adapter comes back to
the Mac with scripts/convert_adapter.py.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
DATASETS = ROOT / "data" / "datasets"
CONFIGS = ROOT / "configs"

# mlx-lm's default LoRA targets for Mistral/Llama blocks.
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Validation loss on a fixed subset, so successive evaluations are comparable.
# (mlx-lm samples a fresh random 100 each time, which made its curve mostly noise.)
VAL_EXAMPLES = 500


def load_configs() -> dict:
    model_cfg = yaml.safe_load((CONFIGS / "model.yaml").read_text())
    train_cfg = yaml.safe_load((CONFIGS / "training.yaml").read_text())
    return {**model_cfg, **train_cfg}


def load_split(name: str, limit: int | None = None, data_dir: Path = DATASETS):
    """A split as trl prompt/completion pairs; provenance (`meta`) is dropped."""
    from datasets import Dataset

    rows = []
    for line in (data_dir / f"{name}.jsonl").open():
        user, assistant = json.loads(line)["messages"]
        rows.append({"prompt": [user], "completion": [assistant]})
        if limit and len(rows) >= limit:
            break
    return Dataset.from_list(rows)


def stop_on_nan():
    from transformers import TrainerCallback

    class StopOnNaN(TrainerCallback):
        """Same guard as train.py: a diverged run should stop, not save NaNs for hours."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "eval_loss"):
                value = (logs or {}).get(key)
                if value is not None and not math.isfinite(value):
                    print(f"\n{key} is {value} at step {state.global_step}; stopping.")
                    control.should_training_stop = True

    return StopOnNaN()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="override configs/model.yaml hf_base_model")
    ap.add_argument("--data", type=Path, default=DATASETS,
                    help="directory with train.jsonl and val.jsonl (default: data/datasets)")
    ap.add_argument("--out", type=Path, default=ROOT / "adapters" / "cuda",
                    help="output directory (default: adapters/cuda)")
    ap.add_argument("--micro-batch", type=int,
                    help="per-step batch; accumulation is adjusted to keep the effective batch")
    ap.add_argument("--max-steps", type=int, help="override: optimizer updates")
    ap.add_argument("--train-limit", type=int, help="use only the first N train pairs (smoke tests)")
    ap.add_argument("--cpu", action="store_true", help="smoke-test on CPU in float32")
    args = ap.parse_args()

    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    config = load_configs()
    model_id = args.model or config["hf_base_model"]
    lora = config["lora_parameters"]

    effective = config["batch_size"] * config["grad_accumulation_steps"]
    micro = args.micro_batch or config["batch_size"]
    if effective % micro:
        sys.exit(f"--micro-batch {micro} does not divide the effective batch {effective}")
    updates = args.max_steps or config["iters"] // config["grad_accumulation_steps"]
    every = lambda iters: max(1, iters // config["grad_accumulation_steps"])  # iters -> updates

    if not args.cpu and not torch.cuda.is_available():
        sys.exit("no CUDA device; pass --cpu for a smoke test")
    dtype = torch.float32 if args.cpu else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    num_blocks = model.config.num_hidden_layers
    top = list(range(max(0, num_blocks - config["num_layers"]), num_blocks))

    peft_config = LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["scale"] * lora["rank"],
        lora_dropout=lora["dropout"],
        target_modules=LORA_MODULES,
        layers_to_transform=top,
        task_type="CAUSAL_LM",
    )
    schedule = config["lr_schedule"]
    sft = SFTConfig(
        output_dir=str(args.out),
        max_steps=updates,
        per_device_train_batch_size=micro,
        per_device_eval_batch_size=micro,
        gradient_accumulation_steps=effective // micro,
        learning_rate=config["learning_rate"],
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr": schedule["arguments"][2]},
        warmup_steps=schedule["warmup"],
        weight_decay=0.01,                      # mlx's AdamW default
        max_length=config["max_seq_length"],
        completion_only_loss=True,
        logging_steps=every(config["steps_per_report"]),
        eval_strategy="steps",
        eval_steps=every(config["steps_per_eval"]),
        save_steps=every(config["save_every"]),
        bf16=not args.cpu,
        use_cpu=args.cpu,
        gradient_checkpointing=config.get("grad_checkpoint", False),
        seed=config["seed"],
        report_to="none",
    )

    print(f"model      {model_id} ({dtype}, {num_blocks} blocks)")
    print(f"lora       rank {lora['rank']}  alpha {peft_config.lora_alpha} (= mlx scale "
          f"{lora['scale']})  blocks {top[0]}-{top[-1]}  {', '.join(LORA_MODULES)}")
    print(f"schedule   {updates} updates, batch {micro} x {effective // micro} = {effective}, "
          f"lr {config['learning_rate']:.1e} cosine to {schedule['arguments'][2]:.0e}, "
          f"{schedule['warmup']} warmup")
    print(f"data       {args.data}")
    print(f"output     {args.out}\n")

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=load_split("train", args.train_limit, args.data),
        eval_dataset=load_split("val", VAL_EXAMPLES, args.data),
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[stop_on_nan()],
    )
    trainer.train()
    trainer.save_model(str(args.out))
    print(f"\nadapter saved to {args.out}; on the Mac: "
          f"python3 scripts/convert_adapter.py {args.out} adapters/<name>")


if __name__ == "__main__":
    main()
