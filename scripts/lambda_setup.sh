#!/usr/bin/env bash
# One-time setup on a fresh Lambda GPU instance, run from the repo root after
# copying it over (README Appendix C). Creates .venv and checks everything the
# training run needs before you start paying for it to fail.
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Lambda's Ubuntu 22.04 image has Python 3.10; the pinned versions need >= 3.11.
# uv fetches the Python the pins were tested on.
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -q --allow-existing --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements-cuda.txt
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch cannot see the GPU'; \
print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

for split in train val; do
    [ -f "data/datasets/$split.jsonl" ] || { echo "missing data/datasets/$split.jsonl"; exit 1; }
done
if [ -z "${HF_TOKEN:-}" ] && [ ! -s ~/.cache/huggingface/token ]; then
    echo "No Hugging Face token. Mistral-7B-Instruct-v0.3 is gated: accept its licence on"
    echo "huggingface.co, create a read token, then: export HF_TOKEN=hf_..."
    echo "(or save it to ~/.cache/huggingface/token)"
    exit 1
fi
.venv/bin/python -c "from huggingface_hub import model_info; \
model_info('mistralai/Mistral-7B-Instruct-v0.3'); print('Hugging Face access: ok')"
echo "ready. In tmux: .venv/bin/python scripts/train_cuda.py --micro-batch 16"
