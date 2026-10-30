"""
A peft adapter converted by scripts/convert_adapter.py must compute the same
function in MLX as in peft.

mlx-lm loads adapters non-strictly, so a misnamed key or an untransposed matrix
would load "fine" and silently change nothing, or the wrong thing. Comparing
logits on a tiny random Mistral catches both. Needs torch, peft and mlx; skipped
otherwise (peft is not in requirements.txt; see requirements-cuda.txt).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("mlx")

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("convert_adapter", ROOT / "scripts" / "convert_adapter.py")
convert_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_adapter)


def tiny_mistral(path: Path):
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(0)
    config = MistralConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                           num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                           max_position_embeddings=64, tie_word_embeddings=False)
    model = MistralForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def test_converted_adapter_matches_peft(tmp_path):
    import mlx.core as mx
    import numpy as np
    from mlx_lm.tuner.utils import load_adapters
    from mlx_lm.utils import load_model
    from peft import LoraConfig, get_peft_model

    base = tiny_mistral(tmp_path / "base")
    peft_model = get_peft_model(base, LoraConfig(
        r=4, lora_alpha=4 * 20, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=[2, 3], task_type="CAUSAL_LM"))
    # B starts at zero, which would make any conversion look right.
    with torch.no_grad():
        for name, p in peft_model.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.05)
    peft_model.save_pretrained(tmp_path / "peft")

    n = convert_adapter.convert(tmp_path / "peft", tmp_path / "mlx_adapter", "tiny")
    assert n == 2 * 2 * 7
    cfg = json.loads((tmp_path / "mlx_adapter" / "adapter_config.json").read_text())
    assert cfg["num_layers"] == 2 and cfg["lora_parameters"]["scale"] == 20.0

    tokens = [[1, 5, 9, 33, 70, 2, 101, 64]]
    with torch.no_grad():
        expected = peft_model(torch.tensor(tokens)).logits.numpy()

    mlx_model, _ = load_model(tmp_path / "base")
    load_adapters(mlx_model, str(tmp_path / "mlx_adapter"))
    got = np.array(mlx_model(mx.array(tokens)))

    assert np.abs(got - expected).max() < 1e-3
    # And the adapter matters: the comparison is not passing on base weights alone.
    with peft_model.disable_adapter(), torch.no_grad():
        plain = peft_model(torch.tensor(tokens)).logits.numpy()
    assert np.abs(plain - expected).max() > 1e-2


def test_a_partial_adapter_is_refused(tmp_path):
    import mlx.core as mx

    (tmp_path / "peft").mkdir()
    (tmp_path / "peft" / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    mx.save_safetensors(str(tmp_path / "peft" / "adapter_model.safetensors"), {
        "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight": mx.zeros((4, 64)),
        "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight": mx.zeros((64, 4)),
        "base_model.model.model.layers.3.self_attn.v_proj.lora_A.weight": mx.zeros((4, 64)),
    })
    with pytest.raises(SystemExit, match="full grid"):
        convert_adapter.convert(tmp_path / "peft", tmp_path / "out", "tiny")
