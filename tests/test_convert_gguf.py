"""
scripts/convert_gguf.py must write the adapter llama.cpp expects for the base it
will be applied to.

llama.cpp applies an adapter whose tensors match nothing without complaint, and
the q/k row permutation is invisible in any single tensor. So: tensor names and
alpha, the permutation (as the same function of the merged weight that llama.cpp's
converter applies to the base), and refusal of an adapter that does not fit.
Numerical parity on the real model is a separate, manual check (README §8).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
pytest.importorskip("safetensors")

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("convert_gguf", ROOT / "scripts" / "convert_gguf.py")
convert_gguf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_gguf)

HIDDEN, FFN, N_HEAD, N_HEAD_KV, RANK = 32, 48, 4, 2, 4
KV = HIDDEN // N_HEAD * N_HEAD_KV
SHAPES = {  # (out, in), as torch stores them
    "self_attn.q_proj": (HIDDEN, HIDDEN), "self_attn.k_proj": (KV, HIDDEN),
    "self_attn.v_proj": (KV, HIDDEN), "self_attn.o_proj": (HIDDEN, HIDDEN),
    "mlp.gate_proj": (FFN, HIDDEN), "mlp.up_proj": (FFN, HIDDEN), "mlp.down_proj": (HIDDEN, FFN),
}


def fake_base(path: Path, layers=(0, 1)) -> Path:
    """Just enough of a base GGUF: architecture, head counts, tensor names and shapes."""
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_head_count(N_HEAD)
    writer.add_head_count_kv(N_HEAD_KV)
    for layer in layers:
        for module, shape in SHAPES.items():
            name = convert_gguf._GGUF_NAME[module]
            writer.add_tensor(f"blk.{layer}.{name}.weight", np.zeros(shape, np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def fake_peft(path: Path, layers=(1,), shapes=SHAPES) -> dict:
    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    tensors = {}
    for layer in layers:
        for module, (out, inp) in shapes.items():
            prefix = f"base_model.model.model.layers.{layer}.{module}"
            tensors[f"{prefix}.lora_A.weight"] = rng.normal(size=(RANK, inp)).astype(np.float32)
            tensors[f"{prefix}.lora_B.weight"] = rng.normal(size=(out, RANK)).astype(np.float32)
    path.mkdir()
    save_file(tensors, str(path / "adapter_model.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps({"r": RANK, "lora_alpha": 80.0}))
    return tensors


def read(path: Path) -> tuple[dict, dict]:
    reader = gguf.GGUFReader(path)
    fields = {k: f.contents() for k, f in reader.fields.items() if not k.startswith("GGUF.")}
    # Back to torch's (out, in) order.
    tensors = {t.name: np.asarray(t.data).reshape(tuple(reversed([int(d) for d in t.shape])))
               for t in reader.tensors}
    return fields, tensors


def test_writes_a_llama_cpp_lora(tmp_path):
    peft = fake_peft(tmp_path / "peft")
    n = convert_gguf.convert(tmp_path / "peft", tmp_path / "out.gguf", fake_base(tmp_path / "base.gguf"))
    assert n == 7 * 2

    fields, tensors = read(tmp_path / "out.gguf")
    assert fields["general.architecture"] == "llama"
    assert fields["general.type"] == "adapter"
    assert fields["adapter.type"] == "lora"
    assert fields["adapter.lora.alpha"] == 80.0      # raw: llama.cpp divides by r itself
    assert sorted(tensors) == sorted(f"blk.1.{name}.weight.lora_{ab}"
                                     for name in convert_gguf._GGUF_NAME.values() for ab in "ab")

    v = "base_model.model.model.layers.1.self_attn.v_proj"
    np.testing.assert_array_equal(tensors["blk.1.attn_v.weight.lora_a"], peft[f"{v}.lora_A.weight"])
    np.testing.assert_array_equal(tensors["blk.1.attn_v.weight.lora_b"], peft[f"{v}.lora_B.weight"])


@pytest.mark.parametrize("module,name,heads", [("self_attn.q_proj", "attn_q", N_HEAD),
                                               ("self_attn.k_proj", "attn_k", N_HEAD_KV)])
def test_q_and_k_are_permuted_like_the_base(tmp_path, module, name, heads):
    peft = fake_peft(tmp_path / "peft")
    convert_gguf.convert(tmp_path / "peft", tmp_path / "out.gguf", fake_base(tmp_path / "base.gguf"))
    _, tensors = read(tmp_path / "out.gguf")

    a = peft[f"base_model.model.model.layers.1.{module}.lora_A.weight"]
    b = peft[f"base_model.model.model.layers.1.{module}.lora_B.weight"]
    got = tensors[f"blk.1.{name}.weight.lora_b"] @ tensors[f"blk.1.{name}.weight.lora_a"]
    # The adapter's update must be permuted exactly as the converter permutes the base weight.
    np.testing.assert_allclose(got, convert_gguf.permute_rows(b @ a, heads), rtol=1e-5, atol=1e-5)
    assert not np.allclose(got, b @ a)


def test_an_adapter_that_does_not_fit_the_base_is_refused(tmp_path):
    fake_peft(tmp_path / "peft", shapes={**SHAPES, "mlp.up_proj": (FFN + 8, HIDDEN)})
    with pytest.raises(SystemExit, match="does not fit"):
        convert_gguf.convert(tmp_path / "peft", tmp_path / "out.gguf", fake_base(tmp_path / "base.gguf"))


def test_layers_missing_from_the_base_are_refused(tmp_path):
    fake_peft(tmp_path / "peft", layers=(5,))
    with pytest.raises(SystemExit, match="not a tensor of the base"):
        convert_gguf.convert(tmp_path / "peft", tmp_path / "out.gguf", fake_base(tmp_path / "base.gguf"))
