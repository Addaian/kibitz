"""
Unit tests for kibitz.model.

Anything needing real weights is skipped unless mlx-lm is installed, so the
suite still runs on a machine with no model downloaded.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import model as M
from kibitz import review

def test_default_base_model_comes_from_the_config():
    """The config file is the record of what a run used, so code must read it."""
    import yaml
    configured = yaml.safe_load((Path(__file__).parent.parent / "configs" / "model.yaml").read_text())
    assert M.default_base_model() == configured["base_model"]
    assert M.default_base_model().startswith("mlx-community/")

# --------------------------------------------------------------------------- #
# adapter resolution
# --------------------------------------------------------------------------- #

def test_no_adapter_path_means_base_model():
    assert M.resolve_adapter(None) is None

def test_a_missing_adapter_directory_resolves_to_none(tmp_path):
    """resolve_adapter reports it; KibitzCommentator then refuses (tested below)."""
    assert M.resolve_adapter(tmp_path / "nope") is None

def test_an_empty_adapter_directory_resolves_to_none(tmp_path):
    (tmp_path / "adapters").mkdir()
    assert M.resolve_adapter(tmp_path / "adapters") is None

def test_a_trained_adapter_is_used(tmp_path):
    adapter = tmp_path / "adapters"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"")
    assert M.resolve_adapter(adapter) == adapter

# --------------------------------------------------------------------------- #
# the Commentator contract
# --------------------------------------------------------------------------- #

def test_the_commentator_classes_satisfy_the_protocol():
    """Structural check — no weights needed, so this runs everywhere."""
    for cls in (M.KibitzCommentator, M.BaselineCommentator):
        assert hasattr(cls, "comment")
        assert callable(cls.comment)

def test_a_stub_with_the_same_shape_is_accepted_by_the_harness():
    class Stub:
        name = "KIBITZ (generated)"
        def comment(self, meta, move):
            return "Sample commentary."

    assert isinstance(Stub(), review.Commentator)

def test_baseline_always_ignores_the_adapter(tmp_path, monkeypatch):
    """The baseline is only meaningful if it is genuinely un-tuned."""
    captured = {}

    def fake_load(base, adapter_path=None):
        captured["adapter"] = adapter_path
        return object(), object()

    adapter = tmp_path / "adapters"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"")

    import mlx_lm
    monkeypatch.setattr(mlx_lm, "load", fake_load)
    c = M.BaselineCommentator(base_model="fake/model")
    assert captured["adapter"] is None
    assert c.name == "BASELINE (no fine-tune)"

def test_the_tuned_commentator_passes_its_adapter_through(tmp_path, monkeypatch):
    captured = {}

    def fake_load(base, adapter_path=None):
        captured["adapter"] = adapter_path
        return object(), object()

    adapter = tmp_path / "adapters"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"")

    import mlx_lm
    monkeypatch.setattr(mlx_lm, "load", fake_load)
    M.KibitzCommentator(base_model="fake/model", adapter_path=adapter)
    assert captured["adapter"] == str(adapter)

def test_a_missing_adapter_is_refused_not_silently_replaced(tmp_path, monkeypatch):
    """Otherwise the base model is labelled KIBITZ and every comparison is base-vs-base."""
    import mlx_lm
    monkeypatch.setattr(mlx_lm, "load", lambda base, adapter_path=None: (object(), object()))
    with pytest.raises(FileNotFoundError):
        M.KibitzCommentator(base_model="fake/model", adapter_path=tmp_path / "nope")


E4 = {"fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
      "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"}


def sampling_commentator(monkeypatch, outputs, samples):
    """A commentator whose generate() returns `outputs` in turn; records each call."""
    import mlx_lm
    monkeypatch.setattr(mlx_lm, "load", lambda base, adapter_path=None: (object(), object()))
    calls = iter(outputs)
    monkeypatch.setattr(mlx_lm, "generate", lambda *a, **k: next(calls))
    c = M.KibitzCommentator(base_model="fake/model", adapter_path=None, samples=samples)
    monkeypatch.setattr(c, "render", lambda meta, move, length=None: "prompt")
    return c


def test_problems_counts_each_detectable_fault():
    assert M.problems(E4, "White takes the centre.") == 0
    assert M.problems(E4, "The knight on e4 is strong.") == 1
    assert M.problems(E4, "The most popular move. Qxh7 wins.") == 2
    assert M.problems(E4, None) == 1


def test_sampling_stops_at_the_first_clean_comment(monkeypatch):
    c = sampling_commentator(monkeypatch, ["The knight on e4 is strong.", "White takes the centre.",
                                           "never reached"], samples=3)
    assert c.comment({}, E4) == "White takes the centre."


def test_with_no_clean_sample_the_least_bad_is_kept(monkeypatch):
    c = sampling_commentator(monkeypatch, ["The most popular move. Qxh7 wins.",
                                           "The knight on e4 is strong."], samples=2)
    assert c.comment({}, E4) == "The knight on e4 is strong."


def test_one_sample_is_returned_unchecked(monkeypatch):
    c = sampling_commentator(monkeypatch, ["The knight on e4 is strong."], samples=1)
    assert c.comment({}, E4) == "The knight on e4 is strong."
