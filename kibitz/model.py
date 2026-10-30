"""
Loading a fine-tuned Kibitz and generating commentary.

`KibitzCommentator` satisfies the `Commentator` protocol in `kibitz.review`, so a
trained model drops straight into the review harness beside the human annotator:

    from kibitz.model import KibitzCommentator
    from kibitz.review import HumanAnnotation, render_move
    render_move(meta, move, source, [HumanAnnotation(), KibitzCommentator()])

The prompt is built by `kibitz.prompt`, the same function that built the training
pairs, so what the model sees at inference is what it saw in training. That
parity is asserted in tests/test_prompt.py rather than left to convention.

mlx is imported lazily: the review harness, the dataset tools and most of the
test suite must work on a machine with no model installed.
"""

from __future__ import annotations

from pathlib import Path

from kibitz import grounding, prompt
from kibitz.dataset import ungrounded_reason

ROOT = Path(__file__).parent.parent
DEFAULT_ADAPTER = ROOT / "adapters"

# Commentary is 1-3 sentences; the corpus median target is ~40 tokens. A tight
# cap keeps generation fast and discourages rambling.
DEFAULT_MAX_TOKENS = 160


def default_base_model() -> str:
    """The base model recorded in configs/model.yaml."""
    import yaml
    config = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    return config["base_model"]


def resolve_adapter(adapter_path) -> Path | None:
    """
    An adapter directory only counts if it actually holds trained weights.

    Returns None for a missing or empty directory. `KibitzCommentator` treats
    that as an error; only `BaselineCommentator` runs without an adapter.
    """
    if adapter_path is None:
        return None
    path = Path(adapter_path)
    return path if (path / "adapters.safetensors").exists() else None


class KibitzCommentator:
    """A fine-tuned model as a source of commentary."""

    def __init__(self, base_model: str | None = None,
                 adapter_path: str | Path | None = DEFAULT_ADAPTER,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temp: float = 0.7,
                 samples: int = 1,
                 name: str = "KIBITZ (generated)"):
        from mlx_lm import load

        self.name = name
        self.base_model = base_model or default_base_model()
        adapter = resolve_adapter(adapter_path)
        # Asked for a fine-tune but there is none: refuse, rather than label the
        # base model "KIBITZ" and let every comparison against it read as base-vs-base.
        if adapter_path is not None and adapter is None:
            raise FileNotFoundError(
                f"no trained adapter at {adapter_path} (expected adapters.safetensors)")
        self.adapter_path = adapter
        self.max_tokens = max_tokens
        self.temp = temp
        self.samples = samples
        self.model, self.tokenizer = load(
            self.base_model, adapter_path=str(adapter) if adapter else None)

    def __repr__(self) -> str:
        tuned = self.adapter_path or "no adapter (base model)"
        return f"KibitzCommentator({self.base_model}, {tuned})"

    def render(self, meta: dict, move: dict, length: str | None = None) -> str:
        """
        The exact text handed to the model, chat template included.

        `length` defaults to what the features call for (`prompt.length_for_move`);
        evaluation overrides it to match the reference it is scored against.
        """
        content = prompt.build_prompt(meta, move, length or prompt.length_for_move(move))
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)

    def comment(self, meta: dict, move: dict, length: str | None = None) -> str | None:
        """
        With `samples` > 1, sample up to that many and return the first comment
        with no detectable problem, or the one with the fewest. Sampling reads
        far more like a person than greedy decoding but invents more; this keeps
        the voice and drops the inventions the checks can see.
        """
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        rendered = self.render(meta, move, length)
        sampler = make_sampler(temp=self.temp)
        best, fewest = None, None
        for _ in range(max(1, self.samples)):
            text = " ".join(generate(self.model, self.tokenizer, prompt=rendered,
                                     max_tokens=self.max_tokens, sampler=sampler,
                                     verbose=False).split()) or None
            if self.samples == 1:
                return text
            n = problems(move, text)
            if fewest is None or n < fewest:
                best, fewest = text, n
            if n == 0:
                break
        return best


def problems(move: dict, text: str | None) -> int:
    """Detectable faults in a comment: false placements, unplayable moves, ungrounded talk."""
    if not text:
        return 1
    g = grounding.check(move, text)
    return len(g["false_placements"]) + len(g["unplayable_moves"]) + bool(ungrounded_reason(text))


class BaselineCommentator(KibitzCommentator):
    """
    The base model with no adapter, prompted identically.

    Worth keeping around: if the fine-tune cannot beat this, the training did
    not buy anything, and comparing them side by side in the review harness is
    the fastest way to see it.
    """

    def __init__(self, base_model: str | None = None, **kwargs):
        kwargs.setdefault("name", "BASELINE (no fine-tune)")
        super().__init__(base_model=base_model, adapter_path=None, **kwargs)
