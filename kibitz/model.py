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

import json
from pathlib import Path

from kibitz import grounding, prompt
from kibitz.dataset import ungrounded_reason

ROOT = Path(__file__).parent.parent
DEFAULT_ADAPTER = ROOT / "adapters"
DEFAULT_GGUF_ADAPTER = ROOT / "adapters" / "kibitz.gguf"

# Commentary is 1-3 sentences; the corpus median target is ~40 tokens. A tight
# cap keeps generation fast and discourages rambling.
DEFAULT_MAX_TOKENS = 160


def _model_config() -> dict:
    import yaml
    return yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())


def default_base_model() -> str:
    """The base model recorded in configs/model.yaml."""
    return _model_config()["base_model"]


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

    def _generate(self, rendered) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        return generate(self.model, self.tokenizer, prompt=rendered, max_tokens=self.max_tokens,
                        sampler=make_sampler(temp=self.temp), verbose=False)

    def comment(self, meta: dict, move: dict, length: str | None = None) -> str | None:
        """
        With `samples` > 1, sample up to that many and return the first comment
        with no detectable problem, or the one with the fewest. Sampling reads
        far more like a person than greedy decoding but invents more; this keeps
        the voice and drops the inventions the checks can see.
        """
        rendered = self.render(meta, move, length)
        best, fewest = None, None
        for _ in range(max(1, self.samples)):
            text = " ".join(self._generate(rendered).split()) or None
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


def _hub_file(repo: str, filename: str) -> Path:
    """A file from the Hugging Face hub, downloaded once and cached."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename))


class ChatTokenizer:
    """
    The base model's own tokenizer and chat template, without transformers.

    llama.cpp's tokenizer splits the spaces around [INST] differently from the
    Hugging Face tokenizer the model was trained with, so prompts for llama.cpp
    are tokenized here, to exactly the ids training used.
    """

    def __init__(self, repo: str):
        import jinja2.sandbox
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(_hub_file(repo, "tokenizer.json")))
        config = json.loads(_hub_file(repo, "tokenizer_config.json").read_text())
        template = config["chat_template"]
        if isinstance(template, list):          # named templates; apply_chat_template uses "default"
            template = next(t["template"] for t in template if t["name"] == "default")

        def raise_exception(message):
            raise ValueError(message)

        # The environment transformers renders chat templates in.
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(template)
        self.special = {"bos_token": config["bos_token"], "eos_token": config["eos_token"]}

    def encode(self, content: str) -> list[int]:
        text = self.template.render(messages=[{"role": "user", "content": content}],
                                    add_generation_prompt=True, **self.special)
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)


class GGUFCommentator(KibitzCommentator):
    """
    The same model through llama.cpp, which runs on Mac, Windows and Linux, CPU
    or GPU: a stock GGUF of the base model with the adapter from
    scripts/convert_gguf.py applied at load time.

    Sampling is plain temperature sampling, as in mlx-lm: llama.cpp's default
    top-k, top-p and min-p filters are switched off so the two backends draw
    from the same distribution.
    """

    def __init__(self, adapter_path: str | Path | None = DEFAULT_GGUF_ADAPTER,
                 base_gguf: str | Path | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temp: float = 0.7,
                 samples: int = 1,
                 name: str = "KIBITZ (generated)"):
        # Same rule as the MLX class: a requested adapter that is not there is an error.
        if adapter_path is not None and not Path(adapter_path).is_file():
            raise FileNotFoundError(
                f"no GGUF adapter at {adapter_path} (convert one with scripts/convert_gguf.py)")
        from llama_cpp import Llama

        config = _model_config()
        self.name = name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.base_model = str(base_gguf or _hub_file(config["gguf_base"]["repo"],
                                                     config["gguf_base"]["file"]))
        self.max_tokens = max_tokens
        self.temp = temp
        self.samples = samples
        self.tokenizer = ChatTokenizer(config["tokenizer_repo"])
        self.model = Llama(self.base_model,
                           lora_path=str(self.adapter_path) if self.adapter_path else None,
                           n_ctx=1024, n_gpu_layers=-1, verbose=False)

    def render(self, meta: dict, move: dict, length: str | None = None) -> list[int]:
        """The token ids handed to the model, chat template included."""
        content = prompt.build_prompt(meta, move, length or prompt.length_for_move(move))
        return self.tokenizer.encode(content)

    def _generate(self, rendered: list[int]) -> str:
        out = []
        for token in self.model.generate(rendered, temp=self.temp, top_k=0, top_p=1.0,
                                         min_p=0.0, repeat_penalty=1.0, reset=True):
            if token == self.model.token_eos() or len(out) >= self.max_tokens:
                break
            out.append(token)
        return self.tokenizer.decode(out)
