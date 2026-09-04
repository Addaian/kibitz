# Kibitz

> Fine-tuned LLM commentary for PGN chess games.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Overview](#2-architecture-overview)
3. [Repository Structure](#3-repository-structure)
4. [Data Pipeline](#4-data-pipeline)
   - 4.1 [Sourcing PGN Games](#41-sourcing-pgn-games)
   - 4.2 [Parsing PGN Files](#42-parsing-pgn-files)
   - 4.3 [Generating Training Pairs](#43-generating-training-pairs)
     - 4.3.1 [Enrichment (`features.py`)](#431-enrichment-featurespy)
     - 4.3.2 [Filtering & Shaping (`build_dataset.py`)](#432-filtering--shaping-build_datasetpy)
   - 4.4 [Dataset Format](#44-dataset-format)
5. [Feature Engineering](#5-feature-engineering)
   - 5.1 [Position Evaluation](#51-position-evaluation)
   - 5.2 [Move Classification](#52-move-classification)
   - 5.3 [Game Phase Detection](#53-game-phase-detection)
   - 5.4 [Tactical Motif Detection](#54-tactical-motif-detection)
6. [Model Selection](#6-model-selection)
7. [Fine-Tuning](#7-fine-tuning)
   - 7.1 [Prompt Template Design](#71-prompt-template-design)
   - 7.2 [Training Configuration](#72-training-configuration)
   - 7.3 [LoRA / PEFT (Parameter-Efficient Fine-Tuning)](#73-lora--peft)
   - 7.4 [Evaluation Metrics](#74-evaluation-metrics)
8. [Inference Pipeline](#8-inference-pipeline)
9. [CLI Design](#9-cli-design)
10. [Testing Strategy](#10-testing-strategy)
11. [Stretch Goals](#11-stretch-goals)
12. [Recommended Libraries](#12-recommended-libraries)
13. [Reference Material](#13-reference-material)
14. [Appendix A — Decision Log](#appendix-a--decision-log)
15. [Appendix B — Running the Pipeline](#appendix-b--running-the-pipeline)

---

## 1. Project Overview

**Kibitz** takes a chess game in [PGN format](https://en.wikipedia.org/wiki/Portable_Game_Notation) and produces human-readable, move-by-move commentary — the kind a knowledgeable spectator or analyst might offer.

**Example input:**
```
[Event "World Championship"]
[White "Kasparov, G"]
[Black "Deep Blue"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 ...
```

**Example output:**
```
1. e4 — Kasparov opens with the King's Pawn, staking a claim to the center.
   c5 — Deep Blue responds with the Sicilian Defense, one of the sharpest replies to 1.e4.

2. Nf3 — Developing the knight toward the center, preparing to challenge the c5 pawn.
   d6 — Reinforcing the center and freeing the dark-squared bishop.
...
```

The project is split into three major phases:
1. **Data** — sourcing, parsing, and structuring annotated chess games into training pairs.
2. **Model** — fine-tuning a base LLM on those pairs to produce commentary.
3. **Inference** — a CLI (and optionally a web UI) that accepts a PGN and streams commentary.

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        Kibitz Pipeline                       │
│                                                              │
│  PGN File                                                    │
│     │                                                        │
│     ▼                                                        │
│  PGN Parser          (python-chess)                         │
│     │                                                        │
│     ▼                                                        │
│  Feature Extractor   (Stockfish + custom heuristics)        │
│     │                                                        │
│     ▼                                                        │
│  Prompt Builder      (structured context → prompt string)   │
│     │                                                        │
│     ▼                                                        │
│  Fine-tuned LLM      (base model + LoRA adapter)            │
│     │                                                        │
│     ▼                                                        │
│  Commentary Output   (per-move or full-game)                │
└──────────────────────────────────────────────────────────────┘
```

The key insight is that **raw PGN is not enough context** for an LLM to produce good commentary. You need to enrich each move with structured features — evaluation deltas, tactical motifs, game phase — before handing it to the model.

---

## 3. Repository Structure

```
kibitz/
├── data/
│   ├── raw/                  # Raw PGN files (not committed)
│   ├── processed/            # Parser output: one JSONL per PGN source
│   ├── enriched/             # Feature-enriched JSONL (after features.py)
│   └── datasets/             # Final train/val/test splits
│
├── kibitz/
│   ├── __init__.py
│   ├── parser.py             # PGN parsing logic
│   ├── features.py           # Feature extraction (Stockfish, heuristics)
│   ├── prompt.py             # Prompt template construction
│   ├── model.py              # Model loading, fine-tuning, inference
│   └── cli.py                # Entry point
│
├── scripts/
│   ├── build_dataset.py      # Runs full data pipeline
│   ├── train.py              # Fine-tuning entry point
│   └── evaluate.py           # Runs evaluation metrics
│
├── tests/
│   ├── test_parser.py
│   ├── test_features.py
│   └── test_prompt.py
│
├── configs/
│   ├── training.yaml         # Hyperparameters
│   └── model.yaml            # Model selection & LoRA config
│
├── notebooks/                # Exploratory work (not production)
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 4. Data Pipeline

Data enhancement happens in two distinct stages with different purposes:

```
PGN → parser.py → data/processed/ → features.py → data/enriched/ → build_dataset.py → data/datasets/
```

- **`features.py`** — enrichment: adds computed chess features that aren't in the PGN
- **`build_dataset.py`** — cleaning & shaping: filters, normalizes, and formats records into training pairs

### 4.1 Sourcing PGN Games

You need games that already have **human-written annotations** — not just raw moves. Annotation quality directly determines commentary quality.

**Source: Lichess Studies**

All training data is sourced from public [Lichess Studies](https://lichess.org/study) — annotated collections created by titled players, coaches, and prolific community contributors. Studies are exported via the Lichess API with human comments and Stockfish evaluations included.

```
GET https://lichess.org/api/study/by/{username}/export.pgn?comments=true&variations=true&clocks=true
```

Studies were selected from known annotators including GMs, IMs, and FMs with large public study libraries. Raw PGN files are stored in `data/raw/lichess_studies/`.

**Why Lichess Studies:**
- Human-written commentary on moves (the training target)
- Stockfish `%eval` scores embedded alongside human annotations
- Free, programmatically downloadable, CC-licensed

**Target dataset size:**
- Minimum viable: ~5,000 annotated move pairs
- Comfortable: ~50,000–100,000 move pairs
- Diminishing returns above ~500,000 for a single-domain fine-tune

**What the corpus actually yielded** (33 study collections, 52 MB of PGN):

| stage | count |
|---|---:|
| games parsed | 42,252 |
| plies | 1,205,315 |
| moves with a real prose comment | 82,175 |
| after cleaning and filtering | **47,520 training pairs** |

Note the attrition: only **6.8%** of plies carry human prose. Of the brace comments in
the raw PGN, **71% are `[%clk]` / `[%eval]` tags with no text at all** — sizing the job
off a raw `grep` for `{...}` overestimates the corpus by roughly 6x.

---

### 4.2 Parsing PGN Files

Use the [`python-chess`](https://python-chess.readthedocs.io/) library. It handles the full PGN spec including variations, comments, NAGs (Numeric Annotation Glyphs like `!`, `?`, `!!`), and FEN positions.

**Key objects you'll use:**
- `chess.pgn.read_game(handle)` — reads one game at a time from a PGN file
- `game.headers` — metadata dict (`Event`, `White`, `Black`, `Result`, etc.)
- `node.move` — the `chess.Move` object for each node
- `node.comment` — human comment string attached to the move (may be empty)
- `node.nags` — set of NAG values (see `chess.pgn.NAG_*` constants)
- `node.board()` — the `chess.Board` after this move is played

**Decision — variations are stripped.** The parser walks `game.mainline()` only, so
sideline commentary is discarded. Sidelines describe moves that were never played, so
their comments do not pair with a real position in the game. This costs some annotated
moves and could be revisited if the corpus ever needs to be larger.

**Watch out for:**
- Encoding issues in older PGN files (Latin-1 vs UTF-8)
- Malformed PGN — `read_game()` itself raises on some files; unguarded, it aborts the
  whole file part-written. Guard the read, not just the move loop.

**What the parser already gives you (no engine needed):**
- `san`, `uci`, `fen_before`, `fen_after` per move
- `comment` — the human annotation (training target)
- `nags` — NAG codes (`!`, `?`, `!!`, etc.)
- `eval` — centipawn score if the PGN includes `%eval` tags (common in Lichess exports)
- Game metadata: players, event, date, ECO, opening name

This is stored in `data/processed/`. The next step (`features.py`) reads these files and computes additional features that the PGN does not contain.

---

### 4.3 Generating Training Pairs

Training pairs are produced in two sequential stages.

#### 4.3.1 Enrichment (`features.py`)

This is the primary enhancement stage. It reads from `data/processed/` and writes enriched records to `data/enriched/`. It adds computed features that the PGN does not contain:

- **Game phase** — opening / middlegame / endgame via material count heuristic
- **Move classification** — best / good / inaccuracy / mistake / blunder, derived from eval delta thresholds. Anything orthogonal to *quality* stays its own field (`is_sacrifice`, `is_forcing`, `tactical_motifs`), since a move can be a sacrifice **and** the best move on the board
- **Tactical motifs** — fork, pin, skewer, discovery, etc., detected via board geometry
- **Stockfish evals** — `eval_before`, `eval_after`, `eval_delta` if not already present as `%eval` tags in the PGN

After enrichment, each record looks like:

```python
{
  "move_number": 24, "color": "white", "ply": 46,
  "san": "cxb6", "uci": "c5b6",
  "fen_before": "...", "fen_after": "...",
  "nags": [],

  # engine (White's point of view, pawns)
  "eval_before": 1.30, "eval_after": 2.79,
  "best_move": "d2c2", "is_best": False,
  "mate_in": None, "is_mate_threat": False,

  # derived judgement
  "eval_delta": 1.49,                 # MOVER's point of view, pawns
  "move_classification": "good",
  "game_phase": "middlegame",
  "tactical_motifs": ["fork"],
  "is_forcing": True,
  "is_sacrifice": False, "material_offered": -1.0,

  # target
  "comment": "A\nsurprising but not very...",   # raw, as written in the PGN
  "comment_text": "A surprising but not very..."  # [%clk]/[%eval] tags stripped
}
```

Game-level fields (`game_id`, `source`, `metadata.white/black/event/eco/opening/variant`,
`total_plies`) live on the enclosing game record, not repeated per move.

**Point-of-view gotcha.** `eval_before` / `eval_after` are White-relative, the engine
convention. `eval_delta` is **mover-relative**, so positive always means "this move
helped the side that played it". Mixing the two up silently scores every strong Black
move as a blunder.

The `comment` field is the **training target**. Everything else is **input context**.

#### 4.3.2 Filtering & Shaping (`build_dataset.py`)

This stage reads from `data/enriched/` and writes final splits to `data/datasets/`. Its responsibilities are cleaning and formatting — not adding chess knowledge:

- **Strip Lichess auto-analysis.** Roughly one annotated move in ten carries machine
  text — `"Inaccuracy. Best move was Qb6."` — often repeated verbatim several times.
  It is *stripped, not dropped*: a minority of those comments have real human prose
  behind the machine prefix, and dropping the record would lose it.
- **Strip the result before measuring length.** `"Checkmate! (1-0)"` is sixteen
  characters that say nothing about the position. The length bar applies to the
  analysis, not to punctuation.
- **Normalize whitespace** — collapse the newlines PGN wrapping introduces mid-sentence
- **Balance across move classifications** — available as `--max-per-class`, applied to
  **train only**. Validation and test keep the corpus's real distribution, because a
  metric measured on a rebalanced set does not describe behaviour on real games.
  Off by default: the right cap is a modelling decision, not a pipeline one.
- **Format as Messages pairs:**
  ```json
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  ```
  This is a **neutral interchange format**, not a commitment to a provider. `trl`'s
  `SFTTrainer` consumes it directly and applies the base model's chat template at
  training time, so the same file trains Mistral, Llama or Qwen without regeneration.
  The prompt *content* is built by `kibitz/prompt.py`; the chat *markup* (§7.1) belongs
  to whichever base model §6 selects. Keeping those two separate is what stops the
  dataset from being welded to one model family.
- **Split by game** (not move) into `train.jsonl`, `val.jsonl`, `test.jsonl` to avoid data leakage

---

### 4.4 Dataset Format

Store training data as **JSONL** (one JSON object per line). Split into:

- `train.jsonl` — 80%
- `val.jsonl` — 10%
- `test.jsonl` — 10%

Split **by game**, not by move, to avoid data leakage (consecutive moves from the same game are highly correlated).

---

## 5. Feature Engineering

The parser captures what is *written* in the PGN. `features.py` computes what must be *derived* — things that require chess logic or an engine. It reads from `data/processed/` and writes enriched records to `data/enriched/`. A second stage, `build_dataset.py`, then reads those enriched records and applies filtering and shaping to produce the final training pairs in `data/datasets/`.

Raw move notation alone is insufficient. The model needs to understand *why* a move is interesting. These features provide that signal.

### 5.1 Position Evaluation

Use [Stockfish](https://stockfishchess.org/) via python-chess's UCI interface.

```python
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/path/to/stockfish")
info = engine.analyse(board, chess.engine.Limit(depth=18))
score = info["score"].relative.score(mate_score=10000)  # centipawns
engine.quit()
```

**Derived features:**
- `eval_before`, `eval_after` — raw centipawn scores
- `eval_delta` — the swing; large positive = good move, large negative = blunder
- `mate_in` — signed mate distance from White's point of view, else `None`
- `is_mate_threat` — boolean, is the forced mate *in the mover's favour*?

**Mate scores are not centipawns.** They are clamped to ±20.00 pawns before any
arithmetic, so an eval delta stays a meaningful number. About 5% of annotated moves sit
on a mate score, and their `eval_delta` is therefore saturated — `mate_in` carries the
real information for those.

**Performance note:** Engine analysis is slow. Pre-compute and cache all evals during the data pipeline stage, not at inference time. At inference, you can optionally run a quick shallow analysis (depth 10–12) if real-time evals are desired.

---

### 5.2 Move Classification

Classify each move into one of these categories based on eval delta thresholds:

| Classification | Criteria |
|---|---|
| `best` | Matches engine's top choice |
| `good` | Small positive or neutral eval delta |
| `inaccuracy` | Eval drops by ~0.3–0.5 pawns |
| `mistake` | Eval drops by ~0.5–1.5 pawns |
| `blunder` | Eval drops by >1.5 pawns |
| `forcing` | Check, capture, or forced-reply move |

These can be derived from Stockfish output. NAGs from the PGN source provide human-assigned labels that can supplement or override.

**Classification is one axis: quality.** Two labels from the table above are deliberately not part of it.

- `brilliant` — a chess.com construct with no counterpart in Lichess studies. It means "hard to spot", a property of the player rather than the position, which an eval delta cannot measure.
- `sacrifice` and `forcing` — both real and objective, but statements about *material* and *forcedness*, not about quality. They live in `is_sacrifice` / `material_offered` and `is_forcing`. 78% of detected sacrifices are also the engine's top move; folding them into the classification would discard that.

---

### 5.3 Game Phase Detection

```python
def detect_phase(board: chess.Board) -> str:
    total_material = sum(
        len(board.pieces(pt, color)) * value
        for color in chess.COLORS
        for pt, value in [
            (chess.QUEEN, 9), (chess.ROOK, 5),
            (chess.BISHOP, 3), (chess.KNIGHT, 3)
        ]
    )
    if board.fullmove_number <= 10:
        return "opening"
    elif total_material > 20:
        return "middlegame"
    else:
        return "endgame"
```

You can refine this with opening book detection (see ECO codes in PGN headers).

---

### 5.4 Tactical Motif Detection

These are harder but valuable. Consider detecting:

- **Fork** — one piece attacks two or more enemy pieces simultaneously
- **Pin** — a piece is pinned to a more valuable piece behind it
- **Skewer** — like a pin, but the more valuable piece is in front
- **Discovery** — moving a piece reveals an attack from another piece
- **Promotion** — the move promotes; **Promotion threat** — a pawn reaches the 7th

**Not implemented: Zwischenzug.** It is defined by what the opponent was *expected* to
play, so detecting it needs a search over the expected sequence rather than board
geometry. Every other motif here is a static property of the resulting position.

Motifs are a **list**, not a category — a move can fork and threaten promotion at once.

python-chess gives you full board access to implement these as geometric checks. Start simple — fork and pin detection are the most tractable.

---

## 6. Model Selection

You're fine-tuning an **instruction-following LLM** on a domain-specific task. Key considerations:

| Model | Size | Notes |
|---|---|---|
| `Mistral-7B-Instruct-v0.2` | 7B | Strong baseline, good instruction following, widely supported |
| `LLaMA-3-8B-Instruct` | 8B | Meta's latest small model, competitive with Mistral |
| `Phi-3-mini-4k-instruct` | 3.8B | Very efficient, good for constrained compute |
| `Gemma-2-9B-it` | 9B | Google's instruct model, strong reasoning |
| `Qwen2.5-7B-Instruct` | 7B | Strong multilingual + reasoning, good for technical text |

**Recommendation:** Start with `Mistral-7B-Instruct-v0.2` or `LLaMA-3-8B-Instruct`. Both are well-documented, have active communities, and work well with LoRA fine-tuning on a single GPU.

**Hardware requirements for fine-tuning (LoRA, 4-bit quantized):**
- Minimum: 16GB VRAM (RTX 3090 / RTX 4080 / A4000)
- Comfortable: 24GB VRAM (RTX 3090 Ti / A5000)
- Cloud alternative: A100 40GB on Lambda Labs, Vast.ai, or Google Colab Pro+

---

## 7. Fine-Tuning

### 7.1 Prompt Template Design

Your prompt template must be consistent between training and inference. Here is a recommended structure using the Mistral instruct format:

```
<s>[INST] You are Kibitz, an expert chess commentator. Given the following game context and move, write a short, insightful commentary in the style of a grandmaster analyst.

Game: {{ white_player }} vs. {{ black_player }} ({{ event }}, {{ year }})
Opening: {{ opening_name }}
Phase: {{ game_phase }}
Move {{ move_number }} ({{ side }}): {{ san }}
Classification: {{ move_classification }}
Evaluation: {{ eval_before }} → {{ eval_after }} (Δ{{ eval_delta }})
{% if nags %}Annotation: {{ nag_description }}{% endif %}

Write 1–3 sentences of commentary for this move. [/INST]
{{ comment }}</s>
```

**Design decisions to make:**
- How verbose should the prompt context be? More context = better quality but slower inference.
- Do you want the model to explain *why* a move is a blunder, or just describe what it does?
- Should commentary vary in length based on move importance (blunders get more text, quiet moves get less)?

---

### 7.2 Training Configuration

A baseline `training.yaml`:

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  quantization: "4bit"          # Use bitsandbytes NF4

lora:
  r: 16
  lora_alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

training:
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4   # Effective batch size = 16
  learning_rate: 2.0e-4
  warmup_ratio: 0.03
  lr_scheduler_type: "cosine"
  max_seq_length: 512
  fp16: true
  logging_steps: 50
  save_steps: 200
  eval_steps: 200
  save_total_limit: 3
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
```

---

### 7.3 LoRA / PEFT

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) fine-tunes only a small number of additional parameters (~1–2% of the model) rather than the full weights. This makes fine-tuning feasible on consumer hardware.

**Recommended stack:**
- [`peft`](https://github.com/huggingface/peft) — Hugging Face PEFT library for LoRA
- [`transformers`](https://github.com/huggingface/transformers) — model loading & training
- [`trl`](https://github.com/huggingface/trl) — `SFTTrainer` for supervised fine-tuning
- [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) — 4-bit quantization
- [`datasets`](https://github.com/huggingface/datasets) — dataset loading and processing
- [`accelerate`](https://github.com/huggingface/accelerate) — multi-GPU / mixed precision

**Training outline (`scripts/train.py`):**
1. Load base model in 4-bit with `BitsAndBytesConfig`
2. Apply LoRA config via `get_peft_model()`
3. Load and tokenize your JSONL dataset
4. Use `SFTTrainer` with your training config
5. Save the LoRA adapter (not the full model weights) to disk
6. At inference time: load base model + merge adapter

---

### 7.4 Evaluation Metrics

Automated metrics for commentary generation are imperfect but useful for tracking training progress:

| Metric | Tool | Notes |
|---|---|---|
| `eval_loss` | Built into training loop | Primary signal during training |
| `BLEU` | `sacrebleu` | Measures n-gram overlap with reference commentary |
| `ROUGE-L` | `rouge-score` | Measures longest common subsequence |
| `BERTScore` | `bert-score` | Semantic similarity; more meaningful than BLEU |
| `Perplexity` | Compute from model | How "surprised" is the model by the reference text |

**Human evaluation** is the gold standard. Reserve ~100 games from your test set and manually assess:
- Factual accuracy (did the model describe what the move actually does?)
- Insight (does the comment go beyond the obvious?)
- Fluency (does it read naturally?)

---

## 8. Inference Pipeline

At inference time, Kibitz should:

1. Accept a PGN file path or string as input
2. Parse the game with python-chess
3. For each move, extract features (optionally run Stockfish at shallow depth)
4. Build the prompt from the template
5. Run the fine-tuned model to generate commentary
6. Stream or batch output to stdout / file

**Latency considerations:**
- Without Stockfish: ~0.5–2s per move (GPU inference)
- With Stockfish (depth 12): ~1–3s per move
- For a 40-move game: 40–120s total without batching

**Batching:** If you don't need streaming output, batch all prompts for the game and run them through the model in parallel. This can reduce wall-clock time significantly.

**Output formats to support:**
- `--format text` — plain annotated game, moves with comments inline
- `--format pgn` — valid PGN with `{ ... }` comment blocks
- `--format json` — structured output for programmatic use

---

## 9. CLI Design

```
kibitz [OPTIONS] <pgn_file>

Options:
  --output-format [text|pgn|json]   Output format (default: text)
  --stockfish PATH                  Path to Stockfish binary
  --depth INT                       Engine analysis depth (default: 12)
  --model PATH                      Path to fine-tuned model or adapter
  --stream                          Stream commentary move by move
  --moves RANGE                     Only annotate moves e.g. "10-20"
  --verbose                         Include feature data in output
  --help                            Show this message and exit.

Examples:
  kibitz game.pgn
  kibitz game.pgn --output-format pgn --depth 15
  kibitz game.pgn --moves 15-30 --stream
  echo "<pgn string>" | kibitz -
```

Use [`click`](https://click.palletsprojects.com/) or [`typer`](https://typer.tiangolo.com/) for CLI construction.

---

## 10. Testing Strategy

| Test type | What to cover |
|---|---|
| Unit | PGN parser handles edge cases (castling, en passant, promotion) |
| Unit | Feature extractor returns correct phase/classification |
| Unit | Prompt builder produces correctly-formatted strings |
| Unit | Eval delta computation is correct (sign, scale) |
| Integration | Full pipeline on a known game produces output without crashing |
| Integration | PGN output is valid (round-trippable through python-chess) |
| Regression | A set of reference games produce consistent output across versions |

Use `pytest`. Store reference PGN files in `tests/fixtures/`.

---

## 11. Stretch Goals

These are not required for v1 but worth considering:

- **Opening book integration** — use a Polyglot `.bin` opening book to name openings more precisely than ECO codes alone
- **Variation commentary** — comment on the "best line" that wasn't played (requires Stockfish + prompt engineering)
- **Style modes** — e.g. `--style grandmaster`, `--style beginner`, `--style entertaining`
- **Web UI** — a simple FastAPI backend + React frontend with a PGN board viewer (chessboard.js or chess-board component)
- **Lichess integration** — accept a Lichess game URL directly and fetch the PGN via the Lichess API
- **Streaming over WebSocket** — push commentary move-by-move to a live viewer
- **Multilingual output** — the base model likely has multilingual capability; a `--lang` flag could route to language-specific LoRA adapters

---

## 12. Recommended Libraries

```
# Chess
python-chess          # PGN parsing, board logic, Stockfish UCI interface

# Data
datasets              # Hugging Face datasets
jsonlines             # JSONL read/write

# ML / Fine-tuning
torch
transformers
peft
trl
bitsandbytes
accelerate

# Evaluation
sacrebleu
rouge-score
bert-score

# CLI
typer
rich                  # Pretty terminal output

# Testing
pytest

# Optional
fastapi               # Web API
uvicorn               # ASGI server
click                 # Alternative CLI library
```

Install Stockfish separately as a system binary — it is not a Python package.

---

## 13. Reference Material

**Chess & PGN**
- [PGN Standard Specification](http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm)
- [python-chess documentation](https://python-chess.readthedocs.io/)
- [Stockfish UCI protocol](https://www.chessprogramming.org/UCI)
- [Numeric Annotation Glyphs (NAGs)](https://en.wikipedia.org/wiki/Numeric_Annotation_Glyph)
- [ECO Opening Codes](https://www.365chess.com/eco.php)

**Fine-tuning**
- [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT docs](https://huggingface.co/docs/peft)
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)

**Evaluation**
- [BERTScore paper](https://arxiv.org/abs/1904.09675)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)

---

*Good luck. The position is yours.

---

## Appendix A — Decision Log

**Rule: a deviation from this document is not done until it appears in this table.**
Every row below was a silent deviation first and got caught later by someone reading the
output and asking why. The cost of a silent deviation is that the same mistake stays
invisible in the rows next to it — `forcing` was demoted for exactly the reason
`sacrifice` should have been, and the gap between the two went unnoticed for a full day.

| # | The plan said | What we do | Why |
|---|---|---|---|
| 1 | `brilliant` classification | dropped | A chess.com construct with no Lichess counterpart. It means "hard to spot" — a property of the player, not the position, which an eval delta cannot measure. |
| 2 | `sacrifice` classification | `is_sacrifice` + `material_offered` | Orthogonal to quality. 78% of detected sacrifices are also the engine's top move, and the label was overwriting `best`. |
| 3 | `forcing` classification | `is_forcing` | Same reason: a move can be forcing *and* a blunder. |
| 4 | `side` | `color` | Naming only. |
| 5 | `depth_of_mate` | `mate_in` | Signed and White-relative, so it composes with the eval fields. |
| 6 | `opening_name` per move | `metadata.opening` per game | It is a property of the game, not the move. |
| 7 | Zwischenzug detection | not implemented | Defined by the opponent's *expected* reply, so it needs a search rather than board geometry. |
| 8 | "decide whether to include variations" | stripped | Sidelines comment on moves that were never played, so they have no real position to pair with. |
| 9 | balance across classifications | opt-in, train split only | A metric measured on a rebalanced val/test set does not describe behaviour on real games. |
| 10 | "Claude Messages API pairs" | Messages as neutral interchange | Decouples the dataset from any one model family; the chat template is applied at training time. |

**Fields we added that the plan never specified:** `is_best`, `best_move` (both free from
the same engine call), `material_offered`, `comment_text`, `ply`, `total_plies`.

**Classification is one axis: move quality.** `best` / `good` / `inaccuracy` / `mistake` /
`blunder`. Anything orthogonal to quality is its own field. When a new label needs a
precedence rule against an existing one, that is the signal it belongs on a different
axis — not that it needs a tie-break.

---

## Appendix B — Running the Pipeline

```bash
python3 scripts/download_data.py                      # -> data/raw/        (~52 MB)
python3 -m kibitz.parser                              # -> data/processed/  (~3 min)
python3 -m kibitz.features --depth 14 --workers 8     # -> data/enriched/   (~40 min)
python3 scripts/build_dataset.py                      # -> data/datasets/
python3 scripts/inspect.py stats                      # look at what you got
```

**Stockfish exits on illegal positions, and stays dead.** A kingless pawn diagram — which
studies contain as teaching material — makes it terminate with SIGSEGV, after which every
later analysis in that worker fails. `analyse()` therefore refuses any position failing
`board.is_valid()`, and restarts the engine if it dies anyway. Without both, one bad
diagram silently poisons every game a worker touches afterwards, and the run still exits 0.

**Variant games are skipped** (478 of them: Chess960, Crazyhouse, Antichess, Horde,
Racing Kings, Three-check, King of the Hill, Atomic). A standard engine's evaluation of
an Antichess position is meaningless, and its commentary describes different tactics.

**Only prose-commented moves are analysed.** A move with no comment has no training
target, so evaluating it is wasted engine time — this is a ~25x saving.

**Depth 14, not 18.** Five times cheaper, and the classification thresholds (0.3 / 0.5 /
1.5 pawns) are far coarser than the difference between the two depths.

**Changing a label does not need the engine.** `python3 -m kibitz.features --relabel`
replays classification, phase, motifs and the sacrifice test against the evals already on
disk — 10 seconds for the whole corpus instead of 40 minutes. Re-freeze the golden fixture
(`scripts/inspect.py freeze`) and rebuild the datasets afterwards.
