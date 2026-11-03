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
   - 5.5 [Threats and Loose Pieces](#55-threats-and-loose-pieces)
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
12. [Libraries](#12-libraries)
13. [Reference Material](#13-reference-material)
14. [Appendix A — Decision Log](#appendix-a--decision-log)
15. [Appendix B — Running the Pipeline](#appendix-b--running-the-pipeline)
16. [Appendix C — Training on a Rented GPU](#appendix-c--training-on-a-rented-gpu)

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
│   ├── raw/                  # Raw PGN files (not committed, ~55 MB)
│   ├── processed/            # Parser output: one JSONL per PGN source
│   ├── enriched/             # Feature-enriched JSONL (after features.py)
│   └── datasets/             # Final train/val/test splits
│
├── kibitz/
│   ├── __init__.py
│   ├── parser.py             # PGN parsing logic
│   ├── features.py           # Feature extraction (Stockfish, heuristics)
│   ├── dataset.py            # Cleaning, filtering and splitting training pairs
│   ├── prompt.py             # Prompt construction, shared by training and inference
│   ├── model.py              # Model loading and generation (llama.cpp, or mlx-lm)
│   ├── stockfish.py          # Finds Stockfish, or downloads the official build
│   ├── grounding.py          # Checks commentary claims against the board
│   ├── review.py             # Side-by-side review panels (human vs model)
│   ├── tui.py                # Shared terminal rendering
│   └── cli.py                # Entry point: PGN in, commentary out (§9)
│
├── scripts/
│   ├── download_data.py      # Fetches Lichess studies
│   ├── build_dataset.py      # enriched -> train/val/test
│   ├── inspect_dataset.py    # Stats, side-by-side browsing, golden fixture
│   ├── train.py              # LoRA fine-tune via mlx-lm
│   └── evaluate.py           # Held-out metrics, incl. the board check
│
├── tests/                    # pytest; fixtures/golden.jsonl is the regression set
│
├── configs/
│   ├── training.yaml         # Hyperparameters
│   └── model.yaml            # Model selection & LoRA config
│
├── adapters/                 # Trained LoRA adapters (not committed)
├── runs/                     # Evaluation generations (not committed)
├── notebooks/                # Exploratory work (not production)
├── requirements.txt          # everything, for a checkout
├── pyproject.toml            # the installable `kibitz` command (§9)
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
- Stockfish `%eval` scores embedded alongside human annotations (not used: every
  position is re-analysed at a fixed depth so all evals are comparable — §5.1)
- Free, programmatically downloadable, CC-licensed

**Target dataset size:**
- Minimum viable: ~5,000 annotated move pairs
- Comfortable: ~50,000–100,000 move pairs
- Diminishing returns above ~500,000 for a single-domain fine-tune

**What the corpus actually yielded** (38 study collections, 55 MB of PGN):

| stage | count |
|---|---:|
| games parsed | 45,128 |
| plies | 1,289,108 |
| moves with a real prose comment | 86,897 |
| after the length bar, boilerplate, link and study-scaffolding stripping | 50,012 |
| after dropping non-English and non-prose targets | 46,343 |
| after dropping targets that rely on facts outside the prompt | **42,758 training pairs** |

Note the attrition: only **6.7%** of plies carry human prose. Of the brace comments in
the raw PGN, **71% are `[%clk]` / `[%eval]` tags with no text at all** — sizing the job
off a raw `grep` for `{...}` overestimates the corpus by roughly 6x.

The corpus also comes from only 38 annotators, and five of them supply 68% of the
pairs (one supplies 28%). Style learned from it is largely their style. The last five
added (Juggernaunt, Sack_o_Potatoes, WassimBerbar, BenedictChessman, SarahLongson) were
found through links and `[Annotator]` credits in the others and contribute 3,540 pairs
(7.5%), nearly all opening and trap tutorials — so they add opening prose, not middlegame
or endgame judgement.

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
- `eval` — score in pawns if the PGN includes `%eval` tags (common in Lichess exports; discarded by `features.py`)
- Game metadata: players, event, date, ECO, opening name, and the Lichess `study` id
  (from the `ChapterURL` header), which the dataset split is keyed on

This is stored in `data/processed/`. The next step (`features.py`) reads these files and computes additional features that the PGN does not contain.

---

### 4.3 Generating Training Pairs

Training pairs are produced in two sequential stages.

#### 4.3.1 Enrichment (`features.py`)

This is the primary enhancement stage. It reads from `data/processed/` and writes enriched records to `data/enriched/`. It adds computed features that the PGN does not contain:

- **Game phase** — opening / middlegame / endgame via material count heuristic
- **Move classification** — best / good / inaccuracy / mistake / blunder, derived from eval delta thresholds. Anything orthogonal to *quality* stays its own field (`is_sacrifice`, `is_forcing`, `tactical_motifs`), since a move can be a sacrifice **and** the best move on the board
- **Tactical motifs** — fork, pin, skewer, discovery, etc., detected via board geometry
- **Stockfish evals** — `eval_before`, `eval_after`, `eval_delta`, always from a fresh depth-14 analysis; the PGN's own `%eval` tags are discarded

After enrichment, each record looks like:

```python
{
  "move_number": 24, "color": "white", "ply": 46,
  "san": "cxb6", "uci": "c5b6",
  "fen_before": "...", "fen_after": "...",
  "nags": [],

  # engine (White's point of view, pawns)
  "eval_before": 1.30, "eval_after": 2.79,
  "best_move": "d2c2", "best_move_san": "Qc2", "is_best": False,
  "mate_in": None, "is_mate_threat": False,

  # board facts after the move (no engine; approximate exchange evaluation)
  "captured": "pawn",
  "threatens": ["rook on a8"],        # opponent pieces the mover could win next
  "left_hanging": [],                 # mover's pieces the opponent could win

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
  text — `"Inaccuracy. Best move was Qb6."`, `"Lost forced checkmate sequence. Qf5+
  was best."` — often repeated verbatim several times.
  It is *stripped, not dropped*: a minority of those comments have real human prose
  behind the machine prefix, and dropping the record would lose it.
- **Drop targets that are not English commentary.** About 7% of what survives the
  length bar is something else: Turkish and Spanish prose, engine logs pasted by hand
  (`d=22, pd=Re1, pv=…`), bare ECO labels (`aka C01 French Defense: Exchange
  Variation.`) and game citations. `dataset.non_prose_reason` names which, and
  `build_dataset.py` reports the counts. Language is judged by the share of common
  English words — crude, but it was checked by reading what it drops.
- **Strip links.** Annotators link studies and videos mid-sentence; 2% of targets did.
  The link is removed and the prose kept, so the model never learns to emit URLs.
- **Strip study scaffolding.** Puzzle verdicts (`"Correct."`, `"Exactly!"`), initials
  used as speaker tags (`"AL:"`), click instructions (`"Play ...Kg7 for Black."`) and
  resignation sentences are removed. The prose around them is usually good, so trimming
  them keeps the comment. 2,308 targets were trimmed.
- **Drop targets that rely on facts outside the prompt.** The first 7B fine-tune
  invented win rates, move popularity and material counts. 2% of its training targets
  talked that way, and it did so 3.5 times as often. `dataset.ungrounded_reason` drops
  four kinds:

  | Kind | Dropped | Example |
  |---|---:|---|
  | addresses the reader | 1,954 | "What should White play?", "the arrows show…", chapters, videos |
  | database statistics | 1,078 | "most popular", "62% win rate", "LMDB" |
  | other games | 415 | "was played in Votava vs Lujan", "In Game 5…", "already seen" |
  | hindsight | 138 | "went on to win", "the rest of the game" |

  The rules came from hand-labelling 350 targets: 50 per candidate rule, plus 50 that no
  rule flagged (`runs/target_audit.csv`). A rule was kept only where nearly every match
  deserved dropping. Pattern matches that were usually fine were left out: move numbers
  (mostly alternatives like "11.c3"), "eventually" (mostly plans), and the board check,
  which flags quoted variations and piece names like "the Bc6". Against those labels the
  rules are right 90% of the time when they drop, and catch 75% of what should go.
  The rest is semantic: a quarter of the comments no rule flags still cite other games,
  trivia or outside opinions. Catching those needs a judge that reads the comment
  (§4.3.3, not built).
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
  This is a **neutral interchange format**, not a commitment to a provider. mlx-lm's
  `ChatDataset` (and, on the CUDA fallback, `trl`'s `SFTTrainer`) consumes it directly and applies the base model's chat template at
  training time, so the same file trains Mistral, Llama or Qwen without regeneration.
  The prompt *content* is built by `kibitz/prompt.py`; the chat *markup* (§7.1) belongs
  to whichever base model §6 selects. Keeping those two separate is what stops the
  dataset from being welded to one model family.
- **Split by study** (not move, not game) into `train.jsonl`, `val.jsonl`, `test.jsonl` to avoid data leakage

---

### 4.4 Dataset Format

Store training data as **JSONL** (one JSON object per line). Split into:

- `train.jsonl` — 80%
- `val.jsonl` — 10%
- `test.jsonl` — 10%

Split **by Lichess study**, not by move or game. Consecutive moves of one game are
highly correlated, and so are the chapters of one study: same author, same theme, often
whole passages of repeated text. A game-level split puts near-duplicates of test targets
in train. Because studies vary in size, the realised split is about 83 / 10 / 7.

**Studies that share content split together.** The same game is often published in
more than one study — by its author, or copied by someone else — so a plain study split
still put 6% of test targets verbatim in train. `build_dataset.py` therefore joins any
studies that share an identical (position, comment) pair into one group and splits by
group. What remains in common between test and train is 18 pairs of stock phrases ("Forking
the Bishop and pawn.") at different positions. Groups can merge when new studies are
added, which reshuffles the studies involved.

The split is still within annotators: 31 of the 38 sources survive filtering, and the
same people write train and test (24 of them appear in test). A test
score therefore says how well the model writes *like these annotators*, not how it
generalises to a new one. Splitting by annotator would answer that, but with 38 of them
and five supplying 67% of the data, the test set would be two or three people.

---

## 5. Feature Engineering

The parser captures what is *written* in the PGN. `features.py` computes what must be *derived* — things that require chess logic or an engine. It reads from `data/processed/` and writes enriched records to `data/enriched/`. A second stage, `build_dataset.py`, then reads those enriched records and applies filtering and shaping to produce the final training pairs in `data/datasets/`.

Raw move notation alone is insufficient. The model needs to understand *why* a move is interesting. These features provide that signal.

### 5.1 Position Evaluation

Use [Stockfish](https://stockfishchess.org/) via python-chess's UCI interface.

```python
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci("/path/to/stockfish")
info = engine.analyse(board, chess.engine.Limit(depth=14))   # see Appendix B
score = info["score"].white()                                # White's point of view
engine.quit()
```

**Derived features:**
- `eval_before`, `eval_after` — White-relative scores, stored in pawns
- `eval_delta` — the swing; large positive = good move, large negative = blunder
- `mate_in` — signed mate distance from White's point of view, else `None`
- `is_mate_threat` — boolean, is the forced mate *in the mover's favour*?

**Mate scores are not centipawns.** They are clamped to ±20.00 pawns before any
arithmetic, so an eval delta stays a meaningful number. About 11% of annotated moves
have a ±20 eval and so a saturated `eval_delta`: half are mates delivered on the move
(the SAN's `#` says so), half are positions with a forced mate pending, where `mate_in`
carries the real information.

**A delivered checkmate scores as mate, not as 0.** The engine has nothing to analyse
in a finished game, so `analyse()` returns 0 — correct for stalemate, but it made every
mating move a 20-pawn drop: all 4,436 annotated mates were labelled `blunder` (in the
training pairs, 19% of all blunder labels) until `enrich_move` started scoring a
checkmated position as ±20 for the side that mated. Any mating move also counts as
`is_best`, since the engine's choice is often a different mate. The move's SAN carries
the `#`; `mate_in` stays None, since there is no mate left to count.

**Features are a function of the position alone.** `analyse()` clears the engine's hash
(`ucinewgame`) before every search. A depth-limited search reuses whatever the hash
holds, so it used to depend on which positions the worker had searched before — and
the pool hands games to workers in a different order on every run. Re-analysing 300
positions in two orders gave the same eval for 37 of them; rerunning enrichment on 400
training moves reproduced the stored classification for 76% and the best move for 77%.
With the hash cleared, all 300 match, at about 10% more time per position. The hash
size is fixed at 64 MB for the same reason: a fresh 16 MB table disagrees on 7% of
positions. And the in-process cache keys a position by its halfmove clock too, since
Stockfish scales its eval by it: without the clock, the first game to reach a position
decided the features for every other, which exposed 5% of annotated moves.

**Performance note:** Engine analysis is slow. Pre-compute and cache all evals during the data pipeline stage, not at inference time. At inference, features must be computed at the training depth (14): labels, `is_best` and `Engine preferred` all shift with depth, and the model has only seen depth-14 values. A shallower live search is an option only after measuring how often its labels agree with depth 14.

---

### 5.2 Move Classification

Classify each move into one of these categories based on eval delta thresholds:

| Classification | Criteria |
|---|---|
| `best` | Matches engine's top choice — checked first, so depth noise between the two searches can never make it an error |
| `good` | Small positive or neutral eval delta |
| `inaccuracy` | Eval drops by ~0.3–0.5 pawns |
| `mistake` | Eval drops by ~0.5–1.5 pawns |
| `blunder` | Eval drops by >1.5 pawns |

These can be derived from Stockfish output. NAGs from the PGN source are human verdicts written in hindsight, so they are not used as labels or shown to the model (Decision Log row 19).

**Known gap: the thresholds ignore how decided the game already is.** They are fixed
pawn drops, so a slide from +20 to +8 is a "blunder" though the game is still won. 11%
of `mistake` / `blunder` labels sit in positions where both evals are 4+ pawns in the
same side's favour, where annotators write "still winning". Lichess avoids this by
thresholding on win probability rather than pawns; switching is a relabel (`--relabel`,
no engine) but changes the labels the golden fixture locks, so it is a deliberate
follow-up, not a silent fix.

**Classification is one axis: quality.** Three labels from the original plan are deliberately not part of it.

- `brilliant` — a chess.com construct with no counterpart in Lichess studies. It means "hard to spot", a property of the player rather than the position, which an eval delta cannot measure.
- `sacrifice` and `forcing` — both real and objective, but statements about *material* and *forcedness*, not about quality. They live in `is_sacrifice` / `material_offered` and `is_forcing`. 74% of detected sacrifices are also the engine's top move; folding them into the classification would discard that.

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
    if total_material <= 20:
        return "endgame"
    if board.fullmove_number <= 10 and not from_position:
        return "opening"
    return "middlegame"
```

Study chapters set up from a FEN restart at move 1 — endgames and middlegame puzzles
alike — so the move number alone labelled about 16% of training pairs `opening` that
were nothing of the kind. Material is checked first, and a game the parser marks
`from_position` (it has a `FEN` header) is never in the opening. A live game always
starts from the standard position, so this costs it nothing.

You can refine this with opening book detection (see ECO codes in PGN headers).

---

### 5.4 Tactical Motif Detection

These are harder but valuable. Consider detecting:

- **Fork** — one piece attacks two or more enemy pieces simultaneously
- **Pin** — a piece is pinned to its king (absolute pins only; a knight pinned to the
  queen, as in the QGD's Bg5, is not detected)
- **Skewer** — like a pin, but the more valuable piece is in front
- **Discovery** — moving a piece reveals an attack from another piece
- **Promotion** — the move promotes; **Promotion threat** — the move puts a pawn on the 7th (a pawn already waiting there does not re-trigger it on every later move)

**Not implemented: Zwischenzug.** It is defined by what the opponent was *expected* to
play, so detecting it needs a search over the expected sequence rather than board
geometry. Every other motif here is a static property of the resulting position.

Motifs are a **list**, not a category — a move can fork and threaten promotion at once.

**Known gaps in the motif detectors** (round-5 review; not yet fixed):
- *Skewer* fires when a slider attacks a more valuable piece with *anything* behind it:
  a pawn behind counts (261 of 679), and so does a front piece that can simply take the
  slider (281). Needs: the rear piece must be worth winning, and the slider must survive.
- *Fork* requires every target to be worth at least the attacker, so a queen forking
  king and a loose rook is never a fork. Needs: also count targets the attacker can win.
- Relative pins, as above.

Comments mention "pin" in 2,607 pairs and "fork" in many more than are flagged, so the
motif line under-reports what annotators see. Under-reporting is the safer failure — the
model is not told about tactics that are absent — but it is a ceiling on what the motif
line can teach.

python-chess gives you full board access to implement these as geometric checks. Start simple — fork and pin detection are the most tractable.

---

### 5.5 Threats and Loose Pieces

The first fine-tune (§7.4) invented threats: *"White threatens …Rxc3 and …Bxc2"* about
positions where neither move exists. A 7B model cannot reliably read a FEN string, so
anything the prompt does not state, it will make up. Three board facts are therefore
computed and given to it:

- `captured` — what the move took
- `threatens` — opponent pieces the mover could win with a capture sequence next
- `left_hanging` — the mover's own pieces the opponent could now win

All three use the static exchange evaluator in `features.py`, which ignores pins, so
they are approximate. Four rules keep them from saying what is false or obvious:
- the first capture must be legal — after a check the opponent usually cannot take
  anything, and listing a loose piece there taught mating attacks as blunders;
- a gain under one pawn is not a threat (B=330 vs N=320 made every bishop-for-knight
  trade look like one);
- `threatens` lists only targets this move creates, not ones the mover could already
  win — the same rule as `promotion_threat`;
- a capturing piece that can merely be taken back is an even trade, not hanging.

Promotions are valued as the pawn the mover staked, not the new piece, and a finished
game has no threats. "Threatens to win" appears in 28% of training prompts and "Left
hanging" in 20%.

The evaluator must play each capture as the capturing side. An early version did not:
after 3.Bb5 it reported the defended c6 knight as a White "threat", because the bishop
changed colour on capture and no recapture was ever considered. "Threatens to win" was
in 82% of prompts until that was fixed, and the regression test is the Bb5 position.

The engine's preferred move is also given in SAN (`Qc2`), the notation annotators write,
rather than UCI (`d2c2`), which the model has never seen in commentary.

**The engine's line and the game so far.** When the move was not the engine's choice,
the prompt also gives the first four plies of its principal variation (`Engine line:
14.Qc2 Nf6 15.Rd1 Be7`), and every prompt gives the six plies before the move
(`Previous moves:`). Annotators mostly write about the alternative ("Better was
...hxg5") and the story of the game, and without these the model could only guess at
either. `analyse()` returns the whole line, so adding them meant one more Stockfish
pass over the corpus.

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

**Chosen:** `mlx-community/Mistral-7B-Instruct-v0.3-4bit` (`configs/model.yaml`), a
4-bit MLX conversion of v0.3. `Llama-3.2-3B-Instruct-4bit` is used for fast pipeline
runs; the only trained adapters so far are on it (§7.4).

**Hardware:** training runs locally on an Apple M1 Max with 32 GB unified memory
(Decision Log row 11). On a CUDA machine, a 7B 4-bit LoRA needs about 16 GB VRAM
minimum, 24 GB to be comfortable.

---

## 7. Fine-Tuning

### 7.1 Prompt Template Design

The prompt must be identical between training and inference, so both are built by one
function, `kibitz/prompt.py`, and tests assert the parity. It writes plain
`{"role", "content"}` messages; the base model's chat markup is applied by its tokenizer
at training and inference time (Decision Log row 10), never baked into the data.

A real user turn from the test split:

```
Phase: opening
Move 8 (white): h4
Classification: best
Evaluation: +1.31 -> +1.21 (delta -0.10 for white)
Left hanging: pawn on d4
Position after the move (FEN): rnbq1rk1/ppp2ppp/4pb2/8/3P3P/3B1N2/PPP2PP1/R1BQK2R b KQ - 0 8
Length: full

Write 1-3 sentences of commentary for this move.
```

Lines whose field is missing or unknown (`?` players, `?` opening) are left out rather
than rendered, so the model never learns to condition on a literal `?`. The other
optional lines are `Game:`, `Opening:`, `Previous moves:` (the six plies before),
`Engine preferred:` and `Engine line:` (in SAN, only when the move was not best), `Captures:`, `Threatens to win:`, `Left hanging:`, `Forced mate: White
mates in N`, `Tactical motifs:` and `Notes:`
(forcing, sacrifice).

The annotator's own `!` / `?` marks (NAGs) are not in the prompt. They were, in 22% of
training prompts, until it was noticed they are hindsight from the same person who wrote
the target, and that a live game or an unannotated PGN never has them.

There is no system turn. `prompt.SYSTEM_PROMPT` is defined but never sent, in training
or at inference — harmless for the fine-tune, which learns the task from the pairs, but
it means the `--baseline` model is given no persona or length guidance either.

**`Length:` is how "comment on every move" stays readable** (§8). In training it is read
off the target: `brief` if the annotator wrote 12 words or fewer (43% of the corpus),
else `full`. At inference `length_for_move` sets it from the features. The instruction
says 1–3 sentences, but 10% of targets are longer; at `max_tokens=160`, about 1% of
generations would be cut mid-sentence.

### 7.2 Training Configuration

Hyperparameters live in `configs/model.yaml` (base model, adapter shape) and
`configs/training.yaml` (the training run). `scripts/train.py` merges them over
mlx-lm's defaults; command-line flags override both, so the config files stay an
honest record of what a real run used.

```bash
python3 scripts/train.py --measure     # token lengths for this corpus
python3 scripts/train.py               # train with the configs as written
python3 scripts/train.py --iters 200 --warmup 5 --model mlx-community/Llama-3.2-3B-Instruct-4bit   # short runs need a short warmup
```

**`max_seq_length: 512` is measured, not guessed.** The prompt is templated, so
token counts are tight — with the Mistral tokenizer, prompts have a median of 221
tokens and a p95 of 273, and full pairs a median of 249 and a p95 of 322. 512
truncates 0.19% of the training split. (Before `Previous moves:` and `Engine line:`
it was 208 and 278.) `--measure` reprints the distribution for
any tokenizer. Re-measure whenever the prompt gains a line.

**Throughput on the M1 Max** is about 0.1–0.13 iterations/s for the 7B (roughly 100
tokens/s), so one epoch takes about 20 hours. `grad_checkpoint` is off: measured, it
is ~25% faster and peaks at 18.7 GB instead of 7.2 GB, which fits in 32 GB. Other GPU
users (browser video) slow the run noticeably.

**`mask_prompt: true` matters.** Loss is computed on the assistant turn only.
Without it most of the gradient goes into learning to reproduce the feature block
the model is handed, which it does not need to predict.

**`iters` counts micro-batches, not optimizer updates.** mlx-lm applies an update
every `grad_accumulation_steps` iterations and steps the LR schedule per update. One
epoch over 35.4k pairs at batch 4 is therefore ~8,860 iters but only ~2,215 updates, and
the cosine horizon must be the latter — `train.py` derives it as `iters /
grad_accumulation_steps`. The config previously said 2,400 iters for "one epoch", which
was 0.27 of one, with the LR still at 90% of its peak when training stopped.

**A note on `scale`.** MLX applies `scale * BA` directly; it is *not* peft's
`alpha / rank`. The `alpha: 32, r: 16` of a peft config corresponds to `scale:
2.0`, whereas mlx-lm's own convention — and what its defaults are tuned for — is
`scale: 20.0`. Copying an alpha across without converting is a silent way to
train at the wrong effective learning rate. The learning rate has the same trap: the
first 7B run used 1e-4, a peft-style rate, at `scale: 20`. The loss bottomed out at ~2.8
within 25 updates, climbed as the warmup raised the rate, blew up at ~120 updates and
was NaN for the remaining ten hours. The config now uses mlx-lm's 1e-5, and `train.py`
stops when a reported loss is not finite.

---

### 7.3 LoRA / PEFT

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) fine-tunes only a small number of additional parameters (~1–2% of the model) rather than the full weights. This makes fine-tuning feasible on consumer hardware.

**Stack in use — MLX** (see Decision Log row 11; the peft stack below is CUDA-only
and cannot run on an Apple M1):

- [`mlx-lm`](https://github.com/ml-explore/mlx-lm) — LoRA training, loading and generation
- pre-quantised 4-bit base models from the `mlx-community` org on Hugging Face

`mlx-lm`'s `ChatDataset` reads `{"messages": [...]}` directly and applies the base
model's chat template itself, so the dataset built in §4 is consumed unmodified —
which is the whole point of keeping it a neutral interchange format. It expects
`train/valid/test.jsonl`; §4.4 names the middle split `val.jsonl`, so
`scripts/train.py` symlinks rather than duplicating the data.

**On a rented NVIDIA GPU:** `scripts/train_cuda.py` (`peft` + `transformers` + `trl`).
An epoch takes about an hour there, against 11–20 hours here. It reads the same two
config files and translates them, so the two runs do the same math:

| `configs/` (MLX terms) | peft / trl |
|---|---|
| LoRA on every linear layer of the top `num_layers` blocks | `target_modules` q k v o gate up down, `layers_to_transform` top blocks |
| `scale: 20`, `rank: 16` | `lora_alpha = scale × rank = 320`, so the learning rate carries over unchanged |
| `iters` (micro-batches) | `max_steps = iters / grad_accumulation_steps` (updates) |
| `mask_prompt: true` | prompt/completion pairs with `completion_only_loss`; trains the same tokens (checked) |

Two differences remain:
- **Full-precision base.** It trains the bf16 original, not a 4-bit copy, so there is no
  `bitsandbytes`.
- **Fixed validation set.** It measures validation loss on the same 500 pairs each time,
  where mlx-lm draws a fresh random 100.

`scripts/convert_adapter.py` turns the peft adapter into mlx-lm's format: the same
matrices transposed, about 80 MB. The adapter then loads onto the 4-bit MLX base like
one trained here. `tests/test_convert_adapter.py` checks that a converted adapter gives
the same logits in MLX as in peft. That matters because mlx-lm loads adapter weights
non-strictly: a misnamed key would be ignored without a word.

**What `scripts/train.py` does:**
1. Merge `configs/model.yaml` and `configs/training.yaml` over mlx-lm's defaults,
   then apply command-line overrides
2. Symlink `train/val/test.jsonl` to the `train/valid/test.jsonl` names mlx-lm expects
3. Hand the merged config to `mlx_lm.lora`, which loads the pre-quantised base model,
   attaches LoRA to the top `num_layers` layers and trains with the prompt masked
4. Save only the adapter weights to `adapter_path`

At inference `kibitz/model.py` loads the base model with the adapter attached; nothing
is merged into the base weights.

---

### 7.4 Evaluation Metrics

The one failure that matters most for a commentator is saying something false about the
position. Overlap metrics cannot see it, so evaluation has two halves.

**Board check (`kibitz/grounding.py`).** Every comment is scanned for the two kinds of
claim that can be checked mechanically, and the share of comments with at least one
failure is reported:

| Metric | Fails when |
|---|---|
| `false placement %` | "the knight on e5" / "the e5 knight" — no such piece there before or after the move |
| `unplayable move %` | a named piece move (`Qxh7`) is legal neither before the move, after it, nor as the mover's follow-up |

Annotators quote longer lines, which also count as unplayable, so the human references
go through the same check and the model is read against that baseline, not against
zero. Claims the check cannot parse — "wins the exchange", "the pin is decisive" — pass
unexamined, and it ignores colour ("White's knight on e5" passes if a *black* knight is
there), so a clean score is necessary, not sufficient.

**Overlap metrics**, for tracking one model across checkpoints:

| Metric | Tool | Notes |
|---|---|---|
| `eval_loss` | Training loop | Primary signal during training |
| `BLEU` | `sacrebleu` | n-gram overlap with the single reference |
| `ROUGE-L` | `rouge-score` | Longest common subsequence |
| `distinct-2` | built in | Collapses toward 0 when a model repeats itself |
| words (pred / ref) | built in | Length gap against the annotator |

BERTScore and perplexity were in the original plan and are not implemented.

```bash
python3 scripts/evaluate.py --limit 200 --baseline --out runs/eval.jsonl
```

`--baseline` scores the un-tuned base model in the same run: **if the fine-tune cannot
beat the base model on the same prompts, the training bought nothing.** `--out` keeps
every generation with its board-check result, for reading.

Three settings keep the comparison honest:
- **Greedy decoding** (`--temp 0`, the default), so a rerun gives the same numbers. At
  n=200, differences of a BLEU point are noise either way.
- **The reference's length setting.** Each move is generated with `Length:` taken from
  the annotator's comment, not from `length_for_move`; the two agree on only 47% of
  pairs, and mixing them confounds the length and overlap columns.
- **A missing adapter is an error.** Otherwise a mistyped `--adapter` scores the base
  model under the fine-tune's name.

**First result — `adapters/llama3b-300`** (Llama-3.2-3B, 300 iterations, the prompt
before §5.5 and the old game-level split; 200 test moves):

| | fine-tune | base model | human |
|---|---:|---:|---:|
| BLEU | 0.23 | 0.25 | |
| ROUGE-L | 10.3 | 9.1 | |
| distinct-2 | 0.68 | 0.43 | |
| words | 10.8 | 58.5 | 21.7 |
| false placement % | 5.0 | 14.0 | 0.0 |
| unplayable move % | 11.5 | 0.0 | 6.5 |

The fine-tune learned the annotators' register — short and varied — and none of their
accuracy: *"White's Bishop is pinned to a pawn."*, *"White threatens …Rxc3 and
…Bxc2."* The base model rarely names a move, so it scores 0 on the second check by
saying nothing checkable. Neither number is a verdict on a 7B run: because of the
`iters` miscount (§7.2), this adapter saw 75 optimizer updates — 1,200 examples, 3% of an
epoch — and it was sampled at temperature 0.7. It is the bar the next run has to clear.

**7B result — `adapters/`** (Mistral-7B, one epoch at LR 1e-5, current prompt and study
split, trained *before* the grounding filters in §4.3.2; the same 200 test moves,
greedy). Checkpoints from across the epoch show whether
it was still improving:

| | iter 2,400 | 4,800 | 7,200 | final (9,728) | base model | human |
|---|---:|---:|---:|---:|---:|---:|
| BLEU | 0.97 | 0.65 | 0.57 | 0.71 | 0.35 | |
| ROUGE-L | 14.4 | 14.8 | 14.2 | 14.2 | 10.7 | |
| distinct-2 | 0.32 | 0.35 | 0.38 | 0.33 | 0.37 | |
| words | 11.7 | 13.7 | 13.1 | 14.5 | 68.9 | 23.1 |
| false placement % | 6.5 | 2.0 | 1.5 | 4.5 | 38.0 | 2.5 |
| unplayable move % | 13.5 | 10.5 | 7.0 | 9.5 | 16.0 | 5.5 |

- **Against the base model**, the fine-tune is shorter, closer to the annotators, and far
  more accurate. False placements fall from 38% to 4.5%, and the base model now names
  moves, so its 16% unplayable rate is a real comparison. On the 3B adapter it scored 0
  by naming none.
- **Across the epoch**, the board check improves up to the halfway point and then
  flattens. At n=200 one point is two comments, so 4,800, 7,200 and the final are
  within noise of each other. Nothing here argues for a second epoch.
- **Still above the human rate on both checks.** The failures are confident claims with
  nothing in the prompt behind them: *"White has a 50% winrate here"*, *"Black has a
  2-to-1 advantage in material"*, *"This is the Barnes Opening … the most popular one in
  the 1.e4 category"*. Opening-explorer phrasing ("most popular", "winrate", a
  percentage) appears in 7% of generations against 1.5% of references and 2% of
  training targets. It is safe-sounding filler, over-produced under greedy decoding.
- **Too short and too generic.** The fine-tune writes 14.5 words against the human 23.
  The most common outputs are stock lines such as *"White has a winning position."*

**After the grounding filters — `adapters/mistral7b-cuda-v2`** (the same recipe on the
filtered data of §4.3.2, trained in bf16 on a Lambda A10 and converted, Appendix C). On
the 200 moves of the filtered test split, against the adapter above:

| | before filters | after | human |
|---|---:|---:|---:|
| ungrounded talk % | 8.0 | **0.5** | 0.0 |
| false placement % | 3.5 | 2.5 | 0.5 |
| unplayable move % | 9.0 | 8.5 | 2.5 |
| BLEU | 0.51 | 1.21 | |
| ROUGE-L | 13.7 | 14.5 | |
| words | 13.1 | 14.5 | 18.8 |

All 16 ungrounded comments ("What is Black's best move?", "the most common move in the
LMDB") are gone and one new one appeared. The board checks moved within noise (15
unplayable moves fixed, 14 new). Two things changed at once, the data and the
framework, but exactly the filtered patterns vanished.

**Greedy decoding is why it sounds like a template.** Under greedy decoding 36% of
comments say "threat" (humans: 6%; training targets: 8%) and half begin "White is" or
"Black is". With 38 annotators saying different things, the single most likely
continuation is what they have in common. Sampling restores the voice but invents more,
so the CLI samples up to four and keeps the first with no detectable fault
(`KibitzCommentator(samples=4)`):

| same adapter | greedy | temp 0.8 | temp 0.8, best of 4 | human |
|---|---:|---:|---:|---:|
| distinct-2 | 0.37 | 0.77 | 0.76 | 0.86 |
| says "threat" | 72 | 16 | 4 | 12 |
| first person | 8 | 17 | 29 | 37 |
| words | 14.5 | 16.8 | 17.5 | 18.8 |
| false placement % | 2.5 | 6.0 | 1.0* | 0.5 |
| unplayable move % | 8.5 | 13.5 | 0.0* | 2.5 |
| seconds per comment | 1.0 | 1.0 | 2.5 | |

\* Selected on these checks, so near zero by construction: they show that four samples
almost always contain a clean one, not that the comments are right. Errors the checks
cannot see remain (*"The Scotch Gambit, a common response to the Caro-Kann"*). Fixing
those needs better information in the prompt, not decoding (§5: engine line, previous
moves).

**Human evaluation** is still the gold standard and is not yet scheduled. Proposed:
100 held-out moves, two raters, each comment marked for factual accuracy (does it
describe what the move does?), insight (beyond the obvious?) and fluency.

---

## 8. Inference Pipeline

### Two modes, one pipeline

Kibitz must work in both directions:

- **Live** — following a game in progress, seeing only the moves played so far.
- **Retrospective** — commenting on a complete PGN.

**These are the same pipeline, not two systems.** Every feature is computed from the
position *before* the move and the position the move creates; nothing reads ahead. The
difference is only how moves are fed in: one at a time as they arrive, or all at once.

That property is load-bearing, so it is partly enforced rather than assumed:
`tests/test_prompt.py` asserts that no result or termination header can reach a prompt,
and that the only position a prompt contains is the one the move created. Move records
carry no field describing a later position, so the features cannot read ahead. The
exceptions are PGN headers. `Opening:` (73% of training prompts) is derived from the
whole game (item 3 below). `Game:` carries the Event header, which for 9% of pairs is a
study chapter title written by the annotator — "… – BLUNDER", "Defend by pinning",
"Sacrifice like a PRO!" — the same hindsight that got NAGs removed (row 19). Neither is
guarded by a test yet; dropping the Event for study chapters is the likely fix.

Three things genuinely *do* differ, and they are the real work:

1. **Engine depth.** Retrospective analysis is offline and can afford depth 14+. Live
   commentary is on a clock. But the features the model was trained on came from depth
   14, so a shallower live search changes the inputs, not just the latency (§5.1).
2. **How much to say.** Kibitz comments on every move (Decision Log row 13), but a
   wall of three-sentence comments on routine developing moves is unreadable. The
   `Length:` line (§7.1) handles this: `length_for_move` asks for `full` commentary on
   inaccuracies, mistakes, blunders, tactical motifs, sacrifices and forced mates, and
   `brief` on everything else. The same rule serves live and retrospective modes.
3. **Header availability.** `Opening` / `ECO` come from PGN headers, which a finished
   game has and a live one does not. Worse, they are derived from the whole game, so
   quoting them at move 2 is mild lookahead. Live mode should derive the opening from
   the moves played so far, or omit the line until it is determined.

**A caveat about the targets, not the features.** Study annotators wrote with the whole
game in front of them, and about 0.8% of comments say so outright — "as we will see
soon", "the final blunder of the match". That is a small share, and filterable, but the
selection bias underneath it is not: annotators comment on the losing move *because they
know it lost*. A model trained on that corpus will tend to speak with more certainty than
a live position supports. Worth measuring on the held-out set before trusting live mode.

**Known gap: no game history.** In both modes, each prompt describes one move in
isolation: no earlier moves, no earlier commentary. Commenting on every move without
memory invites the same phrase on consecutive moves and invented references to the
opponent's last move. The cheapest fix is the opponent's previous move in SAN; the
measure is distinct-n *within* a game, which the CLI's JSON output allows (the current `distinct-2` is
over a random sample of moves, so it cannot see this).

**Every move, and the data behind it.** Annotators commented on 6.7% of plies, chosen
with hindsight, so the model has never been trained on a move nobody found worth
remarking on. That gap is smaller than it sounds: 64% of the annotated moves are
themselves quiet (`best` or `good`, no motif, no sacrifice), and their comments are as
long as the notable ones (median 14 words each). What the corpus does not contain is
the routine move nobody commented on, so brief output on those moves is untested until
the human evaluation looks at it — the references for them do not exist.

At inference time, Kibitz should:

1. Accept a PGN file path or string as input
2. Parse the game with python-chess
3. For each move, extract features (Stockfish at depth 14, as in training)
4. Build the prompt, with `Length:` from `length_for_move`
5. Run the fine-tuned model to generate commentary
6. Stream or batch output to stdout / file

**Latency** has not been measured on the M1 Max, and the original GPU estimates
(0.5–2 s per move) do not transfer. Measure it with the CLI once the 7B adapter exists;
the 3B smoke adapter took about 7 s per move, but with a training run sharing the GPU.

**Batching:** If you don't need streaming output, batch all prompts for the game and run them through the model in parallel. This can reduce wall-clock time significantly.

**Output formats to support:**
- `--format text` — plain annotated game, moves with comments inline
- `--format pgn` — valid PGN with `{ ... }` comment blocks
- `--format json` — structured output for programmatic use

### Runtime: llama.cpp

Kibitz is meant to be downloaded and run on the user's own machine, and MLX runs only
on Apple Silicon. The shipped runtime is therefore [llama.cpp](https://github.com/ggml-org/llama.cpp)
through `llama-cpp-python` (Decision Log row 21): Mac, Windows and Linux, GPU or CPU.
Training is unchanged; only inference moves.

```bash
python3 scripts/convert_gguf.py adapters/cuda-v2 adapters/kibitz.gguf --base <base>.gguf
python3 scripts/evaluate.py --gguf adapters/kibitz.gguf --out runs/eval-gguf.jsonl
kibitz game.pgn --gguf adapters/kibitz.gguf
```

The model is a stock GGUF of the base (`GGUF_BASE` in `kibitz/model.py`, 4.4 GB at
Q4_K_M) plus the converted adapter (84 MB), applied at load time, so nothing is merged
and an adapter update is a small download. Three things had to be right:

- **Row order of q/k.** llama.cpp's converter permutes the rows of `q_proj` and `k_proj`
  for its rotary layout, so `lora_b` of those modules must be permuted the same way.
  Getting it wrong is nearly invisible: on the full adapter, the correlation between the
  adapter's effect on next-token logits in MLX and in llama.cpp only drops from 0.97 to
  0.96, because most of the adapter acts through the other modules. On an adapter
  restricted to q/k, it is 0.89–0.92 with the permutation and −0.18 to +0.08 without.
  `tests/test_convert_gguf.py` checks names, alpha and the permutation.
- **Tokenization.** llama.cpp's tokenizer splits the spaces around `[INST]` differently
  from the Hugging Face tokenizer the model was trained with (`<s> ␣␣ [INST] ␣ Open…`
  against `<s> ␣ [INST] Open…`). `GGUFCommentator` tokenizes with the base's own
  `tokenizer.json` and chat template and hands llama.cpp the ids;
  `tests/test_model.py` asserts the ids equal training's on all 72 golden prompts.
- **Sampling.** llama.cpp filters with top-k, top-p and min-p by default; mlx-lm does
  not. They are switched off so both draw from the same distribution.

**MLX against llama.cpp**, the same adapter (`mistral7b-cuda-v2`) on the same 200 test
moves, greedy, same day:

| | MLX | llama.cpp | human |
|---|---:|---:|---:|
| BLEU | 1.04 | 1.29 | |
| ROUGE-L | 13.9 | 14.0 | |
| distinct-2 | 0.35 | 0.39 | |
| words | 13.4 | 12.1 | 16.9 |
| false placement % | 1.5 | 4.5 | 2.5 |
| unplayable move % | 17.5 | 12.5 | 3.0 |
| ungrounded talk % | 0.5 | 1.0 | 0.0 |
| comments per second (M1 Max) | 0.7 | 1.0 | |

The two 4-bit quantizations differ, so greedy outputs diverge after a few tokens and
only 29 of 200 comments are identical. The board checks move in opposite directions by
3 and 5 points (6 and 10 comments), within noise at n=200. Nothing here says the port
lost quality, and it is 40% faster.

An earlier greedy run of the same adapter on this split (`runs/eval-v2prompt-greedy`,
30 Oct, 10.5% unplayable) does not reproduce: only 28 of its 200 comments match a rerun,
with the environment, adapter and enriched data ruled out. The prompt code most likely
changed between that run and the commit. Compare against reruns, not that log.

---

## 9. CLI Design

`kibitz/cli.py`. Each move goes through the same steps that built the training data:
`parser.game_record` (shared with `parse_pgn`), `features.enrich_move` at depth 14, and
`prompt.build_prompt` with `Length:` from `length_for_move`.

```
kibitz [OPTIONS] <pgn_file | ->

Options:
  --format, --output-format [text|pgn|json]   Output format (default: text)
  --stockfish PATH        Stockfish binary (default: $STOCKFISH_PATH, else stockfish
                          on PATH, else Stockfish 19, downloaded once)
  --depth INT             Engine depth (default: 14, the training depth)
  --gguf PATH             GGUF adapter for llama.cpp (default: the released adapter,
                          downloaded once)
  --model PATH            MLX adapter directory, run through mlx-lm instead (Apple silicon)
  --moves RANGE           Only comment on these move numbers, e.g. "15-30" or "17"
  --temp FLOAT            Sampling temperature (default: 0.8)
  --samples INT           Sample up to N per move; keep the first that passes the
                          board and grounding checks (default: 4)
  --verbose               Show the features each comment was conditioned on

Examples:
  kibitz game.pgn
  kibitz game.pgn --format pgn > annotated.pgn
  kibitz game.pgn --moves 15-30 --verbose
  cat game.pgn | kibitz -
```

From a checkout, `python3 -m kibitz.cli` is the same command.

### Installing

```bash
# Mac (Apple silicon)
pip install "kibitz @ git+https://github.com/Addaian/kibitz" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
# Windows and Linux, CPU
pip install "kibitz @ git+https://github.com/Addaian/kibitz" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

The extra index has prebuilt `llama-cpp-python` wheels; without it pip compiles llama.cpp,
which needs a C++ compiler and CMake (about a minute on the M1 Max). The install pulls in
neither PyTorch nor MLX: `pyproject.toml` lists only what `kibitz game.pgn` needs, and
`requirements.txt` the rest, for a checkout. `pip install "kibitz[mlx]"` adds `--model`.

The first run downloads, once, into the usual caches:
- **the model**: the base GGUF (4.4 GB) and its tokenizer files from Hugging Face, and the
  Kibitz adapter (84 MB) from `huggingface.co/Addaian/kibitz`. That repo is not published
  until every annotator whose comments trained the model has agreed, so for now pass
  `--gguf PATH`; without it the CLI says so and stops.
- **Stockfish**, only if there is none on `$STOCKFISH_PATH` or PATH: the official
  Stockfish 19 build for the OS and CPU (~80 MB), checked against the release's SHA-256,
  into `$KIBITZ_HOME` (default `~/.cache/kibitz`). Stockfish is GPL-3, so it is fetched
  rather than bundled. Training features came from Stockfish 18, but 18 has no build for
  Linux on ARM and no build that picks the CPU's instruction set at run time; on 400
  enriched moves, Stockfish 19 agrees with the stored classification as often as
  Stockfish 18 itself does on a rerun (75% and 76%, Decision Log row 22).

How it differs from the original sketch:
- **argparse, not click or typer.** It matches the other scripts and adds no dependency.
- **No `--stream` flag.** Text output prints each comment as soon as it is generated,
  so streaming is the default. PGN and JSON need the whole game and print at the end.
- **llama.cpp by default.** MLX remains for checking a training run on the Mac
  (`--model`), Decision Log row 23.

The output formats:
- **PGN** replaces the comments on the moves in range and keeps the headers and any
  comments outside it.
- **JSON** is a list with one object per game: `metadata` plus `moves`, with features
  added under `--verbose`.

A PGN with several games gets every game annotated. An illegal move ends that game's
mainline with a warning, as python-chess reads it.

---

## 10. Testing Strategy

| Test type | What to cover |
|---|---|
| Unit | PGN parser handles edge cases (castling, en passant, promotion) |
| Unit | Feature extractor returns correct phase/classification |
| Unit | Prompt builder produces correctly-formatted strings |
| Unit | Eval delta computation is correct (sign, scale) |
| Unit | Dataset cleaning, non-prose filter, study-level split |
| Unit | Board check catches false placements and unplayable moves |
| Causality | A move's prompt is identical built from the full or the truncated game (§8) |
| Docs | Every Appendix B command resolves (`test_docs.py`) |
| Regression | `tests/fixtures/golden.jsonl` — 72 real moves whose derived features are replayed without Stockfish |
| CLI | Move ranges, every output format, PGN round-trip through python-chess (`test_cli.py`, with a stand-in model) |
| Integration | One known game through real Stockfish (skipped without it). The model is not covered; it needs trained weights. |

Use `pytest`. Fixtures live in `tests/fixtures/`.

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

## 12. Libraries

What the code imports today (`requirements.txt`):

```
chess                 # python-chess: PGN parsing, board logic, Stockfish UCI
requests, tqdm        # downloading studies, progress bars
pyyaml                # configs/
mlx-lm                # LoRA training and generation on Apple silicon
sacrebleu             # BLEU
rouge-score           # ROUGE-L
pytest
```

Install Stockfish separately as a system binary — it is not a Python package.

Training on a rented GPU uses `requirements-cuda.txt`: `torch`, `transformers`, `peft`,
`trl`, `accelerate` and `datasets`, pinned to the versions it was tested with (Appendix C).
The CLI ended up on `argparse`, so `typer` is not used. The web UI stretch goal would add
`fastapi` and `uvicorn`.

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
| 2 | `sacrifice` classification | `is_sacrifice` + `material_offered` | Orthogonal to quality. 74% of detected sacrifices are also the engine's top move, and the label was overwriting `best`. |
| 3 | `forcing` classification | `is_forcing` | Same reason: a move can be forcing *and* a blunder. |
| 4 | `side` | `color` | Naming only. |
| 5 | `depth_of_mate` | `mate_in` | Signed and White-relative, so it composes with the eval fields. |
| 6 | `opening_name` per move | `metadata.opening` per game | It is a property of the game, not the move. |
| 7 | Zwischenzug detection | not implemented | Defined by the opponent's *expected* reply, so it needs a search rather than board geometry. |
| 8 | "decide whether to include variations" | stripped | Sidelines comment on moves that were never played, so they have no real position to pair with. |
| 9 | balance across classifications | opt-in, train split only | A metric measured on a rebalanced val/test set does not describe behaviour on real games. |
| 10 | "Claude Messages API pairs" | Messages as neutral interchange | Decouples the dataset from any one model family; the chat template is applied at training time. |
| 11 | `peft` + `bitsandbytes` 4-bit LoRA (§7.2, §7.3) | MLX / `mlx-lm` | The dev machine is an Apple M1 Max. `bitsandbytes` is CUDA-only, so the documented stack cannot run locally at all. MLX is native to the chip and 32 GB unified memory fits a 7B LoRA. A rented-GPU path was added later for speed, not capability (Appendix C). |
| 12 | (unstated) | Live and retrospective are one pipeline | Every feature is causal already; the difference is only how moves are fed in. Enforced by tests rather than convention — see §8. |
| 13 | Commentary on every move (§1 example) | Kept, with a trained `Length: brief/full` control | Only 6.7% of plies are annotated, but 64% of those are quiet moves, so quiet-move commentary is in the data. What was missing was a way to keep it short. |
| 14 | Split by game (§4.4) | Split by Lichess study | Chapters of one study share an author, a theme and repeated passages, and the same game is republished across studies; a game split put 6% of test targets verbatim in train. Studies sharing a (position, comment) pair are grouped first. |
| 15 | (unstated) | Drop non-English and non-prose targets | 7% of targets were Turkish or Spanish prose, pasted engine logs, ECO labels or game citations. Links inside real prose are stripped instead. |
| 16 | `best_move` in the prompt | Rendered in SAN | Annotators write `Qc2`, never `d2c2`; the model cannot connect the two. |
| 17 | BLEU / ROUGE / BERTScore (§7.4) | Board check added; BERTScore not implemented | Overlap metrics cannot see a false claim, which is the main failure — the first fine-tune named moves no nearby position allows in 11.5% of comments (annotators: 6.5%). |
| 18 | (unstated) | `captured`, `threatens`, `left_hanging` features | A 7B model cannot read a FEN; whatever the prompt does not state, it invents (§5.5). |
| 19 | (unstated) — NAGs were in the prompt | Removed from the prompt | The annotator's `!`/`?` is hindsight from the target's author, present in 22% of training prompts and absent from every live game. |
| 20 | "~1 epoch" at 2,400 iters | One epoch in iters (8,860 now), cosine horizon in updates | mlx-lm counts iters in micro-batches; 2,400 was 0.27 epoch and the LR never decayed (§7.2). |
| 21 | Inference through MLX (§7.3) | llama.cpp: a stock base GGUF plus the converted adapter, prompts tokenized by the base's own tokenizer | Kibitz is to be downloaded and run locally, and MLX runs only on Apple Silicon. llama.cpp runs on Mac, Windows and Linux; same quality on the test split and 40% faster on the M1 Max (§8). |
| 22 | Stockfish from Homebrew (`$STOCKFISH_PATH`) | `$STOCKFISH_PATH`, else PATH, else the official Stockfish 19 build, downloaded and checksum-verified | Users on Windows and Linux have no Homebrew path, and Stockfish's GPL-3 rules out bundling it. 19 rather than the training engine (18) because only 19 has one self-selecting build per OS, Linux ARM included. The features barely notice: rerunning depth 14 on 400 enriched moves, 18 reproduces the stored classification 75.8% of the time and 19 75.2% (best move 77.0% and 77.5%) — hash history moves them more than the version does. The eval itself shifts more (90th-percentile difference 2.1 pawns against 0.45). |
| 23 | `python3 -m kibitz.cli`, MLX (§9) | An installable `kibitz` command (`pyproject.toml`), llama.cpp by default, `--model` for MLX | A download must run on any OS without MLX or a checkout; the runtime's model settings moved from `configs/model.yaml` into `kibitz/model.py` because an installed package has no `configs/`. |
| 24 | (unstated) | Every search from a cleared hash; the analysis cache keyed with the halfmove clock | Depth-limited search reused whatever the hash held and the cache answered every clock with the first one seen, so features depended on the order the pool happened to process games: a rerun reproduced 77% of the stored classifications. Re-enriching changed 22.6% of classifications and best moves; class shares moved under a point, and the pairs, splits and targets are unchanged. Reruns with different worker counts are now byte-identical. Adapters trained before this saw the noisy labels (§5.1). |

**Fields we added that the plan never specified:** `is_best`, `best_move`,
`best_move_san` (all free from the same engine call), `material_offered`, `captured`,
`threatens`, `left_hanging`, `comment_text`, `ply`, `total_plies`, and `study` on the
game metadata.

**Classification is one axis: move quality.** `best` / `good` / `inaccuracy` / `mistake` /
`blunder`. Anything orthogonal to quality is its own field. When a new label needs a
precedence rule against an existing one, that is the signal it belongs on a different
axis — not that it needs a tie-break.

---

## Appendix B — Running the Pipeline

```bash
python3 scripts/download_data.py                      # -> data/raw/        (~55 MB)
python3 -m kibitz.parser                              # -> data/processed/  (~3 min)
python3 -m kibitz.features --depth 14 --workers 8     # -> data/enriched/   (~40 min)
python3 scripts/build_dataset.py                      # -> data/datasets/
python3 scripts/inspect_dataset.py stats               # look at what you got

# phase 2
python3 scripts/train.py --measure                    # token lengths for this corpus
python3 scripts/train.py                              # LoRA fine-tune -> adapters/
python3 scripts/evaluate.py --limit 200 --baseline    # score against held-out test
python3 -m kibitz.cli game.pgn --model adapters      # comment on a game (§9)
```

**Read the output, do not just read the loss.** The review harness puts the model
beside the annotator on the same position:

```python
from kibitz.model import KibitzCommentator
from kibitz.review import HumanAnnotation, render_move
print("\n".join(render_move(meta, move, source,
                            [HumanAnnotation(), KibitzCommentator(adapter_path="adapters")])))
```

Use held-out games for this — `scripts/inspect_dataset.py show --split test` —
or you will be admiring commentary the model memorised.

**Stockfish exits on illegal positions, and stays dead.** A kingless pawn diagram — which
studies contain as teaching material — makes it terminate with SIGSEGV, after which every
later analysis in that worker fails. `analyse()` therefore refuses any position failing
`board.is_valid()`, and restarts the engine if it dies anyway. Without both, one bad
diagram silently poisons every game a worker touches afterwards, and the run still exits 0.

**Variant games are skipped** (478 of them: Chess960, Crazyhouse, Antichess, Horde,
Racing Kings, Three-check, King of the Hill, Atomic). A standard engine's evaluation of
an Antichess position is meaningless, and its commentary describes different tactics.

**Only prose-commented moves are analysed.** A move with no comment has no training
target, so evaluating it is wasted engine time — about an 8x saving (1.24M positions
vs at most two per annotated move).

**Depth 14, not 18.** Five times cheaper, and the classification thresholds (0.3 / 0.5 /
1.5 pawns) are far coarser than the difference between the two depths.

**Changing a label does not need the engine.** `python3 -m kibitz.features --relabel`
replays classification, phase, motifs and the sacrifice test against the evals already on
disk — 10 seconds for the whole corpus instead of 40 minutes. Re-freeze the golden fixture
(`scripts/inspect_dataset.py freeze`) and rebuild the datasets afterwards.

---

## Appendix C — Training on a Rented GPU

One epoch on a Lambda H100 takes about an hour, and 1 h 50 min on an A10 (24 GB, measured),
against 11–20 hours on the M1 Max. The
GPU trains only; evaluation and the CLI stay on the Mac.

**Before you start:** accept the licence of `mistralai/Mistral-7B-Instruct-v0.3` on
Hugging Face and create a read token. Launch an instance (H100 80 GB, or A100) with your
SSH key.

```bash
# on the Mac: copy the code, configs and built datasets (~30 MB), not raw data or adapters
rsync -av --exclude .git --exclude data/raw --exclude data/processed --exclude data/enriched \
      --exclude adapters --exclude runs --exclude .venv ./ ubuntu@<ip>:kibitz/

# on the instance
cd kibitz && export HF_TOKEN=hf_...         # or save it to ~/.cache/huggingface/token
bash scripts/lambda_setup.sh                 # venv, pinned deps, GPU + data + token checks
tmux new -s train                            # survives a dropped SSH connection
.venv/bin/python scripts/train_cuda.py --micro-batch 16 --max-steps 5 --out adapters/cuda-smoke  # ~1 min
.venv/bin/python scripts/train_cuda.py --micro-batch 16    # A100 40 GB: --micro-batch 8
# A10 24 GB: --micro-batch 2, with PYTORCH_ALLOC_CONF=expandable_segments:True
# (4 runs out of memory when a batch of long pairs comes up; 2 peaks at 19 GB)

# back on the Mac: fetch the adapter (not the checkpoints' optimizer state) and convert it
rsync -av --exclude 'checkpoint-*' ubuntu@<ip>:kibitz/adapters/cuda/ adapters/cuda/
python3 scripts/convert_adapter.py adapters/cuda adapters/mistral7b-cuda
python3 scripts/evaluate.py --adapter adapters/mistral7b-cuda --out runs/eval-cuda.jsonl
```

`--micro-batch` changes only how the effective batch of 16 is split, so the math is
unchanged. Larger micro-batches just use more of the GPU per step.

**Comparing two datasets** (say before and after a filtering change): train both on the
same instance with `--data <dir> --out adapters/cuda-<name>`. Pass `--max-steps` for an
older dataset of a different size: one epoch is its train pairs / 16. Then score both
adapters on the current test split. That keeps the base precision, framework and test
set fixed, so the data is the only difference.

**Terminate the instance when the adapter is downloaded.** Lambda bills while it runs,
idle or not.
