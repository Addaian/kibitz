#!/usr/bin/env python3
"""
A terminal viewer for the enriched dataset.

The features in data/enriched/ are only useful if they actually describe the
chess on the board. This renders a move next to everything the pipeline decided
about it, so the two can be compared by eye.

    python3 scripts/inspect.py                        # a random annotated move
    python3 scripts/inspect.py -n 5 --class blunder   # five blunders
    python3 scripts/inspect.py --motif fork --phase endgame
    python3 scripts/inspect.py --source AbasovN --seed 7
    python3 scripts/inspect.py --stats                # corpus dashboard
"""

import argparse
import json
import re
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess

ENRICHED_DIR = Path(__file__).parent.parent / "data" / "enriched"

# --- ANSI ----------------------------------------------------------------- #
COLOR = True
R = BOLD = DIM = ""

def fg(n): return f"\x1b[38;5;{n}m" if COLOR else ""
def bg(n): return f"\x1b[48;5;{n}m" if COLOR else ""


def set_color(enabled):
    """Recompute every escape sequence for colour / mono output."""
    global COLOR, R, BOLD, DIM, CLASS_COLOR
    COLOR = enabled
    R = "\x1b[0m" if enabled else ""
    BOLD = "\x1b[1m" if enabled else ""
    DIM = "\x1b[2m" if enabled else ""
    CLASS_COLOR = {
        "blunder": fg(196), "mistake": fg(208), "inaccuracy": fg(220),
        "good": fg(250), "best": fg(84), "brilliant": fg(51), "sacrifice": fg(207),
    }

# Outline glyphs for White, filled for Black, so the sides stay distinguishable
# even with colour switched off.
GLYPH_WHITE = {"p": "♙", "n": "♘", "b": "♗", "r": "♖", "q": "♕", "k": "♔"}
GLYPH_BLACK = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}



# --- ASCII art primitives -------------------------------------------------- #
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def vlen(s):
    """Visible width, ignoring colour escapes."""
    return len(_ANSI.sub("", s))


def vpad(s, width):
    return s + " " * max(0, width - vlen(s))


def box(lines, width, title=None, colour=""):
    """Draw a framed panel around `lines`."""
    if title:
        bar = "-" * max(0, width - vlen(title) - 5)
        top = f"{colour}+- {BOLD}{title}{R}{colour} {bar}+{R}"
    else:
        top = f"{colour}+{'-' * (width - 2)}+{R}"
    out = [top]
    for line in lines:
        out.append(f"{colour}|{R} {vpad(line, width - 4)} {colour}|{R}")
    out.append(f"{colour}+{'-' * (width - 2)}+{R}")
    return out


BOARD_W = 37


def render_board(fen, from_sq=None, to_sq=None, flip=False, unicode_pieces=False):
    """
    An 8x8 board drawn as ASCII art. Empty dark squares carry a faint dot so the
    checkerboard stays visible; the move's origin and destination are bracketed.

        +---+---+---+---+---+---+---+---+
      8 | r |   | b |   | k | b |   | r |
    """
    board = chess.Board(fen)
    ranks = range(7, -1, -1) if not flip else range(8)
    files = range(8) if not flip else range(7, -1, -1)

    rule = "    +" + "---+" * 8
    lines = [f"{DIM}{rule}{R}"]

    for rank in ranks:
        row = f"  {DIM}{rank + 1}{R} {DIM}|{R}"
        for file in files:
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            dark = (file + rank) % 2 == 0

            if piece:
                if unicode_pieces:
                    table = GLYPH_WHITE if piece.color == chess.WHITE else GLYPH_BLACK
                    glyph = table[piece.symbol().lower()]
                else:
                    glyph = piece.symbol()
                tint = fg(255) if piece.color == chess.WHITE else fg(244)
                cell = f"{tint}{glyph}{R}"
            else:
                cell = f"{DIM}.{R}" if dark else " "

            if sq == from_sq:
                row += f"{fg(214)}({R}{cell}{fg(214)}){R}"
            elif sq == to_sq:
                row += f"{fg(84)}[{R}{cell}{fg(84)}]{R}"
            else:
                row += f" {cell} "
            row += f"{DIM}|{R}"
        lines.append(row)
        lines.append(f"{DIM}{rule}{R}")

    letters = "abcdefgh" if not flip else "hgfedcba"
    lines.append(f"    {DIM}" + "".join(f"  {c} " for c in letters) + f"{R}")
    return lines


def eval_bar(eval_white, width=30):
    """Advantage as a filled bar: all-left is White winning, all-right is Black."""
    share = max(0.0, min(1.0, eval_white / 12 + 0.5))
    filled = round(share * width)
    return f"[{fg(255)}{'#' * filled}{R}{DIM}{'.' * (width - filled)}{R}]"


def eval_verdict(ev):
    """Plain words for a centipawn number."""
    side = "white" if ev > 0 else "black"
    a = abs(ev)
    if a < 0.3:
        return "level"
    if a < 0.8:
        return f"{side} slightly better"
    if a < 1.5:
        return f"{side} better"
    if a < 3.0:
        return f"{side} clearly better"
    return f"{side} winning"


def sparkbar(n, top, width=30):
    return f"{fg(74)}{'#' * max(1, round(n / top * width))}{R}"


def wrap(text, width):
    """Greedy word wrap."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# --- move view -------------------------------------------------------------- #

def describe(move):
    """The panel beside the board: what the pipeline decided about this move."""
    cls = move["move_classification"]
    colour = CLASS_COLOR.get(cls, "")
    dots = "." if move["color"] == "white" else "..."
    played = f"{move['move_number']}{dots} {move['san']}"

    if move["is_best"]:
        best = f"{fg(84)}engine's own choice{R}"
    elif move["best_move"]:
        best = f"preferred {BOLD}{move['best_move']}{R}"
    else:
        best = f"{DIM}n/a{R}"

    flags = []
    if move["is_forcing"]:
        flags.append(f"{fg(214)}forcing{R}")
    if move["is_sacrifice"]:
        flags.append(f"{fg(207)}sacrifice{R}")
    if move["mate_in"] is not None:
        flags.append(f"{fg(196)}mate in {abs(move['mate_in'])}{R}")

    def row(label, value):
        return f"{DIM}{label:<7}{R}{value}"

    rows = [
        f"{BOLD}{played}{R}",
        f"{colour}{BOLD}{cls.upper()}{R}",
        "",
        row("eval", f"{move['eval_before']:+.2f}  ->  {move['eval_after']:+.2f}"),
        row("delta", f"{BOLD}{move['eval_delta']:+.2f}{R} {DIM}for {move['color']}{R}"),
        row("best", best),
        row("phase", move["game_phase"]),
    ]
    if move["is_sacrifice"]:
        rows.append(row("offers", f"{fg(207)}{move['material_offered']:+.1f} pawns{R}"))
    if move["tactical_motifs"]:
        rows.append(row("motifs", f"{fg(51)}{', '.join(move['tactical_motifs'])}{R}"))
    if flags:
        rows.append(row("flags", "  ".join(flags)))
    return rows


def show(meta, move, source, unicode_pieces=False):
    white = meta.get("white") or "?"
    black = meta.get("black") or "?"
    event = " ".join(x for x in (meta.get("event") or "?", (meta.get("date") or "")[:10]) if x)
    opening = meta.get("opening") or ""
    eco = meta.get("eco") or ""
    title = f"{BOLD}{white}{R} {DIM}vs{R} {BOLD}{black}{R}"
    sub = "  ".join(x for x in (opening, f"({eco})" if eco else "", f"[{source}]") if x)

    total = BOARD_W + 2 + PANEL_W
    print()
    for line in box([title, f"{DIM}{event}{R}", f"{DIM}{sub}{R}"], total, colour=DIM):
        print(line)
    print()

    uci = chess.Move.from_uci(move["uci"])
    board_lines = render_board(move["fen_after"], uci.from_square, uci.to_square,
                               unicode_pieces=unicode_pieces)

    ev = move["eval_after"]
    right = box(describe(move), PANEL_W, title="MOVE",
                colour=CLASS_COLOR.get(move["move_classification"], ""))
    right += [""]
    right += box([eval_bar(ev),
                  f"{BOLD}{ev:+.2f}{R}   {DIM}{eval_verdict(ev)}{R}"],
                 PANEL_W, title="EVALUATION", colour=DIM)

    pad = max(len(board_lines), len(right))
    board_lines += [""] * (pad - len(board_lines))
    right += [""] * (pad - len(right))
    for b, i in zip(board_lines, right):
        print(f"{vpad(b, BOARD_W)}  {i}")

    print()
    comment = " ".join(move["comment_text"].split())
    for line in box([f"{fg(229)}{l}{R}" for l in wrap(comment, total - 6)],
                    total, title="ANNOTATION", colour=DIM):
        print(line)


PANEL_W = 42


def iter_moves(sources=None):
    files = sorted(ENRICHED_DIR.glob("*.jsonl"))
    if sources:
        files = [f for f in files if any(s.lower() in f.name.lower() for s in sources)]
    for path in files:
        label = path.stem.replace("_studies", "")
        for line in path.open():
            game = json.loads(line)
            uid = f"{label}:{game['game_id']}"
            for move in game["moves"]:
                yield game["metadata"], move, label, uid


def matches(move, args):
    if args.classification and move["move_classification"] not in args.classification:
        return False
    if args.motif and not set(args.motif) & set(move["tactical_motifs"]):
        return False
    if args.phase and move["game_phase"] not in args.phase:
        return False
    if args.min_comment and len(move["comment_text"]) < args.min_comment:
        return False
    return True


def run_stats(args):
    cls, phase, motif, src = Counter(), Counter(), Counter(), Counter()
    colour = defaultdict(Counter)
    total = 0
    games_seen = 0
    last_uid = None
    deltas, lengths = [], []

    for meta, move, label, uid in iter_moves(args.source):
        if uid != last_uid:          # moves arrive grouped by game
            games_seen += 1
            last_uid = uid
        total += 1
        cls[move["move_classification"]] += 1
        phase[move["game_phase"]] += 1
        src[label] += 1
        colour[move["color"]][move["move_classification"]] += 1
        for m in move["tactical_motifs"]:
            motif[m] += 1
        deltas.append(move["eval_delta"])
        lengths.append(len(move["comment_text"]))

    if not total:
        sys.exit("no enriched data found — run: python3 -m kibitz.features")

    W = 66
    BAR = 26

    def dist(title, counter, colour=None, label_w=18):
        top = counter.most_common(1)[0][1]
        rows = []
        for k, n in counter.most_common():
            tint = (colour or {}).get(k, "")
            rows.append(f"{tint}{k:<{label_w}}{R}{n:>7,}  {sparkbar(n, top, BAR)}")
        return box(rows, W, title=title, colour=DIM)

    print()
    for line in box([f"{BOLD}kibitz enriched corpus{R}",
                     f"{DIM}{total:,} annotated moves | {games_seen:,} games | "
                     f"{len(src)} study collections{R}"], W, colour=DIM):
        print(line)
    print()

    for line in dist("MOVE CLASSIFICATION", cls, CLASS_COLOR):
        print(line)
    print()
    for line in dist("GAME PHASE", phase):
        print(line)
    print()
    for line in dist("TACTICAL MOTIFS", motif, label_w=18):
        print(line)
    print()

    deltas.sort()
    lengths.sort()
    q = lambda xs, pct: xs[int(len(xs) * pct)]

    checks = []
    for side in ("white", "black"):
        n = sum(colour[side].values())
        if n:
            checks.append(f"{side:<10} blunder rate {100 * colour[side]['blunder'] / n:5.2f}%"
                          f"   {DIM}{n:,} moves{R}")
    checks.append("")
    checks.append(f"{DIM}eval delta (pawns, mover's view){R}")
    checks.append(f"  p05 {q(deltas,.05):+.2f}    median {q(deltas,.5):+.2f}"
                  f"    p95 {q(deltas,.95):+.2f}")
    checks.append("")
    checks.append(f"{DIM}comment length (characters){R}")
    checks.append(f"  median {q(lengths,.5)}      p90 {q(lengths,.9)}      max {lengths[-1]}")
    for line in box(checks, W, title="SANITY CHECKS", colour=DIM):
        print(line)
    print()

    rows = []
    top = src.most_common(1)[0][1]
    for k, n in src.most_common(10):
        rows.append(f"{k:<26}{n:>7,}  {sparkbar(n, top, BAR)}")
    for line in box(rows, W, title="TOP SOURCES", colour=DIM):
        print(line)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=1, help="how many moves to show")
    ap.add_argument("--class", dest="classification", nargs="+", help="filter by classification")
    ap.add_argument("--motif", nargs="+", help="filter by tactical motif")
    ap.add_argument("--phase", nargs="+", help="filter by game phase")
    ap.add_argument("--source", nargs="+", help="only these study files")
    ap.add_argument("--min-comment", type=int, default=40, help="skip very short comments")
    ap.add_argument("--seed", type=int, help="reproducible sampling")
    ap.add_argument("--stats", action="store_true", help="corpus dashboard instead of moves")
    ap.add_argument("--no-color", action="store_true", help="plain text output")
    ap.add_argument("--unicode", action="store_true",
                    help="draw pieces as chess glyphs instead of letters")
    args = ap.parse_args()

    set_color(sys.stdout.isatty() and not args.no_color)

    if not ENRICHED_DIR.exists():
        sys.exit(f"{ENRICHED_DIR} does not exist — run: python3 -m kibitz.features")
    if args.stats:
        return run_stats(args)

    rng = random.Random(args.seed)
    # Reservoir sample so we never hold the whole corpus in memory.
    picked, seen = [], 0
    for meta, move, label, _uid in iter_moves(args.source):
        if not matches(move, args):
            continue
        seen += 1
        if len(picked) < args.n:
            picked.append((meta, move, label))
        elif rng.random() < args.n / seen:
            picked[rng.randrange(args.n)] = (meta, move, label)

    if not picked:
        sys.exit("nothing matched those filters.")
    for meta, move, label in picked:
        show(meta, move, label, unicode_pieces=args.unicode)
    print(f"{DIM}  showing {len(picked)} of {seen:,} matching moves{R}\n")


if __name__ == "__main__":
    main()
