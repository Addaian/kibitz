"""
Rendering one enriched move for human review.

The point of this module is the `Commentator` seam. A move is drawn as a board
plus a panel of derived features, followed by one panel per *source of
commentary*. Today the only source is the human annotator who wrote the study.
In phase 2 a fine-tuned model becomes a second source and appears beside it,
with no change to the layout code:

    render_move(meta, move, "AbasovN",
                [HumanAnnotation(), MyFineTune(adapter_path)])

Anything satisfying `Commentator` — a base model, a LoRA adapter, an API call,
a rule-based baseline — can be dropped into that list.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import chess

from kibitz import tui
from kibitz.tui import box, columns, fg

# NB: R / BOLD / DIM are rebound by tui.set_color(), so they must be read through
# the module (tui.R) rather than imported by value at import time.

PANEL_W = 42


@runtime_checkable
class Commentator(Protocol):
    """A source of commentary for a position."""

    name: str

    def comment(self, meta: dict, move: dict) -> str | None:
        """Return commentary for this move, or None if it has nothing to say."""


class HumanAnnotation:
    """The study annotator's own words — the training target."""

    name = "ANNOTATION (human)"

    def comment(self, meta: dict, move: dict) -> str | None:
        text = move.get("comment_text")
        return " ".join(text.split()) if text else None


def header(meta: dict, source: str, width: int) -> list[str]:
    white = meta.get("white") or "?"
    black = meta.get("black") or "?"
    event = " ".join(x for x in (meta.get("event") or "?",
                                 (meta.get("date") or "")[:10]) if x)
    opening = meta.get("opening") or ""
    eco = f"({meta.get('eco')})" if meta.get("eco") else ""
    sub = "  ".join(x for x in (opening, eco, f"[{source}]") if x)
    return box([f"{tui.BOLD}{white}{tui.R} {tui.DIM}vs{tui.R} {tui.BOLD}{black}{tui.R}",
                f"{tui.DIM}{event}{tui.R}",
                f"{tui.DIM}{sub}{tui.R}"], width, colour=tui.DIM)


def feature_panel(move: dict) -> list[str]:
    """What the pipeline decided about this move."""
    cls = move["move_classification"]
    colour = tui.CLASS_COLOR.get(cls, "")
    dots = "." if move["color"] == "white" else "..."
    played = f"{move['move_number']}{dots} {move['san']}"

    if move["is_best"]:
        best = f"{fg(84)}engine's own choice{tui.R}"
    elif move["best_move"]:
        best = f"preferred {tui.BOLD}{move['best_move']}{tui.R}"
    else:
        best = f"{tui.DIM}n/a{tui.R}"

    flags = []
    if move["is_forcing"]:
        flags.append(f"{fg(214)}forcing{tui.R}")
    if move["is_sacrifice"]:
        flags.append(f"{fg(207)}sacrifice{tui.R}")
    if move.get("mate_in") is not None:
        flags.append(f"{fg(196)}mate in {abs(move['mate_in'])}{tui.R}")

    def row(label, value):
        return f"{tui.DIM}{label:<7}{tui.R}{value}"

    rows = [
        f"{tui.BOLD}{played}{tui.R}",
        f"{colour}{tui.BOLD}{cls.upper()}{tui.R}",
        "",
        row("eval", f"{move['eval_before']:+.2f}  ->  {move['eval_after']:+.2f}"),
        row("delta", f"{tui.BOLD}{move['eval_delta']:+.2f}{tui.R} {tui.DIM}for {move['color']}{tui.R}"),
        row("best", best),
        row("phase", move["game_phase"]),
    ]
    if move["is_sacrifice"]:
        rows.append(row("offers", f"{fg(207)}{move['material_offered']:+.1f} pawns{tui.R}"))
    if move["tactical_motifs"]:
        rows.append(row("motifs", f"{fg(51)}{', '.join(move['tactical_motifs'])}{tui.R}"))
    if flags:
        rows.append(row("flags", "  ".join(flags)))
    return box(rows, PANEL_W, title="MOVE", colour=colour)


def eval_panel(move: dict) -> list[str]:
    ev = move["eval_after"]
    return box([tui.eval_bar(ev),
                f"{tui.BOLD}{ev:+.2f}{tui.R}   {tui.DIM}{tui.eval_verdict(ev)}{tui.R}"],
               PANEL_W, title="EVALUATION", colour=tui.DIM)


def commentary_panels(meta, move, commentators, width) -> list[str]:
    """
    One panel per source. A single source spans the full width; two or more are
    laid out side by side so they can be compared line for line.
    """
    produced = [(c.name, c.comment(meta, move) or "(no commentary)")
                for c in commentators]
    if not produced:
        return []

    if len(produced) == 1:
        name, text = produced[0]
        return box([f"{fg(229)}{l}{tui.R}" for l in tui.wrap(text, width - 6)],
                   width, title=name, colour=tui.DIM)

    each = (width - 2 * (len(produced) - 1)) // len(produced)
    blocks = [box([f"{fg(229)}{l}{tui.R}" for l in tui.wrap(text, each - 4)],
                  each, title=name, colour=tui.DIM)
              for name, text in produced]
    height = max(len(b) for b in blocks)
    blocks = [b + [""] * (height - len(b)) for b in blocks]
    return columns(*blocks)


def render_move(meta, move, source, commentators=None, unicode_pieces=False):
    """Full review of one move, as a list of printable lines."""
    commentators = commentators if commentators is not None else [HumanAnnotation()]
    width = tui.BOARD_W + 2 + PANEL_W

    uci = chess.Move.from_uci(move["uci"])
    board = tui.render_board(move["fen_after"], uci.from_square, uci.to_square,
                             unicode_pieces=unicode_pieces)
    right = feature_panel(move) + [""] + eval_panel(move)

    lines = [""] + header(meta, source, width) + [""]
    lines += columns(board, right)
    lines += [""] + commentary_panels(meta, move, commentators, width)
    return lines
