"""
Terminal rendering primitives shared by every kibitz command.

Kept free of chess *judgement* on purpose: this module knows how to draw a
board, a panel and a bar, and nothing about what makes a move good. That keeps
it reusable by the dataset inspector now and by model evaluation later.

Everything is ANSI-aware — `vlen` measures visible width so frames line up
whether or not colour is enabled.
"""

from __future__ import annotations

import re
import shutil

import chess

__all__ = [
    "set_color", "color_enabled", "fg", "bg", "vlen", "vpad", "wrap",
    "box", "columns", "render_board", "eval_bar", "eval_verdict", "sparkbar",
    "BOARD_W", "CLASS_COLOR", "R", "BOLD", "DIM",
]

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

COLOR = True
R = BOLD = DIM = ""
CLASS_COLOR: dict[str, str] = {}

BOARD_W = 37


def fg(n: int) -> str:
    return f"\x1b[38;5;{n}m" if COLOR else ""


def bg(n: int) -> str:
    return f"\x1b[48;5;{n}m" if COLOR else ""


def color_enabled() -> bool:
    return COLOR


def set_color(enabled: bool) -> None:
    """Recompute every escape sequence for colour or mono output."""
    global COLOR, R, BOLD, DIM, CLASS_COLOR
    COLOR = enabled
    R = "\x1b[0m" if enabled else ""
    BOLD = "\x1b[1m" if enabled else ""
    DIM = "\x1b[2m" if enabled else ""
    CLASS_COLOR = {
        "blunder": fg(196), "mistake": fg(208), "inaccuracy": fg(220),
        "good": fg(250), "best": fg(84),
    }


# --------------------------------------------------------------------------- #
# measurement and layout
# --------------------------------------------------------------------------- #

def vlen(s: str) -> int:
    """Visible width, ignoring colour escapes."""
    return len(_ANSI.sub("", s))


def vpad(s: str, width: int) -> str:
    return s + " " * max(0, width - vlen(s))


def wrap(text: str, width: int) -> list[str]:
    """
    Greedy word wrap. Tokens longer than `width` — URLs, engine PV dumps — are
    hard-split rather than allowed to overflow, which would burst any frame
    drawn around them.
    """
    if width <= 0:
        return [text]

    lines, cur = [], ""
    for word in text.split():
        while len(word) > width:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur = f"{cur} {word}"
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def box(lines, width: int, title: str | None = None, colour: str = "") -> list[str]:
    """Frame `lines` in an ASCII panel of exactly `width` visible columns."""
    inner = width - 4
    if title:
        rule = "-" * max(0, width - vlen(title) - 5)
        top = f"{colour}+- {BOLD}{title}{R}{colour} {rule}+{R}"
    else:
        top = f"{colour}+{'-' * (width - 2)}+{R}"

    out = [top]
    for line in lines:
        # Long lines would blow the frame open, so fold them instead.
        pieces = [line] if vlen(line) <= inner else wrap(_ANSI.sub("", line), inner)
        for piece in pieces:
            out.append(f"{colour}|{R} {vpad(piece, inner)} {colour}|{R}")
    out.append(f"{colour}+{'-' * (width - 2)}+{R}")
    return out


def columns(*blocks, gap: int = 2) -> list[str]:
    """Lay blocks of lines side by side, padding each to its widest line."""
    widths = [max((vlen(l) for l in b), default=0) for b in blocks]
    height = max((len(b) for b in blocks), default=0)
    out = []
    for i in range(height):
        row = ""
        for block, width in zip(blocks, widths):
            cell = block[i] if i < len(block) else ""
            row += vpad(cell, width) + " " * gap
        out.append(row.rstrip())
    return out


def terminal_width(default: int = 100) -> int:
    return shutil.get_terminal_size((default, 24)).columns


# --------------------------------------------------------------------------- #
# chess-specific drawing
# --------------------------------------------------------------------------- #

# Outline glyphs for White, filled for Black, so the sides stay distinguishable
# even with colour switched off.
GLYPH_WHITE = {"p": "♙", "n": "♘", "b": "♗", "r": "♖", "q": "♕", "k": "♔"}
GLYPH_BLACK = {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}


def render_board(fen, from_sq=None, to_sq=None, flip=False, unicode_pieces=False):
    """
    An 8x8 board drawn as ASCII art. Empty dark squares carry a faint dot so the
    checkerboard stays visible; the move's origin is (bracketed) and its
    destination [bracketed].

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


def eval_bar(eval_white: float, width: int = 30) -> str:
    """Advantage as a filled bar: all-left is White winning, all-right is Black."""
    share = max(0.0, min(1.0, eval_white / 12 + 0.5))
    filled = round(share * width)
    return f"[{fg(255)}{'#' * filled}{R}{DIM}{'.' * (width - filled)}{R}]"


def eval_verdict(ev: float) -> str:
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


def sparkbar(n: int, top: int, width: int = 26) -> str:
    return f"{fg(74)}{'#' * max(1, round(n / top * width))}{R}"


set_color(True)
