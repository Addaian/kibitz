"""
Checking commentary against the board it describes.

BLEU and ROUGE cannot tell "the knight forks king and rook" from the same
sentence about a position with no knight in it. This module checks the two
kinds of claim that can be checked mechanically:

* **Piece placement** — "the knight on e5", "the e5 knight". False if no such
  piece stands there before or after the move.
* **Moves named** — "Qxh7", "Rfe1". Unplayable if the move is legal in none of:
  the position before the move (an alternative), the position after it (the
  reply), or the position after it with the mover to play again (a threat).
  Moves deeper in a quoted line also land here, so this is a rate to compare
  against the human references, not a count of errors.

Pawn pushes ("e4") are not counted as moves: they cannot be told apart from
square names.
"""

from __future__ import annotations

import re

import chess

_PIECE = r"(king|queen|rook|bishop|knight|pawn)"
_PIECE_ON = re.compile(rf"\b{_PIECE}s?\s+on\s+([a-h][1-8])\b", re.I)
_SQUARE_PIECE = re.compile(rf"\b([a-h][1-8])[- ]{_PIECE}\b", re.I)
_MOVE = re.compile(
    r"(?<![\w-])([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)(?![\w-])")
_PIECE_TYPE = {name: chess.Piece.from_symbol(sym).piece_type
               for name, sym in [("king", "K"), ("queen", "Q"), ("rook", "R"),
                                 ("bishop", "B"), ("knight", "N"), ("pawn", "P")]}


def _playable(san: str, boards: list[chess.Board]) -> bool:
    for board in boards:
        try:
            board.parse_san(san)
            return True
        except chess.AmbiguousMoveError:
            return True
        except ValueError:
            continue
    return False


def check(move: dict, text: str) -> dict:
    """Count checkable claims in `text` about `move`, and how many fail."""
    before = chess.Board(move["fen_before"])
    after = chess.Board(move["fen_after"])
    boards = [before, after]
    if not after.is_check():
        threat = after.copy(stack=False)
        threat.push(chess.Move.null())
        boards.append(threat)

    placements = [(p, sq) for p, sq in _PIECE_ON.findall(text)]
    placements += [(p, sq) for sq, p in _SQUARE_PIECE.findall(text)]
    false_placements = []
    for piece, square in placements:
        sq = chess.parse_square(square.lower())
        kind = _PIECE_TYPE[piece.lower()]
        if not any(b.piece_type_at(sq) == kind for b in (before, after)):
            false_placements.append(f"{piece.lower()} on {square.lower()}")

    moves = _MOVE.findall(text)
    unplayable = [m for m in moves if not _playable(m, boards)]

    return {
        "placements": len(placements),
        "false_placements": false_placements,
        "moves": len(moves),
        "unplayable_moves": unplayable,
    }


def summarise(results: list[dict]) -> dict:
    """Share of texts (in %) with at least one failing claim of each kind."""
    n = max(1, len(results))
    return {
        "false placement %": 100 * sum(bool(r["false_placements"]) for r in results) / n,
        "unplayable move %": 100 * sum(bool(r["unplayable_moves"]) for r in results) / n,
    }
