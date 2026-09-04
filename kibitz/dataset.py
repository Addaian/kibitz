"""
Cleaning and shaping enriched moves into training pairs.

`features.py` adds chess knowledge; this stage adds none. It only decides which
annotations are worth training on, tidies their text, and splits them.

Two things here matter more than they look:

* **Lichess auto-analysis is not human commentary.** Roughly one annotated move
  in ten carries machine text like "Inaccuracy. Best move was Qb6." Training on
  it teaches the model to emit engine boilerplate. It is stripped rather than
  dropped, because a minority of those comments have real prose behind the
  machine prefix.

* **Splitting is by game, not by move.** Consecutive moves of one game are
  highly correlated, so splitting by move would leak the test set into training.
  The split is a stable hash of the game id, so re-running or adding new studies
  never reshuffles games between splits.
"""

from __future__ import annotations

import hashlib
import re

# "Inaccuracy. Best move was Qb6." / "Blunder. Kxe3 was best."
_AUTO = re.compile(
    r'^\s*(?:Inaccuracy|Mistake|Blunder)\.\s*'
    r'(?:Best move was\s+[^.]{1,12}\.|[^.\s]{1,12}\s+was best\.)\s*', re.I)
# Trailing eval swing, "(0.00 → -3.21)" or "(#3 -> #1)"
_EVAL_TAIL = re.compile(
    r'\s*\((?:[-+]?\d+\.?\d*|#-?\d+)\s*(?:→|->|-->)\s*(?:[-+]?\d+\.?\d*|#-?\d+)\)\s*')
_STOCK_PHRASE = re.compile(r'\s*(?:Checkmate is now unavoidable|Lost forever)\.?\s*', re.I)
_RESULT = re.compile(r'\(?\s*(?:1-0|0-1|1/2-1/2|½-½)\s*\)?')


def clean_comment(text: str | None) -> str:
    """Strip engine boilerplate and collapse whitespace, keeping any human prose."""
    text = " ".join((text or "").split())
    previous = None
    while previous != text:          # the boilerplate is often repeated verbatim
        previous = text
        text = _AUTO.sub("", text, count=1)
        text = _EVAL_TAIL.sub(" ", text)
        text = _STOCK_PHRASE.sub(" ", text)
        text = " ".join(text.split())
    return text


def is_usable(text: str, min_chars: int) -> bool:
    """
    A target worth training on.

    The game result is removed before the length is measured, so the bar applies
    to actual analysis. Otherwise "Checkmate! (1-0)" and "Stalemate! (1/2-1/2)"
    coast past on punctuation while saying nothing about the position.
    """
    analysis = _RESULT.sub("", text).strip()
    if len(analysis) < min_chars:
        return False
    if not re.search(r'[a-zA-Z]', analysis):
        return False
    return True


def split_for(uid: str, val: float = 0.1, test: float = 0.1, salt: str = "kibitz") -> str:
    """
    Deterministic split assignment for a game.

    Hashing the game id (rather than shuffling) means the split survives adding
    new studies: an existing game keeps whichever side of the fence it was on.
    """
    digest = hashlib.sha1(f"{salt}:{uid}".encode()).hexdigest()[:8]
    position = int(digest, 16) / 0xFFFFFFFF
    if position < test:
        return "test"
    if position < test + val:
        return "val"
    return "train"


def build_record(meta: dict, move: dict, source: str, uid: str, target: str) -> dict:
    """
    One training pair, Claude Messages format.

    `meta` carries provenance so a generated comment can be traced back to its
    game and compared against the annotator. Pass --strict to build_dataset to
    emit only the `messages` key.
    """
    from kibitz import prompt

    return {
        "messages": prompt.build_messages(meta, move, target),
        "meta": {
            "source": source,
            "uid": uid,
            "ply": move.get("ply"),
            "san": move["san"],
            "classification": move["move_classification"],
            "phase": move["game_phase"],
            "motifs": move["tactical_motifs"],
        },
    }
