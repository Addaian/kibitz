"""
Regression tests against a frozen set of real enriched moves.

tests/fixtures/golden.jsonl holds 72 records covering every move
classification, every tactical motif, mates and sacrifices — with the engine's
numbers frozen alongside them. That means the whole judgement layer (delta sign,
classification thresholds, phase, motifs, SEE) can be replayed here exactly,
with no Stockfish process and no network, in milliseconds.

Regenerate after an intentional change:  python3 scripts/inspect_dataset.py freeze
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import features as F

GOLDEN = Path(__file__).parent / "fixtures" / "golden.jsonl"
RECORDS = [json.loads(line) for line in GOLDEN.read_text().splitlines()]


def ident(rec):
    m = rec["move"]
    return f"{rec['source']}:{rec['uid'].split(':')[-1]}:{m['move_number']}{m['san']}"


def replay(rec, monkeypatch):
    """Re-derive every feature from the frozen position and engine scores."""
    stored = rec["move"]
    # eval_* are pawns rounded to 2dp of an integer centipawn score, so this
    # recovers the engine's exact numbers.
    before_cp = round(stored["eval_before"] * 100)
    after_cp = round(stored["eval_after"] * 100)
    calls = []

    def fake_analyse(board):
        calls.append(board)
        if len(calls) == 1:
            return before_cp, None, stored["best_move"]
        return after_cp, stored["mate_in"], None

    monkeypatch.setattr(F, "analyse", fake_analyse)
    fresh = {
        "uci": stored["uci"],
        "fen_before": stored["fen_before"],
        "fen_after": stored["fen_after"],
        "comment": stored.get("comment"),
    }
    return F.enrich_move(fresh, from_position=bool(rec["meta"].get("from_position")))


def test_the_fixture_covers_the_whole_judgement_space():
    classes = {r["move"]["move_classification"] for r in RECORDS}
    motifs = {m for r in RECORDS for m in r["move"]["tactical_motifs"]}
    assert classes == {"best", "good", "inaccuracy", "mistake", "blunder"}
    assert any(r["move"]["is_sacrifice"] for r in RECORDS)
    assert motifs == {"fork", "pin", "skewer", "discovery",
                      "promotion", "promotion_threat"}
    assert any(r["move"]["mate_in"] is not None for r in RECORDS)
    assert any(r["move"]["is_sacrifice"] for r in RECORDS)


@pytest.mark.parametrize("rec", RECORDS, ids=ident)
def test_derived_features_are_unchanged(rec, monkeypatch):
    got = replay(rec, monkeypatch)
    want = rec["move"]
    for field in ("eval_delta", "move_classification", "game_phase",
                  "tactical_motifs", "is_best", "is_forcing",
                  "is_sacrifice", "material_offered", "is_mate_threat",
                  "best_move_san", "captured", "threatens", "left_hanging"):
        assert got[field] == want[field], (
            f"{field}: {got[field]!r} != frozen {want[field]!r}")


@pytest.mark.parametrize("rec", RECORDS, ids=ident)
def test_the_move_is_legal_in_its_own_position(rec):
    """A frozen record whose uci is not playable would mean the parser lied."""
    import chess
    move = rec["move"]
    board = chess.Board(move["fen_before"])
    uci = chess.Move.from_uci(move["uci"])
    assert uci in board.legal_moves
    assert board.san(uci) == move["san"]
    board.push(uci)
    assert board.fen() == move["fen_after"]


@pytest.mark.parametrize("rec", RECORDS, ids=ident)
def test_delta_sign_follows_the_mover(rec):
    """The bug this suite exists for: Black improving must be positive."""
    m = rec["move"]
    swing = m["eval_after"] - m["eval_before"]
    expected = swing if m["color"] == "white" else -swing
    assert m["eval_delta"] == pytest.approx(expected, abs=0.011)
