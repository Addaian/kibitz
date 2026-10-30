"""Unit tests for kibitz.grounding — checking commentary against the board."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import grounding as G

# 1.e4 e5 2.Nf3, Black to move.
MOVE = {
    "san": "Nf3",
    "fen_before": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "fen_after": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
}


@pytest.mark.parametrize("text", [
    "The knight on f3 attacks the pawn on e5.",
    "The f3 knight eyes e5.",
    "The knight on g1 comes out.",          # true before the move
])
def test_true_placements_pass(text):
    assert G.check(MOVE, text)["false_placements"] == []


@pytest.mark.parametrize("text, wrong", [
    ("The bishop on f3 is strong.", "bishop on f3"),
    ("Black's queen on h4 is loose.", "queen on h4"),
])
def test_false_placements_are_caught(text, wrong):
    assert G.check(MOVE, text)["false_placements"] == [wrong]


@pytest.mark.parametrize("text", [
    "Nc6 defends the pawn.",                # the reply
    "Bc4 was also good.",                   # an alternative to the move played
    "White threatens Nxe5.",                # the mover's follow-up
])
def test_moves_playable_nearby_pass(text):
    assert G.check(MOVE, text)["unplayable_moves"] == []


def test_a_move_no_nearby_position_allows_is_flagged():
    assert G.check(MOVE, "Qxh7 wins.")["unplayable_moves"] == ["Qxh7"]


def test_pawn_pushes_are_not_counted_as_moves():
    """'e5' could be a square or a move; only unambiguous notation counts."""
    assert G.check(MOVE, "The e5 pawn is attacked.")["moves"] == 0


def test_summarise_reports_the_share_of_texts_with_an_error():
    results = [G.check(MOVE, t) for t in ("Qxh7 wins.", "Nc6 defends.", "The bishop on f3.")]
    summary = G.summarise(results)
    assert summary["unplayable move %"] == pytest.approx(100 / 3)
    assert summary["false placement %"] == pytest.approx(100 / 3)
