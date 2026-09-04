"""Unit tests for kibitz.dataset — cleaning, filtering and splitting."""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import dataset as D


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #

def test_lichess_auto_analysis_is_stripped():
    assert D.clean_comment("Inaccuracy. Best move was Qb6.") == ""
    assert D.clean_comment("Blunder. Kxe3 was best.") == ""


def test_repeated_auto_analysis_is_stripped_entirely():
    """Some records repeat the machine sentence a dozen times."""
    text = "Inaccuracy. c5 was best. " * 8
    assert D.clean_comment(text) == ""


def test_human_prose_behind_an_auto_prefix_survives():
    """The reason boilerplate is stripped rather than dropped."""
    raw = "Mistake. Best move was Kc7. Black hopes to save his knight."
    assert D.clean_comment(raw) == "Black hopes to save his knight."


def test_trailing_eval_swing_is_removed():
    assert D.clean_comment("A real comment. (0.00 → -3.21)") == "A real comment."
    assert D.clean_comment("Another one. (#3 -> #1)") == "Another one."


def test_whitespace_and_newlines_are_collapsed():
    assert D.clean_comment("A\nreal   comment\there.") == "A real comment here."


def test_ordinary_commentary_is_left_alone():
    text = "A stunning sacrifice that tears open the king's defenses."
    assert D.clean_comment(text) == text


def test_none_is_handled():
    assert D.clean_comment(None) == ""


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #

def test_short_annotations_are_rejected():
    assert not D.is_usable("Ruy Lopez", 25)
    assert D.is_usable("A stunning sacrifice that tears open the king.", 25)


@pytest.mark.parametrize("text", ["(1/2-1/2)", "1-0", "  0-1  ", "½-½"])
def test_bare_result_markers_are_rejected(text):
    assert not D.is_usable(text, 0)


@pytest.mark.parametrize("text", ["Checkmate! (1-0)", "And white wins! (1-0)",
                                  "Stalemate! (1/2-1/2)"])
def test_result_announcements_carry_no_analysis(text):
    """The result is stripped before measuring, so these fall under the bar."""
    assert not D.is_usable(text, 25)


def test_a_result_marker_inside_real_prose_is_kept():
    assert D.is_usable("White converts the extra pawn cleanly. (1-0)", 10)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #

def test_split_is_deterministic():
    assert D.split_for("src:42") == D.split_for("src:42")


def test_split_is_stable_when_new_games_are_added():
    """Hashing rather than shuffling means an existing game keeps its split."""
    before = {f"s:{i}": D.split_for(f"s:{i}") for i in range(200)}
    after = {f"s:{i}": D.split_for(f"s:{i}") for i in range(400)}
    assert all(after[k] == v for k, v in before.items())


def test_split_proportions_are_close_to_the_request():
    counts = Counter(D.split_for(f"game:{i}", val=0.1, test=0.1) for i in range(20_000))
    assert 0.08 < counts["val"] / 20_000 < 0.12
    assert 0.08 < counts["test"] / 20_000 < 0.12
    assert 0.76 < counts["train"] / 20_000 < 0.84


def test_split_respects_custom_ratios():
    counts = Counter(D.split_for(f"g:{i}", val=0.25, test=0.25) for i in range(20_000))
    assert 0.22 < counts["test"] / 20_000 < 0.28
    assert 0.45 < counts["train"] / 20_000 < 0.55


def test_every_uid_lands_in_a_real_split():
    assert {D.split_for(f"x:{i}") for i in range(500)} <= {"train", "val", "test"}


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

def test_build_record_shape():
    move = {
        "move_number": 1, "color": "white", "san": "e4", "ply": 0,
        "fen_after": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "eval_before": 0.2, "eval_after": 0.3, "eval_delta": 0.1,
        "best_move": "e2e4", "is_best": True, "is_forcing": False,
        "is_sacrifice": False, "material_offered": 0.0, "mate_in": None,
        "game_phase": "opening", "move_classification": "best",
        "tactical_motifs": [], "nags": [],
    }
    rec = D.build_record({"white": "A", "black": "B"}, move, "src", "src:1", "Good move.")
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert rec["messages"][1]["content"] == "Good move."
    assert rec["meta"]["uid"] == "src:1"
    assert rec["meta"]["classification"] == "best"
