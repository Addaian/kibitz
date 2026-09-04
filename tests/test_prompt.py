"""Unit tests for kibitz.prompt."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import prompt

GOLDEN = Path(__file__).parent / "fixtures" / "golden.jsonl"
RECORDS = [json.loads(l) for l in GOLDEN.read_text().splitlines()]

META = {"white": "Alice", "black": "Bob", "event": "Test Open",
        "date": "2024.01.01", "eco": "C88", "opening": "Ruy Lopez: Closed"}
MOVE = {
    "move_number": 12, "color": "white", "san": "Nxf7", "uci": "g5f7",
    "fen_after": "8/8/8/8/8/8/8/K6k w - - 0 1",
    "eval_before": -0.30, "eval_after": 1.40, "eval_delta": 1.70,
    "best_move": "g5e6", "is_best": False, "is_forcing": True,
    "is_sacrifice": True, "material_offered": 2.2, "mate_in": None,
    "game_phase": "middlegame", "move_classification": "sacrifice",
    "tactical_motifs": ["fork"], "nags": [3],
}


def test_context_carries_the_move_and_its_features():
    ctx = prompt.format_context(META, MOVE)
    assert "Alice vs. Bob" in ctx
    assert "Ruy Lopez: Closed (C88)" in ctx
    assert "Move 12 (white): Nxf7" in ctx
    assert "Classification: sacrifice" in ctx
    assert "-0.30 -> +1.40" in ctx
    assert "Tactical motifs: fork" in ctx
    assert "Engine preferred: g5e6" in ctx


def test_nag_is_rendered_as_the_symbol_an_annotator_would_write():
    assert "Annotator's mark: !!" in prompt.format_context(META, MOVE)


def test_no_nag_means_no_line():
    assert "Annotator's mark" not in prompt.format_context(META, dict(MOVE, nags=[]))


def test_unknown_players_are_omitted_not_rendered_as_question_marks():
    """Many studies carry no player headers; the model must not learn to read '?'."""
    ctx = prompt.format_context({"white": "?", "black": "?"}, MOVE)
    assert "?" not in ctx.split("FEN")[0]
    assert "Game:" not in ctx


def test_missing_opening_is_omitted():
    assert "Opening:" not in prompt.format_context({}, MOVE)


def test_engine_preference_is_hidden_when_the_move_was_best():
    ctx = prompt.format_context(META, dict(MOVE, is_best=True))
    assert "Engine preferred" not in ctx


def test_mate_is_announced_when_present():
    assert "Forced mate in 3" in prompt.format_context(META, dict(MOVE, mate_in=-3))


def test_sacrifice_note_quantifies_the_material():
    assert "sacrifices +2.2 pawns" in prompt.format_context(META, MOVE)


# --------------------------------------------------------------------------- #
# training / inference agreement
# --------------------------------------------------------------------------- #

def test_the_user_turn_is_identical_for_training_and_inference():
    """A fine-tune degrades quietly if the serving prompt differs from the
    trained one, so the two paths must build from the same function."""
    training = prompt.build_messages(META, MOVE, "some commentary")[0]["content"]
    inference = prompt.build_prompt(META, MOVE)
    assert training == inference


def test_build_messages_pairs_user_with_assistant():
    msgs = prompt.build_messages(META, MOVE, "A fine sacrifice.")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "A fine sacrifice."


def test_build_messages_without_a_target_is_prompt_only():
    msgs = prompt.build_messages(META, MOVE)
    assert [m["role"] for m in msgs] == ["user"]


def test_the_instruction_is_present_exactly_once():
    assert prompt.build_prompt(META, MOVE).count(prompt.INSTRUCTION) == 1


@pytest.mark.parametrize("rec", RECORDS[:20], ids=lambda r: r["move"]["san"])
def test_every_real_record_builds_a_prompt(rec):
    text = prompt.build_prompt(rec["meta"], rec["move"])
    assert prompt.INSTRUCTION in text
    assert rec["move"]["san"] in text
    assert "None" not in text          # no unrendered Python values leaked in
