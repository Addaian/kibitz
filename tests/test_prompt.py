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
    "best_move": "g5e6", "best_move_san": "Ne6", "is_best": False, "is_forcing": True,
    "captured": "pawn", "threatens": ["rook on h8"], "left_hanging": ["knight on f7"],
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
    assert "Engine preferred: Ne6" in ctx
    assert "Captures: pawn" in ctx
    assert "Threatens to win: rook on h8" in ctx
    assert "Left hanging: knight on f7" in ctx


def test_the_engine_move_is_written_the_way_annotators_write_moves():
    """Annotators write Ne6, never g5e6; the model cannot link the two."""
    assert "g5e6" not in prompt.format_context(META, MOVE)


def test_the_annotators_own_marks_never_reach_the_prompt():
    """NAGs are the annotator's hindsight verdict on the move; inference never has them."""
    ctx = prompt.format_context(META, MOVE)          # MOVE carries nags=[3], i.e. "!!"
    assert "!!" not in ctx and "mark" not in ctx.lower()


def test_unknown_players_are_omitted_not_rendered_as_question_marks():
    """Many studies carry no player headers; the model must not learn to read '?'."""
    ctx = prompt.format_context({"white": "?", "black": "?"}, MOVE)
    assert "?" not in ctx.split("FEN")[0]
    assert "Game:" not in ctx


@pytest.mark.parametrize("date", ["????.??.??", "Marc", "08.2.2019", None])
def test_only_a_real_year_is_rendered(date):
    ctx = prompt.format_context(dict(META, date=date), MOVE)
    assert "Game: Alice vs. Bob (Test Open)" in ctx


def test_missing_opening_is_omitted():
    assert "Opening:" not in prompt.format_context({}, MOVE)


def test_unknown_opening_is_omitted_not_rendered_as_question_marks():
    """Lichess writes "?" for both headers in a quarter of the training prompts."""
    assert "Opening:" not in prompt.format_context({"opening": "?", "eco": "?"}, MOVE)
    assert "Opening: Ruy Lopez\n" in prompt.format_context(
        {"opening": "Ruy Lopez", "eco": "?"}, MOVE) + "\n"


def test_engine_preference_is_hidden_when_the_move_was_best():
    ctx = prompt.format_context(META, dict(MOVE, is_best=True))
    assert "Engine preferred" not in ctx


def test_the_engine_line_follows_its_preference_and_hides_with_it():
    move = dict(MOVE, best_line_san="12.Ne6 Qd7 13.Nxf8")
    assert "Engine preferred: Ne6\nEngine line: 12.Ne6 Qd7 13.Nxf8" in prompt.format_context(META, move)
    assert "Engine line" not in prompt.format_context(META, dict(move, is_best=True))


def test_the_moves_before_this_one_are_shown_when_known():
    ctx = prompt.format_context(META, dict(MOVE, previous_moves="10...Nf6 11.Ng5 d6"))
    assert "Previous moves: 10...Nf6 11.Ng5 d6\nMove 12 (white): Nxf7" in ctx
    assert "Previous moves" not in prompt.format_context(META, MOVE)


def test_mate_is_announced_with_the_side_that_delivers_it():
    """Delivering mate and walking into it must not render the same."""
    assert "Black mates in 3" in prompt.format_context(META, dict(MOVE, mate_in=-3))
    assert "White mates in 2" in prompt.format_context(META, dict(MOVE, mate_in=2))


def test_sacrifice_note_quantifies_the_material():
    assert "sacrifices +2.2 pawns" in prompt.format_context(META, MOVE)


# --------------------------------------------------------------------------- #
# training / inference agreement
# --------------------------------------------------------------------------- #

def test_the_user_turn_is_identical_for_training_and_inference():
    """A fine-tune degrades quietly if the serving prompt differs from the
    trained one, so the two paths must build from the same function."""
    training = prompt.build_messages(META, MOVE, "some commentary")[0]["content"]
    inference = prompt.build_prompt(META, MOVE, "brief")
    assert training == inference


def test_training_length_follows_the_annotator():
    short = " ".join(["word"] * prompt.BRIEF_MAX_WORDS)
    long = short + " more"
    assert "Length: brief" in prompt.build_messages(META, MOVE, short)[0]["content"]
    assert "Length: full" in prompt.build_messages(META, MOVE, long)[0]["content"]


QUIET = dict(MOVE, move_classification="best", tactical_motifs=[],
             is_sacrifice=False, mate_in=None)


def test_quiet_moves_get_a_brief_comment_at_inference():
    assert "Length: brief" in prompt.build_messages(META, QUIET)[0]["content"]


@pytest.mark.parametrize("change", [
    {"move_classification": "blunder"}, {"tactical_motifs": ["pin"]},
    {"is_sacrifice": True}, {"mate_in": 4},
])
def test_notable_moves_get_a_full_comment_at_inference(change):
    move = dict(QUIET, **change)
    assert "Length: full" in prompt.build_messages(META, move)[0]["content"]


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


# --------------------------------------------------------------------------- #
# causality — the guarantee live commentary depends on
# --------------------------------------------------------------------------- #

FULL_GAME_META = {
    "white": "Alice", "black": "Bob", "event": "Test Open", "date": "2024.01.01",
    "eco": "C88", "opening": "Ruy Lopez: Closed",
    "result": "1-0", "variant": "Standard",
}


@pytest.mark.parametrize("rec", RECORDS[:15], ids=lambda r: r["move"]["san"])
def test_the_prompt_never_reveals_the_result(rec):
    """Kibitz must be able to comment on a game in progress, so nothing in the
    prompt may depend on how the game ends."""
    meta = dict(rec["meta"], result="1-0", termination="Normal")
    text = prompt.build_prompt(meta, rec["move"])
    for leak in ("1-0", "0-1", "1/2-1/2", "Normal", "result", "Result"):
        assert leak not in text, f"future information leaked: {leak!r}"


@pytest.mark.parametrize("rec", RECORDS[:15], ids=lambda r: r["move"]["san"])
def test_the_prompt_is_identical_live_and_retrospective(rec):
    """
    The same move must produce the same prompt whether the rest of the game is
    known or not. This is what makes live and finished-game commentary one code
    path rather than two.
    """
    retrospective = prompt.build_prompt(FULL_GAME_META, rec["move"])
    live = prompt.build_prompt(
        {k: v for k, v in FULL_GAME_META.items() if k not in ("result", "termination")},
        rec["move"])
    assert retrospective == live


def test_only_the_moves_own_position_reaches_the_prompt():
    """A move record carries no field describing a later position, so a prompt
    cannot depend on one."""
    move = dict(MOVE)
    text = prompt.build_prompt(META, move)
    assert move["fen_after"] in text
    # fen_after is the position *created by this move* — the frontier, not the future.
    import chess
    board = chess.Board(move["fen_after"])
    assert board.fullmove_number >= 1
