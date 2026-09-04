"""Unit tests for the shared terminal rendering layer."""

import json
import re
import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import review, tui

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GOLDEN = Path(__file__).parent / "fixtures" / "golden.jsonl"


@pytest.fixture(autouse=True)
def mono():
    """Default every test to colour-off, and restore afterwards."""
    tui.set_color(False)
    yield
    tui.set_color(True)


@pytest.fixture
def record():
    return json.loads(GOLDEN.read_text().splitlines()[0])


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #

def test_vlen_ignores_colour_escapes():
    tui.set_color(True)
    assert tui.vlen(f"{tui.BOLD}abc{tui.R}") == 3
    assert tui.vlen(f"{tui.fg(196)}hello{tui.R}") == 5


def test_vpad_pads_to_visible_width():
    tui.set_color(True)
    padded = tui.vpad(f"{tui.BOLD}ab{tui.R}", 6)
    assert tui.vlen(padded) == 6


def test_vpad_never_truncates():
    assert tui.vpad("abcdef", 3) == "abcdef"


def test_wrap_breaks_on_words():
    assert tui.wrap("one two three four", 9) == ["one two", "three", "four"]


def test_wrap_of_empty_text_is_one_empty_line():
    assert tui.wrap("", 10) == [""]


# --------------------------------------------------------------------------- #
# panels
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("width", [20, 40, 66, 81])
def test_box_is_exactly_the_requested_width(width):
    for line in tui.box(["a", "bb"], width, title="T"):
        assert tui.vlen(line) == width


def test_box_folds_lines_that_would_burst_the_frame():
    long = "x" * 200
    lines = tui.box([long], 30)
    assert all(tui.vlen(l) == 30 for l in lines)
    assert len(lines) > 3                       # it wrapped rather than overflowed


def test_box_width_is_identical_with_colour_on_and_off():
    tui.set_color(False)
    mono = [tui.vlen(l) for l in tui.box(["hello"], 40, title="T")]
    tui.set_color(True)
    colour = [tui.vlen(l) for l in tui.box(["hello"], 40, title="T")]
    assert mono == colour


def test_columns_aligns_ragged_blocks():
    out = tui.columns(["aaa", "b"], ["1", "2", "3"])
    assert len(out) == 3
    assert out[1].startswith("b")
    assert out[2].startswith(" ")               # first block ran out, padded


# --------------------------------------------------------------------------- #
# board
# --------------------------------------------------------------------------- #

def test_board_has_a_rule_between_every_rank():
    lines = tui.render_board(chess.STARTING_FEN)
    assert len(lines) == 1 + 8 * 2 + 1          # top rule + (rank, rule) x8 + files


def test_board_marks_origin_and_destination():
    board = chess.Board()
    board.push_san("e4")
    out = "\n".join(tui.render_board(board.fen(), chess.E2, chess.E4))
    assert "( )" in out                          # e2 vacated
    assert "[P]" in out                          # pawn landed on e4


def test_board_uses_case_to_distinguish_sides():
    out = "\n".join(tui.render_board(chess.STARTING_FEN))
    assert " R " in out and " r " in out


def test_board_unicode_uses_distinct_glyphs_per_side():
    """Filled glyphs for both colours would be unreadable without colour."""
    out = "\n".join(tui.render_board(chess.STARTING_FEN, unicode_pieces=True))
    assert "♖" in out and "♜" in out
    assert not (set("♖♘♗♕♔") & set("♜♞♝♛♚"))


def test_board_flip_reverses_the_files():
    normal = tui.render_board(chess.STARTING_FEN)[-1]
    flipped = tui.render_board(chess.STARTING_FEN, flip=True)[-1]
    assert normal.strip().split() == list("abcdefgh")
    assert flipped.strip().split() == list("hgfedcba")


# --------------------------------------------------------------------------- #
# bars
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ev", [-50, -3.0, 0.0, 3.0, 50])
def test_eval_bar_stays_inside_its_width(ev):
    assert tui.vlen(tui.eval_bar(ev, width=30)) == 32     # 30 cells + brackets


def test_eval_bar_direction():
    assert tui.eval_bar(-20, 10).count("#") == 0          # black winning: empty
    assert tui.eval_bar(20, 10).count("#") == 10          # white winning: full


@pytest.mark.parametrize("ev,expected", [
    (0.0, "level"), (0.2, "level"),
    (0.5, "white slightly better"), (-0.5, "black slightly better"),
    (1.0, "white better"), (2.0, "white clearly better"), (5.0, "white winning"),
])
def test_eval_verdict_wording(ev, expected):
    assert tui.eval_verdict(ev) == expected


def test_sparkbar_always_shows_something():
    assert "#" in tui.sparkbar(1, 100_000, width=20)


# --------------------------------------------------------------------------- #
# colour discipline
# --------------------------------------------------------------------------- #

def test_mono_mode_emits_no_escape_codes(record):
    """Regression: R/BOLD/DIM imported by value ignored set_color(False),
    leaking escape codes into piped output."""
    tui.set_color(False)
    out = "\n".join(review.render_move(record["meta"], record["move"], "test"))
    assert ANSI.search(out) is None


def test_colour_mode_does_emit_escape_codes(record):
    tui.set_color(True)
    out = "\n".join(review.render_move(record["meta"], record["move"], "test"))
    assert len(ANSI.findall(out)) > 50


@pytest.mark.parametrize("colour", [True, False])
def test_rendered_move_frames_line_up_in_both_modes(record, colour):
    tui.set_color(colour)
    lines = review.render_move(record["meta"], record["move"], "test")
    widths = {tui.vlen(l) for l in lines if ANSI.sub("", l).startswith("+")}
    assert len(widths) == 1, f"ragged frames: {sorted(widths)}"


# --------------------------------------------------------------------------- #
# the Commentator seam
# --------------------------------------------------------------------------- #

def test_human_annotation_is_a_commentator():
    assert isinstance(review.HumanAnnotation(), review.Commentator)


def test_human_annotation_returns_the_stored_comment(record):
    text = review.HumanAnnotation().comment(record["meta"], record["move"])
    assert text == " ".join(record["move"]["comment_text"].split())


def test_a_second_commentator_renders_beside_the_first(record):
    """Phase 2 drops a model in here; the layout must already handle it."""

    class Stub:
        name = "KIBITZ (generated)"
        def comment(self, meta, move):
            return "A plausible sounding sentence about the position."

    lines = review.render_move(record["meta"], record["move"], "test",
                               [review.HumanAnnotation(), Stub()])
    out = "\n".join(lines)
    assert "ANNOTATION (human)" in out
    assert "KIBITZ (generated)" in out
    # side by side, not stacked
    assert any("ANNOTATION (human)" in l and "KIBITZ (generated)" in l for l in lines)


def test_a_commentator_with_nothing_to_say_is_handled(record):
    class Silent:
        name = "SILENT"
        def comment(self, meta, move):
            return None

    out = "\n".join(review.render_move(record["meta"], record["move"], "test", [Silent()]))
    assert "(no commentary)" in out
