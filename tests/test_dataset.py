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


def test_lichess_mate_sequence_analysis_is_stripped():
    assert D.clean_comment("Lost forced checkmate sequence. Qf5+ was best.") == ""
    assert D.clean_comment("Not the best checkmate sequence. Best move was Re1+.") == ""


def test_an_eval_swing_into_mate_is_removed():
    assert D.clean_comment("A real idea. (3.12 → Mate in 12)") == "A real idea."


def test_links_are_stripped_from_commentary():
    raw = "See the full game here https://lichess.org/study/dYFcDtRq for more."
    assert D.clean_comment(raw) == "See the full game here for more."


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
# non-prose targets
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, reason", [
    ("d=22, sd=22, pd=Re1, mt=11322, pv=Rb2 Rh1 Rb4 Kg2", "engine log"),
    ("depth 32: -0.40, Re1: -0.41, Nc3: -0.43", "engine log"),
    ("aka C01 French Defense: Exchange Variation.", "opening label"),
    ("B32 Sicilian Defense: Kalashnikov Variation", "opening label"),
    ("Beyaz yeni bir savunma düzenine geçti ve artık g3'ü daha iyi koruyor", "not English prose"),
    ("Las blancas recuperaron su peón, pero tenemos mejor estructura", "not English prose"),
    ("https://www.youtube.com/watch?v=hx2AN9xvNhs", "not English prose"),
])
def test_non_prose_targets_are_named(text, reason):
    assert D.non_prose_reason(text) == reason


@pytest.mark.parametrize("text", [
    "Why block your bishop's natural diagonal?",
    "Instead, 18.Rc2 (second rank defense) Ng4 19.Ng3.",
    "Removes annoying Bg5 possibilities",
    "A stunning sacrifice that tears open the king.",
    "Lichess calls this B90 Sicilian Defense: Najdorf Variation, English Attack.",
])
def test_english_commentary_is_kept(text):
    assert D.non_prose_reason(text) is None


@pytest.mark.parametrize("text", [
    "? 31...Re8 32.Kxf4 Ne2+ 33.Kg4 Nc3 34.Re3 Rd8 +- +2.52 (35 ply)",
    "Correct. SF: +4.5/27. 3.h5 is a winning attack.",
])
def test_pasted_engine_lines_are_engine_logs(text):
    assert D.non_prose_reason(text) == "engine log"


# --------------------------------------------------------------------------- #
# ungrounded targets: facts the prompt never shows the model
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text, reason", [
    ("White's win rate is sitting at 62%!", "database statistics"),
    ("The most popular move, but it's bad because of the following move:", "database statistics"),
    ("This is the most played move in the LMDB.", "database statistics"),
    ("With the two ideas shown by the arrows; d6d5 and Qb6.", "addresses the reader"),
    ("Each chapter having 3. d5 is marked with a green book.", "addresses the reader"),
    ("Excellent! Do you see the point of this Queen sacrifice?", "addresses the reader"),
    ("Black currently has a Knight for a pawn. What should White play?", "addresses the reader"),
    ("8.h3 was played in Votava vs C Lujan, 2005 (1-0)", "other games"),
    ("We have already seen that this natural move is a fatal mistake.", "other games"),
    ("A new move in the match. In Game 5 Nepo chose the more popular 11.c3", "other games"),
    ("White went on to win with his extra pawn.", "hindsight"),
])
def test_ungrounded_targets_are_named(text, reason):
    assert D.ungrounded_reason(text) == reason


@pytest.mark.parametrize("text", [
    "Why block your bishop's natural diagonal?",          # rhetorical, not a quiz
    "Equalizing pressure on e4. The same could be said of 6...Bb4 and 6...g5.",
    "Eventually, White will play for e4.",                # a plan, not hindsight
    "Again Re1+ is a threat and cannot be stopped.",
    "White threatens an embarrassing attack on the d6 pawn with Nc4 and Bf4.",
])
def test_grounded_commentary_is_kept(text):
    assert D.ungrounded_reason(text) is None


@pytest.mark.parametrize("raw, cleaned", [
    ("Correct. Forking the Bishop and pawn.", "Forking the Bishop and pawn."),
    ("Exactly! Now the a-pawn cannot move. Play 1...g1=Q for Black.", "Now the a-pawn cannot move."),
    ("AL: A clear strategic error that gives up the d5 square.",
     "A clear strategic error that gives up the d5 square."),
    ("White's idea is to attack the pinned knight. Play the move indicated for Black.",
     "White's idea is to attack the pinned knight."),
    ("Black resigned. Mate is coming in two moves.", "Mate is coming in two moves."),
    ("1-0 Black resigns. Zugzwang: Kc8 cannot be prevented.", "Zugzwang: Kc8 cannot be prevented."),
])
def test_study_scaffolding_is_trimmed(raw, cleaned):
    assert D.clean_comment(raw) == cleaned


def test_a_verdict_word_inside_a_sentence_is_kept():
    text = "No English Opening today; Nepo is back to 1.e4."
    assert D.clean_comment(text) == text


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #

def test_chapters_of_one_study_share_a_split_key():
    a = D.split_key("src", {"study": "RoBvWqfx"}, "src:1")
    b = D.split_key("src", {"study": "RoBvWqfx"}, "src:2")
    assert a == b


def test_studies_sharing_a_position_and_comment_are_grouped():
    """The same game published in two studies must not straddle splits."""
    group = D.merge_duplicate_keys([
        ("a:study:1", "fenX|Great move."),
        ("b:study:2", "fenX|Great move."),
        ("b:study:2", "fenY|Something else."),
        ("c:study:3", "fenY|Something else."),
        ("d:study:4", "fenZ|Unrelated."),
    ])
    assert group["a:study:1"] == group["b:study:2"] == group["c:study:3"]
    assert group["d:study:4"] == "d:study:4"


def test_grouping_does_not_depend_on_input_order():
    pairs = [("x", "s1"), ("y", "s1"), ("z", "s2"), ("y", "s2")]
    assert D.merge_duplicate_keys(pairs) == D.merge_duplicate_keys(reversed(pairs))


def test_a_game_without_a_study_is_its_own_split_key():
    assert D.split_key("src", {}, "src:7") == "src:7"


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
