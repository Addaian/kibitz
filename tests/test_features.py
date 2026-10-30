"""Unit tests for kibitz.features.

The engine-backed paths are tested by stubbing `features.analyse`, so the whole
suite runs without Stockfish. Everything else is pure board geometry.
"""

import sys
from pathlib import Path

import chess
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import features as F


# --------------------------------------------------------------------------- #
# comment handling
# --------------------------------------------------------------------------- #

def test_strip_annotations_removes_clock_and_eval_tags():
    assert F.strip_annotations("[%eval 0.17] [%clk 0:03:00]") == ""
    assert F.strip_annotations("A fine move [%clk 0:03:00]") == "A fine move"
    assert F.strip_annotations(None) == ""


def test_has_prose_rejects_tag_only_comments():
    assert not F.has_prose("[%eval 0.17] [%clk 0:03:00]")
    assert not F.has_prose(None)
    assert F.has_prose("Ruy Lopez")


def test_has_prose_honours_min_chars():
    assert F.has_prose("Ruy Lopez", min_chars=9)
    assert not F.has_prose("Ruy Lopez", min_chars=10)


# --------------------------------------------------------------------------- #
# game phase
# --------------------------------------------------------------------------- #

def test_detect_phase_opening_is_not_overwritten():
    """Regression: the opening branch used to fall through to the material check."""
    board = chess.Board()  # fullmove 1, full material
    assert F.detect_phase(board) == "opening"


def test_detect_phase_middlegame():
    board = chess.Board("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1B3/PPP2PPP/R2Q1RK1 w - - 0 11")
    assert F.detect_phase(board) == "middlegame"


def test_a_study_ending_set_up_at_move_one_is_an_endgame():
    """Regression: FEN chapters restart at move 1, which read as 'opening'."""
    board = chess.Board("8/5pk1/6p1/3K4/5PP1/8/8/8 b - - 0 1")
    assert F.detect_phase(board) == "endgame"


def test_a_puzzle_set_up_from_a_fen_is_never_the_opening():
    """Regression: middlegame puzzles restart at move 1 too; material alone misses them."""
    board = chess.Board("r1r3k1/pp3pp1/2n1b2p/q2p4/3N4/2P1B3/PPQ2PPP/R4RK1 w - - 0 1")
    assert F.detect_phase(board, from_position=True) == "middlegame"
    assert F.detect_phase(board) == "opening"         # a real game at move 1


def test_detect_phase_endgame():
    board = chess.Board("8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 40")
    assert F.detect_phase(board) == "endgame"


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("delta,expected", [
    (-400, "blunder"),
    (-150, "blunder"),
    (-100, "mistake"),
    (-50, "mistake"),
    (-40, "inaccuracy"),
    (-30, "inaccuracy"),
    (-10, "good"),
    (0, "good"),
])
def test_classify_move_thresholds(delta, expected):
    assert F.classify_move(delta, is_best=False) == expected


def test_classify_move_best_and_good():
    assert F.classify_move(10, is_best=True) == "best"
    assert F.classify_move(200, is_best=True) == "best"
    # A large gain that was not the engine's choice is just a good move.
    assert F.classify_move(200, is_best=False) == "good"


@pytest.mark.parametrize("delta", [-500, -100, 0, 100, 500])
@pytest.mark.parametrize("is_best", [True, False])
def test_classification_is_only_ever_a_quality_tier(delta, is_best):
    """Orthogonal properties belong in their own fields, not this enum."""
    assert F.classify_move(delta, is_best) in {
        "blunder", "mistake", "inaccuracy", "good", "best"}


def test_the_engines_own_move_is_best_whatever_the_delta():
    """Regression: depth noise between two searches labelled engine moves 'blunder'."""
    assert F.classify_move(-200, is_best=True) == "best"


def test_a_sacrifice_keeps_its_quality_tier():
    """Regression: the sacrifice label used to overwrite "best", losing the
    fact that 76% of detected sacrifices are the engine's own top move."""
    assert F.classify_move(0, is_best=True) == "best"
    assert F.classify_move(-400, is_best=False) == "blunder"


# --------------------------------------------------------------------------- #
# static exchange evaluation
# --------------------------------------------------------------------------- #

def test_see_wins_a_hanging_piece():
    # Black knight on e5 is undefended; the d4 pawn takes it for free.
    board = chess.Board("k7/8/8/4n3/3P4/8/8/K7 w - - 0 1")
    assert F.see_gain(board, chess.E5, chess.WHITE) == F.PIECE_VALUE[chess.KNIGHT]


def test_see_accounts_for_the_recapture():
    # Knight on e5 defended by the d6 pawn: dxe5 wins a knight but loses a pawn.
    board = chess.Board("k7/8/3p4/4n3/3P4/8/8/K7 w - - 0 1")
    expected = F.PIECE_VALUE[chess.KNIGHT] - F.PIECE_VALUE[chess.PAWN]
    assert F.see_gain(board, chess.E5, chess.WHITE) == expected


def test_see_declines_a_losing_capture():
    # Qxe5 wins a pawn but hangs the queen; the exchange is declined, so 0.
    board = chess.Board("k7/8/3p4/4p3/8/8/8/K3Q3 w - - 0 1")
    assert F.see_gain(board, chess.E5, chess.WHITE) == 0


def test_winnable_pieces_lists_loose_pieces_most_valuable_first():
    # Black: undefended rook on a5 and knight on e5, both attacked by pawns.
    board = chess.Board("k7/8/8/r3n3/1P1P4/8/8/7K w - - 0 1")
    assert F.winnable_pieces(board, chess.BLACK) == ["rook on a5", "knight on e5"]


def test_winnable_pieces_sees_recaptures_when_the_capturer_is_not_to_move():
    """Regression: after 3.Bb5 (Black to move) the defended c6 knight and e5 pawn
    were listed as White threats, because the capture was played in Black's colour."""
    board = chess.Board()
    for san in "e4 e5 Nf3 Nc6 Bb5".split():
        board.push_san(san)
    assert F.winnable_pieces(board, chess.BLACK) == []


def test_see_gain_leaves_the_side_to_move_alone():
    board = chess.Board("k7/8/8/4n3/3P4/8/8/K7 b - - 0 1")
    F.see_gain(board, chess.E5, chess.WHITE)
    assert board.turn == chess.BLACK


def test_winnable_pieces_ignores_defended_pieces_and_kings():
    board = chess.Board("k7/8/3p4/4n3/8/8/8/K7 w - - 0 1")   # nothing attacks e5
    assert F.winnable_pieces(board, chess.BLACK) == []


def test_see_returns_zero_on_an_empty_square():
    assert F.see_gain(chess.Board(), chess.E4, chess.WHITE) == 0


# --------------------------------------------------------------------------- #
# tactical motifs
# --------------------------------------------------------------------------- #

def test_detect_fork_knight_hits_king_and_rook():
    board = chess.Board("r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("b5c7")
    board.push(move)
    assert F.detect_fork(board, move)


def test_detect_fork_ignores_a_single_target():
    board = chess.Board("4k3/8/8/1N6/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("b5c7")
    board.push(move)
    assert not F.detect_fork(board, move)


def test_detect_pin_when_the_move_creates_it():
    before = chess.Board("4k3/4n3/8/8/8/8/8/R6K w - - 0 1")
    after = before.copy()
    after.push(chess.Move.from_uci("a1e1"))
    assert F.detect_pin(before, after, chess.WHITE)


def test_detect_pin_is_false_when_nothing_changed():
    before = chess.Board("4k3/4n3/8/8/8/8/8/R6K w - - 0 1")
    after = before.copy()
    after.push(chess.Move.from_uci("a1b1"))
    assert not F.detect_pin(before, after, chess.WHITE)


def test_detect_skewer_queen_in_front_of_rook():
    board = chess.Board("4r2k/8/8/8/4q3/8/8/R6K w - - 0 1")
    move = chess.Move.from_uci("a1e1")
    board.push(move)
    assert F.detect_skewer(board, move)


def test_detect_discovery_vacating_a_blocked_diagonal():
    # The d4 knight steps aside, unmasking Ba1 onto the h8 queen.
    board = chess.Board("k6q/8/8/8/3N4/8/8/B6K w - - 0 1")
    move = chess.Move.from_uci("d4f5")
    board.push(move)
    assert F.detect_discovery(board, move)


def test_detect_discovery_is_false_for_an_ordinary_move():
    board = chess.Board("k6q/8/8/8/8/8/3N4/B6K w - - 0 1")
    move = chess.Move.from_uci("d2f3")
    board.push(move)
    assert not F.detect_discovery(board, move)


def test_detect_promotion_threat_when_the_pawn_reaches_the_seventh():
    before = chess.Board("k7/8/P7/8/8/8/8/K7 w - - 0 1")
    after = chess.Board("k7/P7/8/8/8/8/8/K7 b - - 0 1")
    assert F.detect_promotion_threat(before, after, chess.WHITE)
    assert not F.detect_promotion_threat(before, before, chess.WHITE)
    black_before = chess.Board("k7/8/8/8/8/7p/8/K7 b - - 0 1")
    black_after = chess.Board("k7/8/8/8/8/8/7p/K7 w - - 0 1")
    assert F.detect_promotion_threat(black_before, black_after, chess.BLACK)


def test_a_pawn_already_on_the_seventh_is_not_a_new_threat():
    """Regression: every move made while a pawn waited on a7 was tagged."""
    before = chess.Board("k7/P7/8/8/8/8/8/K6R w - - 0 1")
    after = chess.Board("k7/P7/8/8/8/8/8/K5R1 b - - 1 1")      # Rg1, pawn unchanged
    assert not F.detect_promotion_threat(before, after, chess.WHITE)


# --------------------------------------------------------------------------- #
# enrich_move: point of view
# --------------------------------------------------------------------------- #

WHITE_MOVE = {
    "uci": "e2e4", "san": "e4", "color": "white", "move_number": 1,
    "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "comment": "The King's Pawn. [%clk 0:03:00]",
    "eval": None,
}

BLACK_MOVE = {
    "uci": "e7e5", "san": "e5", "color": "black", "move_number": 1,
    "fen_before": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "fen_after": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "comment": "Meeting it head on.",
    "eval": None,
}


def _stub_analyse(monkeypatch, before_cp, after_cp, best_uci=None):
    """Force `analyse` to return fixed White-POV scores for the two positions."""
    seen = {}

    def fake(board):
        # First call is fen_before, second is fen_after.
        if "before" not in seen:
            seen["before"] = True
            return before_cp, None, best_uci
        return after_cp, None, None

    monkeypatch.setattr(F, "analyse", fake)


def test_black_improving_its_position_gives_a_positive_delta(monkeypatch):
    """Regression: eval_delta used to be White-POV always, so every strong
    Black move was scored as a blunder."""
    _stub_analyse(monkeypatch, before_cp=-50, after_cp=-200)
    out = F.enrich_move(dict(BLACK_MOVE))

    assert out["eval_before"] == -0.5
    assert out["eval_after"] == -2.0
    assert out["eval_delta"] == 1.5           # good FOR BLACK
    assert out["move_classification"] != "blunder"


def test_black_worsening_its_position_gives_a_negative_delta(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=-200, after_cp=50)
    out = F.enrich_move(dict(BLACK_MOVE))

    assert out["eval_delta"] == -2.5
    assert out["move_classification"] == "blunder"


def test_white_delta_keeps_its_sign(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=20, after_cp=-200)
    out = F.enrich_move(dict(WHITE_MOVE))

    assert out["eval_delta"] == -2.2
    assert out["move_classification"] == "blunder"


def test_is_best_matches_the_engine_choice(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=20, after_cp=25, best_uci="e2e4")
    out = F.enrich_move(dict(WHITE_MOVE))
    assert out["is_best"] is True
    assert out["move_classification"] == "best"


def test_enrich_move_drops_the_ambiguous_eval_key(monkeypatch):
    """`eval` meant pawns-from-the-PGN in the parser and a centipawn delta here."""
    _stub_analyse(monkeypatch, before_cp=20, after_cp=25)
    out = F.enrich_move(dict(WHITE_MOVE))

    assert "eval" not in out
    assert out["eval_before"] == 0.2 and out["eval_after"] == 0.25
    assert out["comment_text"] == "The King's Pawn."


def test_delivering_checkmate_is_not_a_blunder(monkeypatch):
    """Regression: the game-over score of 0 made every mating move a -20 'blunder'."""
    _stub_analyse(monkeypatch, before_cp=-2000, after_cp=0, best_uci="d8h4")
    mate = {   # 1.f3 e5 2.g4 Qh4#
        "uci": "d8h4",
        "fen_before": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "fen_after": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "comment": "Fool's mate.",
    }
    out = F.enrich_move(mate)
    assert out["eval_after"] == -20.0          # White-POV: Black has mated
    assert out["eval_delta"] == 0.0
    assert out["move_classification"] == "best"


def test_any_checkmate_counts_as_the_best_move(monkeypatch):
    """When two moves mate, the one played is as good as the engine's."""
    _stub_analyse(monkeypatch, before_cp=-2000, after_cp=0, best_uci="f8c5")
    mate = {
        "uci": "d8h4",
        "fen_before": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "fen_after": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "comment": "Fool's mate.",
    }
    out = F.enrich_move(mate)
    assert out["is_best"] is True
    assert out["move_classification"] == "best"


def test_an_even_trade_does_not_leave_the_capturer_hanging(monkeypatch):
    """Nxd5 exd5-style trades: the knight can be taken back, but it took a knight."""
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    trade = {    # White Nc3xd5, Black recaptures with the e6 pawn
        "uci": "c3d5",
        "fen_before": "k7/8/4p3/3n4/8/2N5/8/K7 w - - 0 1",
        "fen_after": "k7/8/4p3/3N4/8/8/8/K7 b - - 0 1",
        "comment": "Trading.",
    }
    assert F.enrich_move(trade)["left_hanging"] == []


def test_a_pawn_attacking_two_pawns_is_not_a_fork():
    board = chess.Board("k7/8/2p1p3/8/3P4/8/8/K7 w - - 0 1")
    move = chess.Move.from_uci("d4d5")
    board.push(move)
    assert not F.detect_fork(board, move)


def test_bishop_for_knight_is_an_even_trade(monkeypatch):
    """B=330, N=320: a 10cp 'gain' is not a loose piece. 3.Bb5 a6 4.Bxc6."""
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    board = chess.Board()
    for san in "e4 e5 Nf3 Nc6 Bb5 a6".split():
        board.push_san(san)
    before = board.fen()
    board.push_san("Bxc6")
    out = F.enrich_move({"uci": "b5c6", "fen_before": before, "fen_after": board.fen(),
                         "comment": "Giving up the bishop pair."})
    assert out["left_hanging"] == []


def test_promoting_into_a_recapture_is_not_a_sacrifice(monkeypatch):
    """e8=Q+ Rxe8 Rxe8: White wins a rook for the pawn, but valuing the new
    queen made it a +4 'sacrifice' with the queen 'left hanging'."""
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    out = F.enrich_move({"uci": "e7e8q", "fen_before": "r6k/4P3/8/8/8/8/8/4R2K w - - 0 1",
                         "fen_after": "r3Q2k/8/8/8/8/8/8/4R2K b - - 0 1", "comment": "Promoting."})
    assert out["is_sacrifice"] is False
    assert out["material_offered"] <= 0
    assert out["left_hanging"] == []


def test_a_finished_game_has_no_threats(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=-2000, after_cp=0, best_uci="d8h4")
    out = F.enrich_move({
        "uci": "d8h4",
        "fen_before": "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "fen_after": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "comment": "Mate."})
    assert out["threatens"] == [] and out["left_hanging"] == []
    assert out["is_sacrifice"] is False


def test_a_piece_that_cannot_legally_be_taken_is_not_hanging(monkeypatch):
    """Regression: after a check the opponent must answer it, so a loose piece
    elsewhere is not 'left hanging' this move. Re1+ with the d2 knight en prise
    to the b4 bishop: ...Bxd2 does not answer the check."""
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    out = F.enrich_move({"uci": "a1e1", "fen_before": "4k3/8/8/8/1b6/8/3N4/R5K1 w - - 0 1",
                         "fen_after": "4k3/8/8/8/1b6/8/3N4/4R1K1 b - - 1 1",
                         "comment": "Check."})
    assert out["left_hanging"] == []


def test_only_new_threats_are_listed(monkeypatch):
    """A target the mover could already win before the move is not a new threat."""
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    # Black knight on e5 already loose to the d4 pawn; White plays a quiet king move.
    out = F.enrich_move({"uci": "a1b1", "fen_before": "k7/8/8/4n3/3P4/8/8/K7 w - - 0 1",
                         "fen_after": "k7/8/8/4n3/3P4/8/8/1K6 b - - 1 1", "comment": "Waiting."})
    assert out["threatens"] == []


def test_the_engine_move_is_also_recorded_in_san(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=20, after_cp=25, best_uci="g1f3")
    assert F.enrich_move(dict(WHITE_MOVE))["best_move_san"] == "Nf3"


def test_threats_and_hanging_pieces_are_seen_from_the_mover(monkeypatch):
    """Black's ...exd4: the capture is recorded, Black threatens nothing, and
    the d4 pawn is not "hanging": Qxd4 only trades it back."""
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    move = {
        "uci": "e5d4",
        "fen_before": "rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2",
        "fen_after": "rnbqkbnr/pppp1ppp/8/8/3p4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 3",
        "comment": "Taking.",
    }
    out = F.enrich_move(move)
    assert out["captured"] == "pawn"
    assert out["left_hanging"] == []      # Qxd4 only wins back the pawn it took
    assert out["threatens"] == []


def test_a_piece_moved_en_prise_is_left_hanging(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=0, after_cp=-300)
    blunder = {    # Nc3-e4?? walks into ...dxe4
        "uci": "c3e4",
        "fen_before": "k7/8/8/3p4/8/2N5/8/K7 w - - 0 1",
        "fen_after": "k7/8/8/3p4/4N3/8/8/K7 b - - 1 1",
        "comment": "Oops.",
    }
    assert F.enrich_move(blunder)["left_hanging"] == ["knight on e4"]


def test_enrich_move_populates_every_documented_field(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    out = F.enrich_move(dict(WHITE_MOVE))

    for field in ("eval_before", "eval_after", "eval_delta", "best_move",
                  "best_move_san", "is_best", "captured", "threatens", "left_hanging",
                  "is_forcing", "is_sacrifice", "material_offered", "mate_in",
                  "is_mate_threat", "game_phase", "move_classification",
                  "tactical_motifs", "comment_text"):
        assert field in out, field


# --------------------------------------------------------------------------- #
# enrich_game
# --------------------------------------------------------------------------- #

def test_enrich_game_keeps_only_prose_commented_moves(monkeypatch):
    import json
    monkeypatch.setattr(F, "analyse", lambda board: (0, None, None))

    game = {
        "game_id": 1,
        "source": "lichess_studies",
        "metadata": {"variant": "Standard"},
        "moves": [dict(WHITE_MOVE), dict(BLACK_MOVE, comment="[%clk 0:03:00]")],
    }
    r = F.enrich_game((json.dumps(game), 1))

    assert r["error"] is None
    assert r["enriched"] == 1 and r["skipped"] == 1 and r["invalid"] == 0
    out = json.loads(r["line"])
    assert out["total_plies"] == 2
    assert len(out["moves"]) == 1
    assert out["moves"][0]["ply"] == 0


def test_enrich_game_reports_errors_instead_of_raising():
    r = F.enrich_game(("{not json", 1))
    assert r["line"] is None and r["enriched"] == 0
    assert r["error"] and "JSONDecodeError" in r["error"]


@pytest.mark.parametrize("variant", ["Crazyhouse", "Antichess", "Horde", "Racing Kings", "Chess960"])
def test_enrich_game_skips_non_standard_variants(monkeypatch, variant):
    """Standard Stockfish evals are meaningless for variant games."""
    import json
    monkeypatch.setattr(F, "analyse", lambda board: (0, None, None))
    game = {"game_id": 1, "metadata": {"variant": variant}, "moves": [dict(WHITE_MOVE)]}
    r = F.enrich_game((json.dumps(game), 1))

    assert r["line"] is None
    assert r["enriched"] == 0
    assert r["variant"] == variant


@pytest.mark.parametrize("variant", ["Standard", "From Position", None])
def test_enrich_game_accepts_standard_play(monkeypatch, variant):
    import json
    monkeypatch.setattr(F, "analyse", lambda board: (0, None, None))
    game = {"game_id": 1, "metadata": {"variant": variant}, "moves": [dict(WHITE_MOVE)]}
    r = F.enrich_game((json.dumps(game), 1))

    assert r["enriched"] == 1 and r["variant"] is None


def test_enrich_game_drops_unanalysable_moves_but_keeps_the_game(monkeypatch):
    """One bad diagram must not cost the whole game."""
    import json

    def picky(board):
        if board.piece_at(chess.E4):        # the position after 1.e4
            raise F.UnanalysablePosition("nope")
        return 0, None, None

    monkeypatch.setattr(F, "analyse", picky)
    good = dict(BLACK_MOVE)
    game = {"game_id": 1, "metadata": {}, "moves": [dict(WHITE_MOVE), good]}
    r = F.enrich_game((json.dumps(game), 1))

    assert r["invalid"] == 2      # both annotated moves touch the 1.e4 position
    assert r["line"] is None
    assert r["error"] is None     # dropped, not crashed


# --------------------------------------------------------------------------- #
# analyse: never hand Stockfish a position that kills it
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fen,label", [
    ("8/8/5p2/5P2/6P1/8/8/8 w - - 0 1", "kingless pawn diagram"),
    ("brnkqbrn/pppp1ppp/8/4p3/4P3/6N1/PPPP1PPP/BRNKQBR1 b KQkq - 1 2", "chess960 castling"),
])
def test_analyse_refuses_illegal_positions_without_touching_the_engine(fen, label):
    """Regression: Stockfish exits with SIGSEGV on these, and every later
    analysis in the same worker then fails."""
    board = chess.Board(fen)
    assert not board.is_valid(), label

    F._engine = None          # any engine access at all would raise AttributeError
    with pytest.raises(F.UnanalysablePosition):
        F.analyse(board)


def test_analyse_uses_the_cache_before_the_engine():
    board = chess.Board()
    F._engine = None
    F._cache[F._cache_key(board)] = (42, None, "e2e4")
    try:
        assert F.analyse(board) == (42, None, "e2e4")
    finally:
        F._cache.clear()


# --------------------------------------------------------------------------- #
# engine line and game history, as the prompt shows them
# --------------------------------------------------------------------------- #

def test_line_san_numbers_moves_like_an_annotator():
    board = chess.Board()
    assert F.line_san(board, "e2e4 e7e5 g1f3 b8c6 f1b5") == "1.e4 e5 2.Nf3 Nc6"
    board.push_uci("e2e4")
    assert F.line_san(board, "c7c5 g1f3") == "1...c5 2.Nf3"


def test_line_san_stops_at_an_illegal_move():
    assert F.line_san(chess.Board(), "e2e4 e2e4") == "1.e4"


def test_previous_moves_are_the_last_few_plies_before_the_move():
    moves = [{"san": s, "color": c, "move_number": n} for s, c, n in
             [("e4", "white", 1), ("e5", "black", 1), ("Nf3", "white", 2), ("Nc6", "black", 2)]]
    assert F.previous_moves(moves, 0) is None
    assert F.previous_moves(moves, 3) == "1.e4 e5 2.Nf3"
    assert F.previous_moves(moves, 3, plies=2) == "1...e5 2.Nf3"


def test_enrich_move_keeps_the_engine_line(monkeypatch):
    monkeypatch.setattr(F, "analyse", lambda board: (0, None, "d2d4 d7d5 c2c4"))
    move = F.enrich_move(dict(WHITE_MOVE))
    assert move["best_move_san"] == "d4" and not move["is_best"]
    assert move["best_line_san"] == "1.d4 d5 2.c4"
