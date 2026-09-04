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


def test_a_sacrifice_keeps_its_quality_tier():
    """Regression: the sacrifice label used to overwrite "best", losing the
    fact that 78% of detected sacrifices are the engine's own top move."""
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


def test_detect_promotion_threat_on_the_seventh():
    assert F.detect_promotion_threat(chess.Board("k7/P7/8/8/8/8/8/K7 w - - 0 1"), chess.WHITE)
    assert not F.detect_promotion_threat(chess.Board("k7/8/P7/8/8/8/8/K7 w - - 0 1"), chess.WHITE)
    assert F.detect_promotion_threat(chess.Board("k7/8/8/8/8/8/7p/K7 w - - 0 1"), chess.BLACK)


# --------------------------------------------------------------------------- #
# enrich_move: point of view
# --------------------------------------------------------------------------- #

WHITE_MOVE = {
    "uci": "e2e4",
    "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "fen_after": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "comment": "The King's Pawn. [%clk 0:03:00]",
    "eval": None,
}

BLACK_MOVE = {
    "uci": "e7e5",
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


def test_enrich_move_populates_every_documented_field(monkeypatch):
    _stub_analyse(monkeypatch, before_cp=0, after_cp=0)
    out = F.enrich_move(dict(WHITE_MOVE))

    for field in ("eval_before", "eval_after", "eval_delta", "best_move", "is_best",
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
