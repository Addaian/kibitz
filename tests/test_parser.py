"""Unit tests for kibitz.parser.

Each test writes a small PGN to a temp directory and parses it, so nothing
touches data/. `parse_pgn` derives the source name from the parent folder and
writes to PROCESSED_DIR, both of which are redirected here.
"""

import json
import sys
from pathlib import Path

import chess
import chess.pgn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import parser as P


HEADERS = '''[Event "Test Event"]
[Site "Nowhere"]
[Date "2024.01.01"]
[Round "1"]
[White "Alice"]
[Black "Bob"]
[Result "1/2-1/2"]
'''


@pytest.fixture
def parse(tmp_path, monkeypatch):
    """Write a PGN and return the parsed game records."""
    out_dir = tmp_path / "out"
    monkeypatch.setattr(P, "PROCESSED_DIR", out_dir)

    def _parse(pgn_text: str, name: str = "someone_studies"):
        src = tmp_path / "lichess_studies"
        src.mkdir(exist_ok=True)
        pgn_file = src / f"{name}.pgn"
        pgn_file.write_text(pgn_text)
        P.parse_pgn(str(pgn_file))
        out = out_dir / f"{name}.jsonl"
        return [json.loads(line) for line in out.read_text().splitlines()]

    return _parse


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #

def test_metadata_is_lifted_from_the_headers(parse):
    games = parse(HEADERS + '\n1. e4 e5 1/2-1/2\n')
    assert len(games) == 1
    meta = games[0]["metadata"]
    assert meta["white"] == "Alice"
    assert meta["black"] == "Bob"
    assert meta["result"] == "1/2-1/2"
    assert meta["event"] == "Test Event"
    assert games[0]["source"] == "lichess_studies"
    assert games[0]["game_id"] == 1


def test_variant_header_is_captured(parse):
    """features.py needs this to skip games a standard engine cannot judge."""
    games = parse(HEADERS + '[Variant "Crazyhouse"]\n\n1. e4 e5 *\n')
    assert games[0]["metadata"]["variant"] == "Crazyhouse"


def test_missing_variant_header_is_none(parse):
    games = parse(HEADERS + '\n1. e4 e5 *\n')
    assert games[0]["metadata"]["variant"] is None


def test_each_game_becomes_one_json_line(parse):
    two = HEADERS + '\n1. e4 e5 1/2-1/2\n\n' + HEADERS + '\n1. d4 d5 1/2-1/2\n'
    games = parse(two)
    assert len(games) == 2
    assert [g["game_id"] for g in games] == [1, 2]
    assert games[1]["moves"][0]["san"] == "d4"


# --------------------------------------------------------------------------- #
# move records
# --------------------------------------------------------------------------- #

def test_move_numbering_and_colour(parse):
    games = parse(HEADERS + '\n1. e4 e5 2. Nf3 Nc6 *\n')
    moves = games[0]["moves"]
    assert [(m["move_number"], m["color"], m["san"]) for m in moves] == [
        (1, "white", "e4"), (1, "black", "e5"),
        (2, "white", "Nf3"), (2, "black", "Nc6"),
    ]


def test_fen_before_and_after_bracket_the_move(parse):
    games = parse(HEADERS + '\n1. e4 e5 *\n')
    first = games[0]["moves"][0]
    assert first["fen_before"] == chess.STARTING_FEN
    board = chess.Board(first["fen_before"])
    board.push(chess.Move.from_uci(first["uci"]))
    assert board.fen() == first["fen_after"]


def test_every_move_round_trips_through_python_chess(parse):
    """fen_before + uci must reproduce san and fen_after for the whole game."""
    games = parse(HEADERS + '\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *\n')
    for m in games[0]["moves"]:
        board = chess.Board(m["fen_before"])
        move = chess.Move.from_uci(m["uci"])
        assert board.san(move) == m["san"]
        board.push(move)
        assert board.fen() == m["fen_after"]


def test_castling(parse):
    games = parse(HEADERS + '\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O *\n')
    castle = games[0]["moves"][-1]
    assert castle["san"] == "O-O"
    assert castle["uci"] == "e1g1"


def test_promotion(parse):
    pgn = (HEADERS + '[SetUp "1"]\n[FEN "8/P6k/8/8/8/8/8/K7 w - - 0 1"]\n\n1. a8=Q Kg7 *\n')
    promo = parse(pgn)[0]["moves"][0]
    assert promo["san"] == "a8=Q"
    assert promo["uci"] == "a7a8q"


def test_en_passant(parse):
    pgn = (HEADERS + '[SetUp "1"]\n[FEN "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"]\n\n1. exd6 *\n')
    ep = parse(pgn)[0]["moves"][0]
    assert ep["san"] == "exd6"
    assert ep["uci"] == "e5d6"
    assert "3p" not in ep["fen_after"].split()[0]      # the black pawn is gone


# --------------------------------------------------------------------------- #
# annotations
# --------------------------------------------------------------------------- #

def test_comments_are_attached_to_the_move_they_follow(parse):
    games = parse(HEADERS + '\n1. e4 {Best by test.} e5 2. Nf3 *\n')
    moves = games[0]["moves"]
    assert moves[0]["comment"] == "Best by test."
    assert moves[1]["comment"] is None


def test_nags_are_recorded(parse):
    games = parse(HEADERS + '\n1. e4! e5? *\n')
    moves = games[0]["moves"]
    assert chess.pgn.NAG_GOOD_MOVE in moves[0]["nags"]
    assert chess.pgn.NAG_MISTAKE in moves[1]["nags"]


def test_empty_comment_becomes_none_not_empty_string(parse):
    games = parse(HEADERS + '\n1. e4 {   } e5 *\n')
    assert games[0]["moves"][0]["comment"] is None


def test_pgn_eval_tag_is_extracted_in_pawns(parse):
    games = parse(HEADERS + '\n1. e4 {[%eval 0.24]} e5 {[%eval -0.31]} *\n')
    moves = games[0]["moves"]
    assert moves[0]["eval"] == pytest.approx(0.24)
    assert moves[1]["eval"] == pytest.approx(-0.31)


def test_missing_eval_tag_is_none(parse):
    games = parse(HEADERS + '\n1. e4 e5 *\n')
    assert games[0]["moves"][0]["eval"] is None


def test_variations_are_excluded(parse):
    """The parser walks mainline() only; sidelines are dropped."""
    games = parse(HEADERS + '\n1. e4 (1. d4 d5 2. c4) 1... e5 *\n')
    assert [m["san"] for m in games[0]["moves"]] == ["e4", "e5"]


# --------------------------------------------------------------------------- #
# resilience
# --------------------------------------------------------------------------- #

def test_a_raising_read_game_does_not_abort_the_file(parse, monkeypatch):
    """Regression: an exception from read_game() used to kill the whole file
    part-written, silently truncating it."""
    real = chess.pgn.read_game
    calls = {"n": 0}

    def flaky(handle):
        calls["n"] += 1
        if calls["n"] == 1:
            raise IndexError("pop from empty list")
        return real(handle)

    monkeypatch.setattr(chess.pgn, "read_game", flaky)
    games = parse(HEADERS + '\n1. e4 e5 1/2-1/2\n\n' + HEADERS + '\n1. d4 d5 1/2-1/2\n')

    assert calls["n"] > 1                 # it kept reading after the failure
    assert len(games) >= 1                # and still wrote real output


def test_repeated_read_failures_stop_instead_of_looping_forever(parse, monkeypatch):
    monkeypatch.setattr(chess.pgn, "read_game",
                        lambda handle: (_ for _ in ()).throw(ValueError("bad")))
    games = parse(HEADERS + '\n1. e4 e5 *\n')
    assert games == []                    # gave up, did not hang


def test_output_lines_are_independently_valid_json(parse):
    two = HEADERS + '\n1. e4 e5 *\n\n' + HEADERS + '\n1. d4 d5 *\n'
    games = parse(two)
    for g in games:
        json.dumps(g)                     # round-trips
        assert set(g) == {"game_id", "source", "metadata", "moves"}
