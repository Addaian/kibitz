"""
Unit tests for kibitz.cli.

The model is replaced by a commentator that echoes the move, and Stockfish by
frozen scores, so these run anywhere. One test uses the real engine and is
skipped without it.
"""

import io
import json
import sys
from pathlib import Path

import chess.pgn
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import cli, features

PGN = """[Event "Casual"]
[White "Anderssen"]
[Black "Kieseritzky"]
[Opening "King's Gambit"]
[Result "*"]

1. e4 e5 2. f4 { The King's Gambit. } exf4 3. Bc4 Qh4+ *
"""


class Echo:
    name = "ECHO"

    def __init__(self):
        self.seen = []

    def comment(self, meta, move):
        self.seen.append((meta, move))
        return f"comment on {move['san']}"


def frozen(board):
    return 0, None, None


def game():
    return chess.pgn.read_game(io.StringIO(PGN))


def run(fmt, verbose=False, moves=None, commentator=None):
    out = io.StringIO()
    g = game()
    result = cli.write_game(g, cli.annotate(g, commentator or Echo(), moves, frozen),
                            fmt, verbose, out)
    return out.getvalue(), result


# --------------------------------------------------------------------------- #
# move ranges
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spec, expected", [
    (None, None), ("", None), ("15-30", (15, 30)), ("15", (15, 15)),
])
def test_parse_range(spec, expected):
    assert cli.parse_range(spec) == expected

@pytest.mark.parametrize("spec", ["0-3", "10-5", "x", "3-y"])
def test_bad_ranges_are_rejected(spec):
    with pytest.raises(ValueError):
        cli.parse_range(spec)

# --------------------------------------------------------------------------- #
# annotate
# --------------------------------------------------------------------------- #

def test_every_mainline_move_is_commented():
    results = list(cli.annotate(game(), Echo(), analyse_fn=frozen))
    assert [m["san"] for _, m, _ in results] == ["e4", "e5", "f4", "exf4", "Bc4", "Qh4+"]
    assert [m["ply"] for _, m, _ in results] == list(range(6))
    assert all(c == f"comment on {m['san']}" for _, m, c in results)

def test_the_commentator_gets_enriched_moves_and_the_headers():
    echo = Echo()
    list(cli.annotate(game(), echo, analyse_fn=frozen))
    meta, move = echo.seen[0]
    assert meta["opening"] == "King's Gambit" and meta["from_position"] is False
    assert move["move_classification"] and move["game_phase"] == "opening"

def test_a_move_range_selects_full_moves():
    results = list(cli.annotate(game(), Echo(), (2, 2), frozen))
    assert [m["san"] for _, m, _ in results] == ["f4", "exf4"]

# --------------------------------------------------------------------------- #
# output formats
# --------------------------------------------------------------------------- #

def test_text_labels_each_move():
    text, _ = run("text")
    lines = text.splitlines()
    assert lines[0] == "Anderssen vs. Kieseritzky"
    assert "1. e4  comment on e4" in lines
    assert "1... e5  comment on e5" in lines

def test_text_verbose_shows_the_features():
    text, _ = run("text", verbose=True)
    assert "[best, +0.00 -> +0.00, brief]" not in text   # frozen engine has no best move
    assert "[good, +0.00 -> +0.00, brief]" in text

def test_pgn_output_is_valid_and_keeps_the_headers():
    text, _ = run("pgn")
    back = chess.pgn.read_game(io.StringIO(text))
    assert back.headers["White"] == "Anderssen"
    assert [n.comment for n in back.mainline()][:2] == ["comment on e4", "comment on e5"]

def test_pgn_keeps_original_comments_outside_the_range():
    text, _ = run("pgn", moves=(1, 1))
    comments = [n.comment for n in chess.pgn.read_game(io.StringIO(text)).mainline()]
    assert comments[:3] == ["comment on e4", "comment on e5", "The King's Gambit."]

def test_json_is_returned_not_printed():
    text, result = run("json")
    assert text == ""
    json.dumps(result)
    assert result["metadata"]["white"] == "Anderssen"
    assert result["moves"][0] == {"move_number": 1, "color": "white", "san": "e4",
                                  "comment": "comment on e4"}

def test_json_verbose_adds_the_features():
    _, result = run("json", verbose=True)
    first = result["moves"][0]
    assert first["move_classification"] == "good" and first["length"] == "brief"

# --------------------------------------------------------------------------- #
# with the real engine
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not Path(features.DEFAULT_STOCKFISH).exists(), reason="no Stockfish")
def test_real_engine_finds_the_blunder():
    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# *"
    features._init_worker(features.DEFAULT_STOCKFISH, 10)
    try:
        results = list(cli.annotate(chess.pgn.read_game(io.StringIO(pgn)), Echo()))
    finally:
        features._close_worker()   # left running, it hangs the interpreter at exit
    by_san = {m["san"]: m for _, m, _ in results}
    assert by_san["Nf6"]["move_classification"] == "blunder"
    assert by_san["Qxf7#"]["is_best"] and by_san["Qxf7#"]["mate_in"] is None


@pytest.mark.skipif(not Path(features.DEFAULT_STOCKFISH).exists(), reason="no Stockfish")
def test_real_engine_analysis_does_not_depend_on_what_came_before():
    """The CLI analyses a range of moves, the pipeline only annotated ones, the
    pool in any order: a position must get the same features either way."""
    games = ["1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 *",
             "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O dxc4 *"]
    boards = [node.board() for pgn in games
              for node in chess.pgn.read_game(io.StringIO(pgn)).mainline()]
    features._init_worker(features.DEFAULT_STOCKFISH, 14)
    try:
        forward = [features.analyse(b) for b in boards]
        features._cache.clear()
        backward = [features.analyse(b) for b in reversed(boards)][::-1]
    finally:
        features._close_worker()
    assert forward == backward
