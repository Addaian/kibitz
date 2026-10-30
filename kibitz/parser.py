import argparse
import chess
import chess.pgn
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

# here, we use the chess package. This allows us to parse a PGN finely.
'''
A .pgn file contains many games separated by a blank space.
Each .pgn file has the following seven tag structure:
[Event][Site][Date][Round][White][Black][Result]

Followed by the optional:
[Annotator][PlyCount][TimeControl][Time][Termination][Mode][FEN]

This is finally followed by the moves, in order.

Example:
[Event "F/S Return Match"]
[Site "Belgrade, Serbia JUG"]
[Date "1992.11.04"]
[Round "29"]
[White "Fischer, Robert J."]
[Black "Spassky, Boris V."]
[Result "1/2-1/2"]

1.e4 e5 2.Nf3 Nc6 3.Bb5 {This opening is called the Ruy Lopez.} 3...a6
4.Ba4 Nf6 5.O-O Be7 6.Re1 b5 7.Bb3 d6 8.c3 O-O 9.h3 Nb8 10.d4 Nbd7
11.c4 c6 12.cxb5 axb5 13.Nc3 Bb7 14.Bg5 b4 15.Nb1 h6 16.Bh4 c5 17.dxe5
Nxe4 18.Bxe7 Qxe7 19.exd6 Qf6 20.Nbd2 Nxd6 21.Nc4 Nxc4 22.Bxc4 Nb6
23.Ne5 Rae8 24.Bxf7+ Rxf7 25.Nxf7 Rxe1+ 26.Qxe1 Kxf7 27.Qe3 Qg5 28.Qxg5
hxg5 29.b3 Ke6 30.a3 Kd6 31.axb4 cxb4 32.Ra5 Nd5 33.f3 Bc8 34.Kf2 Bf5
35.Ra7 g6 36.Ra6+ Kc5 37.Ke1 Nf4 38.g3 Nxh3 39.Kd2 Kb5 40.Rd6 Kc5 41.Ra6
Nf2 42.g4 Bd3 43.Re6 1/2-1/2
'''

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def study_id(chapter_url: str | None) -> str | None:
    """'https://lichess.org/study/RoBvWqfx/FncZMPtk' -> 'RoBvWqfx'."""
    match = re.search(r"/study/([^/#?]+)", chapter_url or "")
    return match.group(1) if match else None


def game_record(game: chess.pgn.Game, game_id: int, source: str) -> dict:
    '''
    One game as the JSON record the rest of the pipeline reads. The CLI builds
    its records here too, so inference sees exactly what training saw.
    Raises if a mainline move is illegal.
    '''
    board = game.board()
    game_json = {
        "game_id": game_id,
        "source": source,
        "metadata": {
            "white":     game.headers.get("White"),
            "black":     game.headers.get("Black"),
            "result":    game.headers.get("Result"),
            "white_elo": game.headers.get("WhiteElo"),
            "black_elo": game.headers.get("BlackElo"),
            "event":     game.headers.get("Event"),
            "date":      game.headers.get("Date"),
            "eco":       game.headers.get("ECO"),
            "opening":   game.headers.get("Opening"),
            "variant":   game.headers.get("Variant"),
            "study":     study_id(game.headers.get("ChapterURL")),
            # Set up from a FEN (a study position), not the start position.
            "from_position": bool(game.headers.get("FEN")),
        },
        "moves": [],
    }
    for node in game.mainline():
        move = {}
        move["move_number"] = board.fullmove_number
        move["color"]       = "white" if board.turn == chess.WHITE else "black"
        move["san"]         = board.san(node.move)
        move["uci"]         = node.move.uci()
        move["fen_before"]  = board.fen()
        board.push(node.move)
        move["fen_after"]   = board.fen()
        move["comment"]     = node.comment.strip() or None
        move["nags"]        = list(node.nags)
        ev = node.eval()
        move["eval"]        = ev.white().score(mate_score=10000) / 100 if ev else None
        game_json["moves"].append(move)
    return game_json


def parse_pgn(pgnfile: str) -> None:
    '''
    Parses a full .pgn file and writes each game as a JSON line to
    data/processed/<source>.jsonl
    '''
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pgn_source = Path(pgnfile).parts[-2]  # folder name, e.g. "lichess_studies"
    out_path = PROCESSED_DIR / f"{Path(pgnfile).stem}.jsonl"

    game_id = 0
    written = 0
    read_failures = 0

    with open(pgnfile) as pgn, open(out_path, "w") as out_f:
        while True:
            # read_game() itself can raise on malformed PGN. Left unguarded it
            # aborts the whole file part-written, so recover and keep going.
            try:
                game = chess.pgn.read_game(pgn)
            except Exception as e:
                read_failures += 1
                print(f"error reading game after {game_id} in {Path(pgnfile).name}: {e}")
                if read_failures > 50:
                    print(f"  too many read failures in {Path(pgnfile).name}, stopping early")
                    break
                continue
            if game is None:
                break
            read_failures = 0

            game_id += 1
            try:
                out_f.write(json.dumps(game_record(game, game_id, pgn_source)) + "\n")
                written += 1
            except Exception as e:
                print(f"error, skipping game {game_id}: {e}")
                continue

    print(f"parsed {written} games into {out_path}")


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def _parse_one(pgnfile: str) -> tuple[str, str | None]:
    try:
        parse_pgn(pgnfile)
        return Path(pgnfile).name, None
    except Exception as exc:
        return Path(pgnfile).name, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse every PGN under data/raw into data/processed as JSONL.")
    ap.add_argument("--input", type=Path, default=RAW_DIR,
                    help="Directory of source folders, or a single .pgn file.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if args.input.is_file():
        files = [args.input]
    else:
        # parse_pgn derives the source name from the parent folder, so PGNs are
        # expected one directory deep (data/raw/<source>/*.pgn).
        files = sorted(args.input.glob("*/*.pgn"))
    if not files:
        sys.exit(f"no .pgn files found under {args.input}")

    print(f"parsing {len(files)} file(s) on {args.workers} workers")
    started = time.time()
    failures = 0
    with mp.Pool(args.workers) as pool:
        for name, error in pool.imap_unordered(_parse_one, [str(f) for f in files]):
            if error:
                failures += 1
                print(f"  FAILED {name}: {error}")

    print(f"done: {len(files) - failures}/{len(files)} files -> {PROCESSED_DIR} "
          f"in {time.time() - started:.0f}s")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
