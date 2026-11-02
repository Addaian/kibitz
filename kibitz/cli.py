"""
Commentary for every move of a PGN, from the fine-tuned model. README §8–9.

    python3 -m kibitz.cli game.pgn
    python3 -m kibitz.cli game.pgn --format pgn --moves 15-30 > annotated.pgn
    cat game.pgn | python3 -m kibitz.cli -

Each move goes through the same steps as the training data: `parser.game_record`,
`features.enrich_move` at the training depth, then `prompt.build_prompt` with the
`Length:` setting from `length_for_move`. Text output prints each comment as it
is generated; pgn and json print once the game is done.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import chess.pgn

from kibitz import features, parser, prompt


def parse_range(spec: str | None) -> tuple[int, int] | None:
    """'15-30' -> (15, 30); '15' -> (15, 15); None -> None. Full-move numbers."""
    if not spec:
        return None
    first, _, last = spec.partition("-")
    lo, hi = int(first), int(last or first)
    if lo < 1 or hi < lo:
        raise ValueError(f"bad move range {spec!r}")
    return lo, hi


def annotate(game: chess.pgn.Game, commentator, moves=None, analyse_fn=None):
    """
    Yield (meta, move, comment) for each mainline move inside `moves`.

    `move` is the enriched record the prompt was built from. `analyse_fn`
    replaces Stockfish, as in `features.enrich_move`.
    """
    record = parser.game_record(game, 1, "cli")
    meta = record["metadata"]
    for ply, move in enumerate(record["moves"]):
        if moves and not moves[0] <= move["move_number"] <= moves[1]:
            continue
        move["ply"] = ply
        move["previous_moves"] = features.previous_moves(record["moves"], ply)
        features.enrich_move(move, analyse_fn, from_position=meta["from_position"])
        yield meta, move, commentator.comment(meta, move)


def _label(move: dict) -> str:
    dots = "." if move["color"] == "white" else "..."
    return f"{move['move_number']}{dots} {move['san']}"


def _details(move: dict) -> str:
    """The features the comment was conditioned on, for --verbose."""
    parts = [move["move_classification"],
             f"{move['eval_before']:+.2f} -> {move['eval_after']:+.2f}",
             prompt.length_for_move(move)]
    if not move["is_best"] and move.get("best_move_san"):
        parts.append(f"engine: {move['best_move_san']}")
    parts += move["tactical_motifs"]
    return "[" + ", ".join(parts) + "]"


def _features(move: dict) -> dict:
    keep = ("eval_before", "eval_after", "eval_delta", "best_move_san", "is_best",
            "move_classification", "game_phase", "tactical_motifs", "captured",
            "threatens", "left_hanging", "is_sacrifice", "mate_in")
    return {k: move[k] for k in keep} | {"length": prompt.length_for_move(move)}


def write_game(game, results, fmt: str, verbose: bool, out) -> dict | None:
    """
    Emit one annotated game. Text streams as `results` is consumed; pgn and json
    need the whole game. Returns the json object instead of printing it, so a
    file of several games becomes one list.
    """
    if fmt == "text":
        white, black = game.headers.get("White", "?"), game.headers.get("Black", "?")
        print(f"{white} vs. {black}\n", file=out)
        for _, move, comment in results:
            print(f"{_label(move)}  {comment or ''}", file=out)
            if verbose:
                print(f"    {_details(move)}", file=out)
            out.flush()
        print(file=out)
        return None

    results = list(results)
    if fmt == "pgn":
        by_ply = {move["ply"]: comment for _, move, comment in results}
        for ply, node in enumerate(game.mainline()):
            if ply in by_ply:
                node.comment = by_ply[ply] or ""
        print(game.accept(chess.pgn.StringExporter(headers=True, variations=False,
                                                   comments=True)) + "\n", file=out)
        return None

    return {
        "metadata": parser.game_record(game, 1, "cli")["metadata"],
        "moves": [{"move_number": move["move_number"], "color": move["color"],
                   "san": move["san"], "comment": comment}
                  | (_features(move) if verbose else {})
                  for _, move, comment in results],
    }


def read_games(source: str):
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    handle = io.StringIO(text)
    while (game := chess.pgn.read_game(handle)) is not None:
        yield game


def main() -> None:
    ap = argparse.ArgumentParser(description="Commentary for every move of a PGN.")
    ap.add_argument("pgn", help="PGN file, or - for stdin")
    ap.add_argument("--format", "--output-format", dest="format", default="text",
                    choices=("text", "pgn", "json"))
    ap.add_argument("--stockfish", default=features.DEFAULT_STOCKFISH,
                    help="Stockfish binary (default: $STOCKFISH_PATH or Homebrew's)")
    ap.add_argument("--depth", type=int, default=features.DEFAULT_DEPTH,
                    help=f"engine depth (default: {features.DEFAULT_DEPTH}, the training depth)")
    ap.add_argument("--model", default=None,
                    help="adapter directory (default: adapters/)")
    ap.add_argument("--gguf", default=None,
                    help="run through llama.cpp with this GGUF adapter instead of MLX")
    ap.add_argument("--moves", help='only comment on these move numbers, e.g. "15-30"')
    # Greedy decoding collapses to "White is threatening..."; sampling reads like
    # a person but invents more, so keep the first of N with no detectable fault.
    ap.add_argument("--temp", type=float, default=0.8, help="sampling temperature (default: 0.8)")
    ap.add_argument("--samples", type=int, default=4,
                    help="sample up to N per move, keep the first that passes the checks (default: 4)")
    ap.add_argument("--verbose", action="store_true",
                    help="show the features each comment was conditioned on")
    args = ap.parse_args()

    try:
        moves = parse_range(args.moves)
    except ValueError as exc:
        sys.exit(str(exc))
    if not Path(args.stockfish).exists():
        sys.exit(f"Stockfish not found at {args.stockfish} (set --stockfish or $STOCKFISH_PATH)")
    games = list(read_games(args.pgn))
    if not games:
        sys.exit(f"no games found in {args.pgn}")
    for game in games:
        if game.headers.get("Variant") not in features.STANDARD_VARIANTS:
            sys.exit(f"unsupported variant: {game.headers.get('Variant')}")

    from kibitz.model import DEFAULT_ADAPTER, GGUFCommentator, KibitzCommentator
    try:
        if args.gguf:
            commentator = GGUFCommentator(adapter_path=args.gguf, temp=args.temp,
                                          samples=args.samples)
        else:
            commentator = KibitzCommentator(adapter_path=args.model or DEFAULT_ADAPTER,
                                            temp=args.temp, samples=args.samples)
    except FileNotFoundError as exc:
        sys.exit(f"{exc} — train one with: python3 scripts/train.py")
    features._init_worker(args.stockfish, args.depth)

    collected = []
    try:
        for game in games:
            # python-chess stops the mainline at an illegal move rather than raising.
            for error in game.errors:
                print(f"warning: {error}", file=sys.stderr)
            result = write_game(game, annotate(game, commentator, moves),
                                args.format, args.verbose, sys.stdout)
            if result is not None:
                collected.append(result)
    finally:
        # Explicitly: at exit Python waits on the engine's thread before it
        # would run the atexit hook that stops it, and hangs.
        features._close_worker()
    if args.format == "json":
        print(json.dumps(collected, indent=2))


if __name__ == "__main__":
    main()
