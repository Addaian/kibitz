"""
Feature enrichment:  data/processed/*.jsonl  ->  data/enriched/*.jsonl

The parser captures what is *written* in the PGN. This stage computes what must
be *derived*: engine evaluations, move classification, game phase and tactical
motifs.

Only moves carrying a real prose comment are enriched. A comment is the training
target, so a move without one is not a training example, and ~71% of the brace
comments in the Lichess exports are pure [%clk]/[%eval] tags with no prose.
Skipping those cuts the engine work by roughly 25x.

UNITS: every eval field written to disk is in PAWNS (float, 2dp), White's point
of view, EXCEPT `eval_delta`, which is in pawns from the MOVER's point of view
(positive = the move improved the mover's position). Internally everything is
centipawns; the conversion happens once, in `enrich_move`.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import chess
import chess.engine

ROOT = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ENRICHED_DIR = ROOT / "data" / "enriched"

DEFAULT_STOCKFISH = os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")
DEFAULT_DEPTH = 14

# Classification thresholds, in pawns, from the mover's point of view.
BLUNDER_CP = -150
MISTAKE_CP = -50
INACCURACY_CP = -30
SACRIFICE_CP = 100        # material offered before we call it a sacrifice
SACRIFICE_TOLERANCE = -50  # ...and the eval may not drop more than this

# Mate scores are not centipawns. Clamp them so eval arithmetic stays meaningful.
MATE_CLAMP_CP = 2000

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

_TAG_RE = re.compile(r"\[%[^\]]*\]")

# Stockfish is a standard-chess engine. Lichess studies also contain variant
# games, whose evals would be meaningless (and whose comments describe entirely
# different tactics), so those games are skipped wholesale.
STANDARD_VARIANTS = {None, "", "Standard", "From Position"}


class UnanalysablePosition(Exception):
    """A position Stockfish cannot safely be asked to evaluate."""


# --------------------------------------------------------------------------- #
# comment handling
# --------------------------------------------------------------------------- #

def strip_annotations(comment: str | None) -> str:
    """Remove PGN [%eval ...] / [%clk ...] tags, leaving human prose."""
    if not comment:
        return ""
    return _TAG_RE.sub("", comment).strip()


def has_prose(comment: str | None, min_chars: int = 1) -> bool:
    """True if the comment contains human text, not just engine/clock tags."""
    return len(strip_annotations(comment)) >= min_chars


# --------------------------------------------------------------------------- #
# engine
# --------------------------------------------------------------------------- #

_engine: chess.engine.SimpleEngine | None = None
_cache: dict[str, tuple[int, int | None, str | None]] = {}
_depth: int = DEFAULT_DEPTH
_stockfish_path: str = DEFAULT_STOCKFISH


def _cache_key(board: chess.Board) -> str:
    """Position identity without halfmove/fullmove counters, for better hit rates."""
    return board.board_fen() + " " + str(int(board.turn)) + " " + board.castling_xfen() + " " + str(board.ep_square)


def _start_engine() -> None:
    global _engine
    _engine = chess.engine.SimpleEngine.popen_uci(_stockfish_path)
    # One thread each: the pool already saturates the cores.
    _engine.configure({"Threads": 1, "Hash": 64})


def _init_worker(stockfish_path: str, depth: int) -> None:
    global _depth, _stockfish_path
    _depth = depth
    _stockfish_path = stockfish_path
    _start_engine()
    atexit.register(_close_worker)


def _restart_engine() -> None:
    """Replace a dead engine. Stockfish exits on some malformed positions, and
    without this every later game handled by the same worker would fail too."""
    global _engine
    try:
        if _engine is not None:
            _engine.quit()
    except Exception:
        pass
    _engine = None
    _start_engine()


def _close_worker() -> None:
    global _engine
    if _engine is not None:
        try:
            _engine.quit()
        except Exception:
            pass
        _engine = None


def analyse(board: chess.Board) -> tuple[int, int | None, str | None]:
    """
    Return (cp_white, mate_white, best_move_uci) for `board`.

    cp_white  - centipawns, White's point of view, clamped to +/-MATE_CLAMP_CP
    mate_white- signed mate distance from White's POV, or None
    best_uci  - the engine's preferred move in this position, or None
    """
    # Stockfish segfaults on positions that are not legal standard chess -- a
    # kingless pawn diagram, or Chess960 castling rights on standard squares --
    # and the engine cannot be used again afterwards. Never hand it one.
    if not board.is_valid():
        raise UnanalysablePosition(f"illegal position (status {board.status()}): {board.fen()}")

    key = _cache_key(board)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    if board.is_game_over(claim_draw=False):
        result = (0, None, None)
        _cache[key] = result
        return result

    try:
        info = _engine.analyse(board, chess.engine.Limit(depth=_depth))
    except chess.engine.EngineTerminatedError:
        # Defence in depth: recover the worker rather than lose every game after this one.
        _restart_engine()
        info = _engine.analyse(board, chess.engine.Limit(depth=_depth))
    score = info["score"].white()
    mate = score.mate()
    if mate is not None:
        cp = MATE_CLAMP_CP if mate > 0 else -MATE_CLAMP_CP
    else:
        cp = max(-MATE_CLAMP_CP, min(MATE_CLAMP_CP, score.score()))

    pv = info.get("pv")
    best = pv[0].uci() if pv else None

    result = (cp, mate, best)
    _cache[key] = result
    return result


# --------------------------------------------------------------------------- #
# static exchange evaluation (approximate; ignores pins and x-rays)
# --------------------------------------------------------------------------- #

def _least_valuable_attacker(board: chess.Board, square: int, color: bool):
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP,
                       chess.ROOK, chess.QUEEN, chess.KING):
        attackers = board.attackers(color, square) & board.pieces(piece_type, color)
        if attackers:
            return next(iter(attackers))
    return None


def see_gain(board: chess.Board, square: int, color: bool) -> int:
    """
    Material (centipawns) that `color` wins by starting a capture sequence on
    `square`, assuming both sides recapture with their least valuable piece.

    Approximate: attackers that are pinned still participate, and x-ray
    attackers behind a capturer are not revealed.
    """
    victim = board.piece_at(square)
    if victim is None or victim.color == color:
        return 0

    attacker_sq = _least_valuable_attacker(board, square, color)
    if attacker_sq is None:
        return 0

    captured_value = PIECE_VALUE[victim.piece_type]
    board.push(chess.Move(attacker_sq, square))
    try:
        gain = captured_value - see_gain(board, square, not color)
    finally:
        board.pop()

    # The side to move may decline the capture, so the exchange is never negative.
    return max(0, gain)


# --------------------------------------------------------------------------- #
# derived features
# --------------------------------------------------------------------------- #

def detect_phase(board: chess.Board) -> str:
    """opening / middlegame / endgame from move number and remaining material."""
    if board.fullmove_number <= 10:
        return "opening"

    total = 0
    for color in chess.COLORS:
        total += len(board.pieces(chess.QUEEN, color)) * 9
        total += len(board.pieces(chess.ROOK, color)) * 5
        total += len(board.pieces(chess.BISHOP, color)) * 3
        total += len(board.pieces(chess.KNIGHT, color)) * 3

    return "middlegame" if total > 20 else "endgame"


def classify_move(delta_cp: int, is_best: bool) -> str:
    """
    Label a move from its eval swing (mover's POV, centipawns).

    This is deliberately a single axis: how good the move was. Properties that
    are orthogonal to quality stay as their own fields — `is_sacrifice`,
    `is_forcing`, `tactical_motifs`, `mate_in` — because a move can be a
    sacrifice *and* the best move on the board, and collapsing those into one
    enum throws the second fact away.

    Two labels are deliberately absent:

    * "brilliant" — a chess.com construct with no counterpart in Lichess
      studies. It means "hard to spot", a property of the player rather than
      the position, which an eval delta cannot measure.
    * "sacrifice" — real and objective, but a statement about material, not
      about quality. It lives in `is_sacrifice` / `material_offered`.
    """
    if delta_cp <= BLUNDER_CP:
        return "blunder"
    if delta_cp <= MISTAKE_CP:
        return "mistake"
    if delta_cp <= INACCURACY_CP:
        return "inaccuracy"
    if is_best:
        return "best"
    return "good"


def _is_valuable_target(target: chess.Piece, attacker: chess.Piece) -> bool:
    """A target worth forking/attacking: the king, or a piece worth at least as much."""
    if target.piece_type == chess.KING:
        return True
    return PIECE_VALUE[target.piece_type] >= PIECE_VALUE[attacker.piece_type]


def detect_fork(board_after: chess.Board, move: chess.Move) -> bool:
    """The moved piece now attacks two or more valuable enemy pieces."""
    piece = board_after.piece_at(move.to_square)
    if piece is None:
        return False

    hits = 0
    for square in board_after.attacks(move.to_square):
        target = board_after.piece_at(square)
        if target and target.color != piece.color and _is_valuable_target(target, piece):
            hits += 1
            if hits >= 2:
                return True
    return False


def detect_pin(board_before: chess.Board, board_after: chess.Board, mover: bool) -> bool:
    """An enemy piece is pinned to its king now, and was not before the move."""
    enemy = not mover
    for square in chess.SquareSet(board_after.occupied_co[enemy]):
        if not board_after.is_pinned(enemy, square):
            continue
        # Same piece must have been there, unpinned, before the move.
        prior = board_before.piece_at(square)
        if prior and prior.color == enemy and not board_before.is_pinned(enemy, square):
            return True
    return False


def _walk_ray(board: chess.Board, origin: int, through: int) -> int | None:
    """First occupied square beyond `through`, continuing away from `origin`."""
    file_step = chess.square_file(through) - chess.square_file(origin)
    rank_step = chess.square_rank(through) - chess.square_rank(origin)
    file_step = (file_step > 0) - (file_step < 0)
    rank_step = (rank_step > 0) - (rank_step < 0)
    if file_step == 0 and rank_step == 0:
        return None

    file, rank = chess.square_file(through), chess.square_rank(through)
    while True:
        file += file_step
        rank += rank_step
        if not (0 <= file <= 7 and 0 <= rank <= 7):
            return None
        square = chess.square(file, rank)
        if board.piece_at(square):
            return square


def detect_skewer(board_after: chess.Board, move: chess.Move) -> bool:
    """A slider attacks a valuable enemy piece with a lesser enemy piece behind it."""
    piece = board_after.piece_at(move.to_square)
    if piece is None or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False

    for square in board_after.attacks(move.to_square):
        front = board_after.piece_at(square)
        if not front or front.color == piece.color:
            continue

        behind_sq = _walk_ray(board_after, move.to_square, square)
        if behind_sq is None:
            continue
        behind = board_after.piece_at(behind_sq)
        if not behind or behind.color == piece.color:
            continue

        front_value = PIECE_VALUE[front.piece_type]
        if front_value > PIECE_VALUE[behind.piece_type] and front_value > PIECE_VALUE[piece.piece_type]:
            return True
    return False


def detect_discovery(board_after: chess.Board, move: chess.Move) -> bool:
    """The moved piece vacated a square that was blocking a friendly slider's attack."""
    piece = board_after.piece_at(move.to_square)
    if piece is None:
        return False
    color = piece.color

    sliders = (board_after.pieces(chess.BISHOP, color)
               | board_after.pieces(chess.ROOK, color)
               | board_after.pieces(chess.QUEEN, color))

    for slider_sq in sliders:
        if slider_sq == move.to_square:
            continue
        slider = board_after.piece_at(slider_sq)
        for target_sq in board_after.attacks(slider_sq):
            target = board_after.piece_at(target_sq)
            if not target or target.color == color:
                continue
            if not _is_valuable_target(target, slider):
                continue
            if chess.between(slider_sq, target_sq) & chess.BB_SQUARES[move.from_square]:
                return True
    return False


def detect_promotion_threat(board_after: chess.Board, color: bool) -> bool:
    """A friendly pawn stands one square from promoting."""
    rank = 6 if color == chess.WHITE else 1
    return any(chess.square_rank(sq) == rank
               for sq in board_after.pieces(chess.PAWN, color))


def detect_motifs(board_before: chess.Board, board_after: chess.Board,
                  move: chess.Move) -> list[str]:
    """Geometric tactical motifs created by `move`. Heuristic, not exhaustive."""
    mover = board_before.turn
    motifs = []

    if detect_fork(board_after, move):
        motifs.append("fork")
    if detect_pin(board_before, board_after, mover):
        motifs.append("pin")
    if detect_skewer(board_after, move):
        motifs.append("skewer")
    if detect_discovery(board_after, move):
        motifs.append("discovery")
    if move.promotion:
        motifs.append("promotion")
    elif detect_promotion_threat(board_after, mover):
        motifs.append("promotion_threat")

    return motifs


# --------------------------------------------------------------------------- #
# per-move enrichment
# --------------------------------------------------------------------------- #

def enrich_move(move: dict, analyse_fn=None) -> dict:
    """
    Add derived features to one parsed move record, in place.

    `analyse_fn` defaults to the engine. Passing a function that returns frozen
    scores replays the whole judgement layer without Stockfish, which is how
    --relabel and the golden tests work.
    """
    analyse_fn = analyse_fn or analyse
    board_before = chess.Board(move["fen_before"])
    board_after = chess.Board(move["fen_after"])
    mover = board_before.turn
    uci = chess.Move.from_uci(move["uci"])

    cp_before, mate_before, best_uci = analyse_fn(board_before)
    cp_after, mate_after, _ = analyse_fn(board_after)

    # White-POV centipawns -> mover-POV swing.
    delta_cp = (cp_after - cp_before) if mover == chess.WHITE else (cp_before - cp_after)

    # Material the opponent can win back on the destination square, net of
    # whatever this move captured.
    captured = board_before.piece_at(uci.to_square)
    if captured is None and board_before.is_en_passant(uci):
        captured_value = PIECE_VALUE[chess.PAWN]
    else:
        captured_value = PIECE_VALUE[captured.piece_type] if captured else 0

    offered = see_gain(board_after.copy(stack=False), uci.to_square, not mover) - captured_value
    is_sacrifice = offered >= SACRIFICE_CP and delta_cp >= SACRIFICE_TOLERANCE
    is_best = best_uci is not None and best_uci == move["uci"]

    move["eval_before"] = round(cp_before / 100, 2)
    move["eval_after"] = round(cp_after / 100, 2)
    move["eval_delta"] = round(delta_cp / 100, 2)
    move["best_move"] = best_uci
    move["is_best"] = is_best
    move["is_forcing"] = board_before.gives_check(uci) or board_before.is_capture(uci)
    move["is_sacrifice"] = is_sacrifice
    move["material_offered"] = round(offered / 100, 2)
    move["mate_in"] = mate_after
    move["is_mate_threat"] = mate_after is not None and (
        (mate_after > 0) if mover == chess.WHITE else (mate_after < 0)
    )
    move["game_phase"] = detect_phase(board_before)
    move["move_classification"] = classify_move(delta_cp, is_best)
    move["tactical_motifs"] = detect_motifs(board_before, board_after, uci)
    move["comment_text"] = strip_annotations(move.get("comment"))

    # The parser wrote the PGN's own %eval here, in pawns. It is superseded by
    # eval_before/eval_after, so drop it rather than leave two meanings on one key.
    move.pop("eval", None)
    return move


def enrich_game(payload: tuple[str, int]) -> dict:
    """
    Worker unit. Takes (raw json line, min_comment_chars) and returns a report:
      line     - the enriched game as JSON, or None if nothing was kept
      enriched - moves enriched
      skipped  - moves with no prose comment
      invalid  - annotated moves whose position Stockfish cannot evaluate
      variant  - the non-standard variant this game was skipped for, if any
      error    - unexpected failure, as a string
    """
    line, min_chars = payload
    blank = {"line": None, "enriched": 0, "skipped": 0, "invalid": 0,
             "variant": None, "error": None}
    try:
        game = json.loads(line)
        total_plies = len(game["moves"])

        variant = (game.get("metadata") or {}).get("variant")
        if variant not in STANDARD_VARIANTS:
            return {**blank, "skipped": total_plies, "variant": variant}

        keep, invalid = [], 0
        for index, move in enumerate(game["moves"]):
            if not has_prose(move.get("comment"), min_chars):
                continue
            move["ply"] = index
            try:
                keep.append(enrich_move(move))
            except (UnanalysablePosition, ValueError):
                # Malformed or non-standard position (Crazyhouse FEN, kingless
                # diagram, ...). Drop the move, keep the rest of the game.
                invalid += 1

        game["moves"] = keep
        game["total_plies"] = total_plies
        return {"line": json.dumps(game) if keep else None,
                "enriched": len(keep),
                "skipped": total_plies - len(keep) - invalid,
                "invalid": invalid,
                "variant": None,
                "error": None}
    except Exception as exc:
        return {**blank, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def enrich_file(path: Path, pool: mp.pool.Pool, min_chars: int) -> dict:
    """Enrich one processed JSONL file. Writes atomically via a .tmp sibling."""
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENRICHED_DIR / path.name
    tmp_path = out_path.with_suffix(".jsonl.tmp")

    with open(path) as f:
        lines = [(line, min_chars) for line in f if line.strip()]

    stats = {"games": 0, "enriched": 0, "invalid": 0, "variants": 0, "errors": 0}
    first_errors: dict[str, str] = {}
    start = time.time()

    with open(tmp_path, "w") as out:
        for i, r in enumerate(pool.imap_unordered(enrich_game, lines, chunksize=1), 1):
            if r["error"]:
                stats["errors"] += 1
                first_errors.setdefault(r["error"].split(":")[0], r["error"])
            if r["variant"]:
                stats["variants"] += 1
            stats["invalid"] += r["invalid"]
            if r["line"]:
                out.write(r["line"] + "\n")
                stats["games"] += 1
                stats["enriched"] += r["enriched"]

            if i % 50 == 0 or i == len(lines):
                rate = i / max(time.time() - start, 1e-9)
                sys.stdout.write(
                    f"\r  {path.name}: {i}/{len(lines)} games  "
                    f"{stats['enriched']} moves  {rate:.0f} games/s  "
                    f"{stats['errors']} errors   "
                )
                sys.stdout.flush()

    tmp_path.replace(out_path)
    print(f"\r  {path.name}: {stats['games']} games, {stats['enriched']} moves, "
          f"{stats['invalid']} unanalysable, {stats['variants']} variant games, "
          f"{stats['errors']} errors, {time.time() - start:.0f}s" + " " * 15)
    # Surfacing these matters: a silently swallowed error once cost a whole run.
    for kind, message in first_errors.items():
        print(f"      first {kind}: {message[:160]}")
    return stats


def relabel_file(path: Path) -> collections.Counter:
    """
    Re-derive every judgement field from the engine scores already on disk.

    Classification, phase, motifs and the sacrifice test are pure functions of
    the position plus the stored evals, so changing any of them costs a pass
    over the JSONL rather than another run of Stockfish.
    """
    tmp = path.with_suffix(".jsonl.tmp")
    changed = collections.Counter()

    with open(path) as f, open(tmp, "w") as out:
        for line in f:
            game = json.loads(line)
            for move in game["moves"]:
                before_cp = round(move["eval_before"] * 100)
                after_cp = round(move["eval_after"] * 100)
                best_uci, mate = move["best_move"], move["mate_in"]
                calls = []

                def frozen(board):
                    calls.append(1)
                    if len(calls) == 1:
                        return before_cp, None, best_uci
                    return after_cp, mate, None

                was = move["move_classification"]
                enrich_move(move, analyse_fn=frozen)
                now = move["move_classification"]
                if now != was:
                    changed[f"{was} -> {now}"] += 1
            out.write(json.dumps(game) + "\n")

    tmp.replace(path)
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description="Enrich parsed games with engine and tactical features.")
    ap.add_argument("--input", type=Path, default=PROCESSED_DIR,
                    help="Processed JSONL file or directory (default: data/processed).")
    ap.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                    help=f"Stockfish search depth (default: {DEFAULT_DEPTH}).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                    help="Parallel engine processes.")
    ap.add_argument("--stockfish", default=DEFAULT_STOCKFISH, help="Path to the Stockfish binary.")
    ap.add_argument("--min-comment-chars", type=int, default=1,
                    help="Skip moves whose prose comment is shorter than this.")
    ap.add_argument("--force", action="store_true", help="Re-enrich files that already have output.")
    ap.add_argument("--relabel", action="store_true",
                    help="Re-derive labels from existing data/enriched output, no engine.")
    args = ap.parse_args()

    if args.relabel:
        files = sorted(ENRICHED_DIR.glob("*.jsonl"))
        if not files:
            sys.exit(f"nothing to relabel in {ENRICHED_DIR}")
        changed = collections.Counter()
        started = time.time()
        for path in files:
            changed.update(relabel_file(path))
        total = sum(changed.values())
        print(f"relabelled {len(files)} files in {time.time() - started:.0f}s; "
              f"{total:,} moves changed class")
        for transition, n in changed.most_common():
            print(f"  {n:>7,}  {transition}")
        return

    if not Path(args.stockfish).exists():
        sys.exit(f"Stockfish not found at {args.stockfish} (set --stockfish or $STOCKFISH_PATH)")

    files = sorted(args.input.glob("*.jsonl")) if args.input.is_dir() else [args.input]
    if not files:
        sys.exit(f"No .jsonl files found in {args.input}")

    if not args.force:
        pending = [f for f in files if not (ENRICHED_DIR / f.name).exists()]
        if len(pending) < len(files):
            print(f"skipping {len(files) - len(pending)} already-enriched file(s); --force to redo")
        files = pending

    if not files:
        print("nothing to do.")
        return

    print(f"enriching {len(files)} file(s) at depth {args.depth} on {args.workers} workers")
    totals = collections.Counter()
    started = time.time()

    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.stockfish, args.depth)) as pool:
        for path in files:
            totals.update(enrich_file(path, pool, args.min_comment_chars))

    print(f"\ndone: {totals['games']} games, {totals['enriched']} enriched moves "
          f"({totals['invalid']} unanalysable, {totals['variants']} variant games, "
          f"{totals['errors']} errors) -> {ENRICHED_DIR} "
          f"in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
