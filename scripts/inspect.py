#!/usr/bin/env python3
"""
Review harness for the kibitz pipeline.

    inspect.py show                         a random annotated move
    inspect.py show -n 5 --class blunder    five blunders
    inspect.py show --motif fork --phase endgame
    inspect.py show --source AbasovN --seed 7 --unicode
    inspect.py stats                        corpus dashboard
    inspect.py freeze                       refresh the golden regression set

`show` renders a position beside everything the pipeline decided about it, then
one panel per source of commentary. Today that is the human annotator; in phase
2 a model joins the same layout (see kibitz/review.py).
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import review, tui
from kibitz.tui import box, sparkbar

# NB: read R / BOLD / DIM through the module — tui.set_color() rebinds them.

ROOT = Path(__file__).parent.parent
ENRICHED_DIR = ROOT / "data" / "enriched"
GOLDEN_PATH = ROOT / "tests" / "fixtures" / "golden.jsonl"


def iter_moves(sources=None):
    """Yield (metadata, move, source_label, game_uid) across the enriched corpus."""
    files = sorted(ENRICHED_DIR.glob("*.jsonl"))
    if sources:
        files = [f for f in files if any(s.lower() in f.name.lower() for s in sources)]
    for path in files:
        label = path.stem.replace("_studies", "")
        for line in path.open():
            game = json.loads(line)
            uid = f"{label}:{game['game_id']}"
            for move in game["moves"]:
                yield game["metadata"], move, label, uid


def matches(move, args):
    if getattr(args, "classification", None) and move["move_classification"] not in args.classification:
        return False
    if getattr(args, "motif", None) and not set(args.motif) & set(move["tactical_motifs"]):
        return False
    if getattr(args, "phase", None) and move["game_phase"] not in args.phase:
        return False
    if len(move["comment_text"]) < getattr(args, "min_comment", 0):
        return False
    return True


# --------------------------------------------------------------------------- #
# show
# --------------------------------------------------------------------------- #

def cmd_show(args):
    rng = random.Random(args.seed)
    picked, seen = [], 0
    # Reservoir sample so the whole corpus never has to be held in memory.
    for meta, move, label, _uid in iter_moves(args.source):
        if not matches(move, args):
            continue
        seen += 1
        if len(picked) < args.n:
            picked.append((meta, move, label))
        elif rng.random() < args.n / seen:
            picked[rng.randrange(args.n)] = (meta, move, label)

    if not picked:
        sys.exit("nothing matched those filters.")

    commentators = [review.HumanAnnotation()]
    for meta, move, label in picked:
        for line in review.render_move(meta, move, label, commentators,
                                       unicode_pieces=args.unicode):
            print(line)
    print(f"{tui.DIM}  showing {len(picked)} of {seen:,} matching moves{tui.R}\n")


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def cmd_stats(args):
    cls, phase, motif, src = Counter(), Counter(), Counter(), Counter()
    colour = defaultdict(Counter)
    total = games = 0
    last_uid = None
    deltas, lengths = [], []

    for meta, move, label, uid in iter_moves(args.source):
        if uid != last_uid:            # moves arrive grouped by game
            games += 1
            last_uid = uid
        total += 1
        cls[move["move_classification"]] += 1
        phase[move["game_phase"]] += 1
        src[label] += 1
        colour[move["color"]][move["move_classification"]] += 1
        for m in move["tactical_motifs"]:
            motif[m] += 1
        deltas.append(move["eval_delta"])
        lengths.append(len(move["comment_text"]))

    if not total:
        sys.exit("no enriched data found — run: python3 -m kibitz.features")

    W, BAR = 66, 26

    def dist(title, counter, tints=None, label_w=18):
        top = counter.most_common(1)[0][1]
        rows = [f"{(tints or {}).get(k, '')}{k:<{label_w}}{tui.R}{n:>7,}  {sparkbar(n, top, BAR)}"
                for k, n in counter.most_common()]
        return box(rows, W, title=title, colour=tui.DIM)

    print()
    for line in box([f"{tui.BOLD}kibitz enriched corpus{tui.R}",
                     f"{tui.DIM}{total:,} annotated moves | {games:,} games | "
                     f"{len(src)} study collections{tui.R}"], W, colour=tui.DIM):
        print(line)
    print()
    for section in (dist("MOVE CLASSIFICATION", cls, tui.CLASS_COLOR),
                    dist("GAME PHASE", phase),
                    dist("TACTICAL MOTIFS", motif)):
        print("\n".join(section) + "\n")

    deltas.sort()
    lengths.sort()
    q = lambda xs, pct: xs[int(len(xs) * pct)]
    checks = []
    for side in ("white", "black"):
        n = sum(colour[side].values())
        if n:
            checks.append(f"{side:<10} blunder rate "
                          f"{100 * colour[side]['blunder'] / n:5.2f}%   {tui.DIM}{n:,} moves{tui.R}")
    checks += ["", f"{tui.DIM}eval delta (pawns, mover's view){tui.R}",
               f"  p05 {q(deltas,.05):+.2f}    median {q(deltas,.5):+.2f}    p95 {q(deltas,.95):+.2f}",
               "", f"{tui.DIM}comment length (characters){tui.R}",
               f"  median {q(lengths,.5)}      p90 {q(lengths,.9)}      max {lengths[-1]}"]
    print("\n".join(box(checks, W, title="SANITY CHECKS", colour=tui.DIM)) + "\n")

    top = src.most_common(1)[0][1]
    rows = [f"{k:<26}{n:>7,}  {sparkbar(n, top, BAR)}" for k, n in src.most_common(10)]
    print("\n".join(box(rows, W, title="TOP SOURCES", colour=tui.DIM)) + "\n")


# --------------------------------------------------------------------------- #
# freeze
# --------------------------------------------------------------------------- #

CLASSES = ["best", "good", "inaccuracy", "mistake", "blunder", "sacrifice", "brilliant"]
MOTIFS = ["fork", "pin", "skewer", "discovery", "promotion", "promotion_threat"]


def cmd_freeze(args):
    """
    Freeze a small, diverse set of enriched moves as a regression fixture.

    The engine numbers are frozen with the record, so tests/test_golden.py can
    replay the pure classification and motif logic without needing Stockfish.
    """
    buckets = defaultdict(list)
    for meta, move, label, uid in iter_moves():
        for key in (f"class:{move['move_classification']}",
                    *(f"motif:{m}" for m in move["tactical_motifs"]),
                    *(["flag:mate"] if move.get("mate_in") is not None else []),
                    *(["flag:sacrifice"] if move["is_sacrifice"] else [])):
            if len(buckets[key]) < args.per_bucket:
                buckets[key].append({
                    "source": label, "uid": uid, "ply": move["ply"],
                    "meta": {k: meta.get(k) for k in
                             ("white", "black", "event", "date", "eco", "opening", "variant")},
                    "move": move,
                })

    seen, records = set(), []
    for key in sorted(buckets):
        for rec in buckets[key]:
            ident = (rec["uid"], rec["ply"])
            if ident not in seen:
                seen.add(ident)
                records.append(rec)
    records.sort(key=lambda r: (r["source"], r["uid"], r["ply"]))

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    covered = Counter(r["move"]["move_classification"] for r in records)
    print(f"wrote {len(records)} records -> {GOLDEN_PATH.relative_to(ROOT)}")
    print("  classifications:", dict(covered.most_common()))
    missing = [c for c in CLASSES if c not in covered]
    if missing:
        print(f"  {tui.DIM}not represented: {', '.join(missing)}{tui.R}")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-color", action="store_true", help="plain text output")
    sub = ap.add_subparsers(dest="command")

    def add_filters(p):
        p.add_argument("--class", dest="classification", nargs="+", choices=CLASSES)
        p.add_argument("--motif", nargs="+", choices=MOTIFS)
        p.add_argument("--phase", nargs="+", choices=["opening", "middlegame", "endgame"])
        p.add_argument("--source", nargs="+", help="only these study files")

    s = sub.add_parser("show", help="render annotated moves")
    add_filters(s)
    s.add_argument("-n", type=int, default=1, help="how many moves to show")
    s.add_argument("--min-comment", type=int, default=40, help="skip very short comments")
    s.add_argument("--seed", type=int, help="reproducible sampling")
    s.add_argument("--unicode", action="store_true", help="chess glyphs instead of letters")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("stats", help="corpus dashboard")
    s.add_argument("--source", nargs="+")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("freeze", help="refresh the golden regression fixture")
    s.add_argument("--per-bucket", type=int, default=4)
    s.set_defaults(func=cmd_freeze)

    # --no-color is accepted before or after the subcommand, so pull it out of
    # argv by hand rather than declaring it in two places.
    argv = [a for a in sys.argv[1:] if a != "--no-color"]
    no_color = len(argv) != len(sys.argv[1:])

    # Bare filters still work: "inspect.py --class blunder" means "show".
    commands = {"show", "stats", "freeze"}
    if not ({"-h", "--help"} & set(argv)) and not (commands & set(argv)):
        argv = ["show"] + argv

    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.error("pick a command: show, stats or freeze")

    tui.set_color(sys.stdout.isatty() and not no_color)
    if not ENRICHED_DIR.exists():
        sys.exit(f"{ENRICHED_DIR} does not exist — run: python3 -m kibitz.features")
    args.func(args)


if __name__ == "__main__":
    main()
