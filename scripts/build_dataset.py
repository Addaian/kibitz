#!/usr/bin/env python3
"""
Turn data/enriched/ into train / val / test splits under data/datasets/.

    python3 scripts/build_dataset.py
    python3 scripts/build_dataset.py --min-comment 40 --max-per-class 6000
    python3 scripts/build_dataset.py --strict          # messages only, no provenance

Only the training split is balanced. Validation and test keep the corpus's real
distribution, because metrics measured on a rebalanced set would not describe
how the model behaves on real games.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz.dataset import (build_record, clean_comment, is_usable, merge_duplicate_keys,
                            non_prose_reason, split_for, split_key, ungrounded_reason)

ROOT = Path(__file__).parent.parent
ENRICHED_DIR = ROOT / "data" / "enriched"
DATASETS_DIR = ROOT / "data" / "datasets"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-comment", type=int, default=25,
                    help="minimum cleaned annotation length (default: 25)")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap each move classification in TRAIN only (0 = no cap)")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0, help="seed for balancing")
    ap.add_argument("--strict", action="store_true",
                    help="emit only the messages key, dropping provenance")
    args = ap.parse_args()

    if not ENRICHED_DIR.exists():
        sys.exit(f"{ENRICHED_DIR} does not exist — run: python3 -m kibitz.features")

    splits = defaultdict(list)
    seen = Counter()
    dropped = Counter()
    games = defaultdict(set)
    studies = defaultdict(set)

    # First pass: every usable pair, with the study it belongs to. Splitting
    # waits until all studies are seen, because duplicates join studies together.
    candidates = []
    for path in sorted(ENRICHED_DIR.glob("*.jsonl")):
        source = path.stem.replace("_studies", "")
        for line in path.open():
            game = json.loads(line)
            uid = f"{source}:{game['game_id']}"
            key = split_key(source, game["metadata"], uid)
            for move in game["moves"]:
                seen["total"] += 1
                raw = move.get("comment_text") or ""
                target = clean_comment(raw)
                if target != " ".join(raw.split()):
                    seen["cleaned"] += 1
                if not is_usable(target, args.min_comment):
                    dropped["too short or boilerplate"] += 1
                    continue
                reason = non_prose_reason(target) or ungrounded_reason(target)
                if reason:
                    dropped[reason] += 1
                    continue
                candidates.append((key, uid, source, game["metadata"], move, target))

    group = merge_duplicate_keys((key, f"{move['fen_before']}|{target}")
                                 for key, _, _, _, move, target in candidates)
    for key, uid, source, meta, move, target in candidates:
        split = split_for(group[key], args.val, args.test)
        splits[split].append(build_record(meta, move, source, uid, target))
        games[split].add(uid)
        studies[split].add(group[key])

    # Balance the training split only.
    if args.max_per_class:
        rng = random.Random(args.seed)
        by_class = defaultdict(list)
        for rec in splits["train"]:
            by_class[rec["meta"]["classification"]].append(rec)
        kept = []
        for cls, recs in by_class.items():
            rng.shuffle(recs)
            if len(recs) > args.max_per_class:
                dropped[f"balanced away ({cls})"] += len(recs) - args.max_per_class
            kept.extend(recs[:args.max_per_class])
        kept.sort(key=lambda r: (r["meta"]["source"], r["meta"]["uid"], r["meta"]["ply"] or 0))
        splits["train"] = kept

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    total_written = 0
    print(f"\nread {seen['total']:,} annotated moves "
          f"({seen['cleaned']:,} had engine boilerplate stripped)")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:>6,}  {reason}")

    print()
    for name in ("train", "val", "test"):
        records = splits[name]
        out = DATASETS_DIR / f"{name}.jsonl"
        with out.open("w") as f:
            for rec in records:
                f.write(json.dumps({"messages": rec["messages"]} if args.strict else rec) + "\n")
        total_written += len(records)
        classes = Counter(r["meta"]["classification"] for r in records)
        share = 100 * len(records) / max(1, sum(len(splits[s]) for s in ("train", "val", "test")))
        print(f"{name:<6} {len(records):>7,} pairs  {len(games[name]):>6,} games  "
              f"({share:4.1f}%)  -> {out.relative_to(ROOT)}")
        print(f"       {dict(classes.most_common())}")

    # A study (or a group of studies sharing content) must never appear in two
    # splits, or the test score is fiction.
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                overlap = studies[a] & studies[b]
                assert not overlap, f"{a}/{b} share {len(overlap)} studies"
    print(f"\nwrote {total_written:,} pairs; no study appears in two splits.")


if __name__ == "__main__":
    main()
