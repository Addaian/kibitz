#!/usr/bin/env python3
"""
Score generated commentary against the held-out annotations (README §7.4).

    python3 scripts/evaluate.py --limit 200
    python3 scripts/evaluate.py --limit 200 --baseline     # base model too
    python3 scripts/evaluate.py --limit 200 --out runs/eval.jsonl

Always runs on data/datasets/test.jsonl — games no split of training ever saw.

**Read the numbers with suspicion.** BLEU and ROUGE measure n-gram overlap with
one reference annotation, and there are many good ways to describe a move. They
are useful for tracking a model against itself across checkpoints, and close to
worthless for deciding whether the commentary is any good. For that, read the
output: `scripts/inspect_dataset.py show --split test`, or the side-by-side
panels in kibitz/review.py.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from kibitz import grounding, prompt, tui
from kibitz.dataset import ungrounded_reason
from kibitz.tui import box

ROOT = Path(__file__).parent.parent
TEST_SPLIT = ROOT / "data" / "datasets" / "test.jsonl"


def load_test(limit: int | None, seed: int):
    import random
    rows = [json.loads(l) for l in TEST_SPLIT.open()]
    # File order follows the parallel enrichment and changes between runs; sort
    # so the sample depends only on which moves are in the split.
    rows.sort(key=lambda r: (r["meta"]["uid"], r["meta"]["ply"]))
    if limit and limit < len(rows):
        random.Random(seed).shuffle(rows)
        rows = rows[:limit]
    return rows


def distinct_n(texts, n=2) -> float:
    """Share of distinct n-grams. Collapses toward 0 when a model repeats itself."""
    grams = Counter()
    for t in texts:
        words = t.split()
        grams.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    total = sum(grams.values())
    return len(grams) / total if total else 0.0


def ungrounded_pct(texts) -> float:
    """% of texts making the claims build_dataset drops from training targets."""
    return 100 * sum(bool(ungrounded_reason(t)) for t in texts) / max(1, len(texts))


def score(predictions, references, moves) -> dict:
    import sacrebleu
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(predictions, references)]
    lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    empty = sum(1 for p in predictions if not p.strip())

    return {
        "n": len(predictions),
        "BLEU": sacrebleu.corpus_bleu(predictions, [references]).score,
        "ROUGE-L": 100 * sum(rouge) / len(rouge),
        "distinct-2": distinct_n(predictions),
        "words (pred)": sum(lengths) / len(lengths),
        "words (ref)": sum(ref_lengths) / len(ref_lengths),
        "empty": empty,
        **grounding.summarise([grounding.check(m, p) for m, p in zip(moves, predictions)]),
        "ungrounded talk %": ungrounded_pct(predictions),
    }


def run_model(commentator, rows, out_handle=None) -> list[str]:
    predictions = []
    started = time.time()
    for i, row in enumerate(rows, 1):
        meta, move = row["meta"], row["move"]
        # Ask for the length the annotator actually wrote, so the length and
        # overlap columns compare like with like.
        length = prompt.length_for_target(row["reference"])
        text = commentator.comment(meta, move, length) or ""
        predictions.append(text)
        if out_handle:
            out_handle.write(json.dumps({
                "model": commentator.name, "uid": row["record"]["meta"]["uid"],
                "san": move["san"], "reference": row["reference"], "prediction": text,
                "grounding": grounding.check(move, text),
            }) + "\n")
        if i % 10 == 0 or i == len(rows):
            rate = i / max(time.time() - started, 1e-9)
            sys.stdout.write(f"\r  {commentator.name}: {i}/{len(rows)}  {rate:.1f}/s   ")
            sys.stdout.flush()
    print()
    return predictions


def enriched_lookup(rows):
    """
    The dataset stores rendered messages; generation needs the structured move.
    Re-read data/enriched and index it by (uid, ply) to recover them.
    """
    from collections import defaultdict
    wanted = {(r["meta"]["uid"], r["meta"]["ply"]) for r in rows}
    found = {}
    for path in (ROOT / "data" / "enriched").glob("*.jsonl"):
        source = path.stem.replace("_studies", "")
        for line in path.open():
            game = json.loads(line)
            uid = f"{source}:{game['game_id']}"
            for move in game["moves"]:
                key = (uid, move.get("ply"))
                if key in wanted:
                    found[key] = (game["metadata"], move)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=200, help="test examples to score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", help="base model override")
    ap.add_argument("--adapter", default="adapters", help="adapter directory")
    ap.add_argument("--baseline", action="store_true",
                    help="also score the un-tuned base model for comparison")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature; 0 (greedy) keeps runs reproducible")
    ap.add_argument("--samples", type=int, default=1,
                    help="sample up to N and keep the first with no detectable problem")
    ap.add_argument("--out", type=Path, help="write every generation to this JSONL")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()

    tui.set_color(sys.stdout.isatty() and not args.no_color)
    if not TEST_SPLIT.exists():
        sys.exit(f"{TEST_SPLIT} missing — run: python3 scripts/build_dataset.py")

    records = load_test(args.limit, args.seed)
    lookup = enriched_lookup(records)
    rows = []
    for rec in records:
        key = (rec["meta"]["uid"], rec["meta"]["ply"])
        if key not in lookup:
            continue
        meta, move = lookup[key]
        rows.append({"record": rec, "meta": meta, "move": move,
                     "reference": rec["messages"][1]["content"]})
    if not rows:
        sys.exit("could not match test rows back to data/enriched")
    print(f"scoring {len(rows)} held-out examples\n")

    from kibitz.model import BaselineCommentator, KibitzCommentator
    commentators = [KibitzCommentator(base_model=args.model, adapter_path=args.adapter,
                                      max_tokens=args.max_tokens, temp=args.temp,
                                      samples=args.samples)]
    if args.baseline:
        commentators.append(BaselineCommentator(base_model=args.model,
                                                max_tokens=args.max_tokens, temp=args.temp))

    references = [r["reference"] for r in rows]
    moves = [r["move"] for r in rows]
    # The human references go through the same board check, so the model's
    # rate has a baseline: annotators quote lines too.
    results = {"HUMAN (ref)": {
        **grounding.summarise([grounding.check(m, r) for m, r in zip(moves, references)]),
        "ungrounded talk %": ungrounded_pct(references)}}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    handle = args.out.open("w") if args.out else None
    try:
        for c in commentators:
            results[c.name] = score(run_model(c, rows, handle), references, moves)
    finally:
        if handle:
            handle.close()

    metrics = ["n", "BLEU", "ROUGE-L", "distinct-2", "words (pred)", "words (ref)", "empty",
               "false placement %", "unplayable move %", "ungrounded talk %"]
    content = 26 + 14 * len(results)      # the widest row
    width = content + 4                   # plus "| " and " |"
    header = f"{'metric':<26}" + "".join(f"{name[:12]:>14}" for name in results)
    lines = [header, "-" * content]
    for m in metrics:
        row = f"{m:<26}"
        for name in results:
            v = results[name].get(m, "-")
            row += f"{v:>14.2f}" if isinstance(v, float) else f"{v:>14}"
        lines.append(row)
    print()
    print("\n".join(box(lines, width, title="HELD-OUT TEST", colour=tui.DIM)))
    if args.out:
        print(f"\ngenerations written to {args.out}")
    print("\nBLEU/ROUGE compare against one reference annotation and reward "
          "phrasing, not chess.\nUse them to track checkpoints; read the output to judge quality."
          "\nThe last two rows check claims against the board: % of comments naming a piece"
          "\nthat is not there, or a move no nearby position allows. Lower is better;"
          "\ncompare with HUMAN (ref), since quoted lines also count as unplayable."
          "\n'Ungrounded talk' is the share that cites databases, other games, hindsight or"
          "\nthe reader -- facts no prompt contains. Test references are filtered, so theirs is 0.")


if __name__ == "__main__":
    main()
