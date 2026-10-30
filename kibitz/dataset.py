"""
Cleaning and shaping enriched moves into training pairs.

`features.py` adds chess knowledge; this stage adds none. It only decides which
annotations are worth training on, tidies their text, and splits them.

Two things here matter more than they look:

* **Lichess auto-analysis is not human commentary.** Roughly one annotated move
  in ten carries machine text like "Inaccuracy. Best move was Qb6." Training on
  it teaches the model to emit engine boilerplate. It is stripped rather than
  dropped, because a minority of those comments have real prose behind the
  machine prefix.

* **Splitting is by study, not by move or game.** Consecutive moves of one game
  are highly correlated, and so are the chapters of one study, so a finer split
  would leak the test set into training. Studies that share a (position,
  comment) pair — the same game republished — are grouped first, and the split
  is a stable hash of the group. Re-running never reshuffles; adding studies
  reshuffles only studies whose group it merges.
"""

from __future__ import annotations

import hashlib
import re

# "Inaccuracy. Best move was Qb6." / "Blunder. Kxe3 was best." /
# "Lost forced checkmate sequence. Qf5+ was best."
_AUTO = re.compile(
    r'^\s*(?:Inaccuracy|Mistake|Blunder|Lost forced checkmate sequence|'
    r'Not the best checkmate sequence)\.\s*'
    r'(?:Best move was\s+[^.]{1,12}\.|[^.\s]{1,12}\s+was best\.)\s*', re.I)
# Trailing eval swing, "(0.00 → -3.21)", "(#3 -> #1)" or "(3.12 → Mate in 12)"
_EVAL_SIDE = r'(?:[-+]?\d+\.?\d*|#-?\d+|Mate in \d+)'
_EVAL_TAIL = re.compile(rf'\s*\({_EVAL_SIDE}\s*(?:→|->|-->)\s*{_EVAL_SIDE}\)\s*')
_STOCK_PHRASE = re.compile(r'\s*(?:Checkmate is now unavoidable|Lost forever)\.?\s*', re.I)
_RESULT = re.compile(r'\(?\s*(?:1-0|0-1|1/2-1/2|½-½)\s*\)?')
# Study-puzzle scaffolding. The verdict word ("Correct.") answers a reader who
# just tried the move, and a speaker tag ("AL:") names a co-annotator; the prose
# after either is often good. "Play ...Kg7 for Black." tells the reader to click.
_FEEDBACK = re.compile(r'^(?:correct|incorrect|right|wrong|good job|well done|exactly|yes|no|nope'
                       r'|excellent|great|perfect|bingo)\s*[!.]+\s*', re.I)
_SPEAKER = re.compile(r'^[A-Z]{1,4}:\s+')
# "Black resigned." is the result, which the prompt never shows; the reason that
# often follows ("Mate is coming in two moves.") is the useful part.
_RESIGNED = re.compile(r'[^.!?]*\bresign(?:s|ed)\b[^.!?]*[.!?]*\s*', re.I)
_PLAY_FOR = re.compile(r'\s*(?:Now\s+)?play (?:the move indicated|the indicated move|\S+) '
                       r'for (?:black|white)\b[.!]?', re.I)


def clean_comment(text: str | None) -> str:
    """Strip engine boilerplate and collapse whitespace, keeping any human prose."""
    text = " ".join((text or "").split())
    previous = None
    while previous != text:          # the boilerplate is often repeated verbatim
        previous = text
        text = _AUTO.sub("", text, count=1)
        text = _EVAL_TAIL.sub(" ", text)
        text = _STOCK_PHRASE.sub(" ", text)
        text = _URL.sub(" ", text)       # links to studies and videos; not commentary
        text = _FEEDBACK.sub("", text, count=1)
        text = _SPEAKER.sub("", text, count=1)
        text = _PLAY_FOR.sub(" ", text)
        text = _RESIGNED.sub(" ", text)
        text = " ".join(text.split())
    return text


def is_usable(text: str, min_chars: int) -> bool:
    """
    A target worth training on.

    The game result is removed before the length is measured, so the bar applies
    to actual analysis. Otherwise "Checkmate! (1-0)" and "Stalemate! (1/2-1/2)"
    coast past on punctuation while saying nothing about the position.
    """
    analysis = _RESULT.sub("", text).strip()
    if len(analysis) < min_chars:
        return False
    if not re.search(r'[a-zA-Z]', analysis):
        return False
    return True


# Annotations that pass the length bar but are not commentary a model should
# learn to write: engine logs pasted by hand, bare ECO labels, and prose in
# another language (about 7% of the targets that pass the length bar).
_ENGINE_LOG = re.compile(r'\bd=\d+\b.*\bpv=|\bdepth \d+:|\(\d+ ply\)|\bSF:\s*[-+]?\d', re.I)
_OPENING_LABEL = re.compile(r'^\s*(?:aka\s+|T:\s*)?[A-E]\d\d\b')
_NOTATION = re.compile(r'\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)')
_URL = re.compile(r'https?://\S+')
_ENGLISH = set("""
    a an the and or but if then so as of to in on at by for with from into out up
    down over under after before about against between through is are was were be
    been being am it its this that these those he she they his her their him them
    we our us you your i my me not no nor only also just very more most less than
    too still again even well now here there when where why how what which who
    can could would should will shall may might must has have had do does did
    all any both each other such own same some because while instead already
    white black move moves piece pieces pawn pawns king queen rook bishop knight
""".split())


def non_prose_reason(text: str) -> str | None:
    """Why `text` is not English commentary, or None if it is."""
    if _ENGINE_LOG.search(text):
        return "engine log"
    if _OPENING_LABEL.match(text):
        return "opening label"
    words = re.findall(r"[^\W\d_]+", _NOTATION.sub(" ", _URL.sub(" ", text)).lower())
    if not words:
        return "not English prose"
    if len(words) >= 4 and sum(w in _ENGLISH for w in words) / len(words) < 0.1:
        return "not English prose"
    return None


# Commentary that depends on something the prompt never shows the model.
# Training on it teaches the model to invent such facts: generated comments
# quoted win rates and move popularity 3.5x as often as the training targets.
# Each rule was kept only if hand-labelling showed it almost always fires on a
# comment worth dropping (runs/target_audit.csv; README §4.3.2).
_UNGROUNDED = (
    ("database statistics", re.compile(
        r'win ?rates?|most (?:popular|common|played)|\b\d+(?:\.\d+)? ?%|percent|database'
        r'|statistic|playrates?|\bLMDB\b|opening explorer', re.I)),
    ("addresses the reader", re.compile(
        r'\b(?:arrows?|click|chapters?|this study|video|subscribe|comment below'
        r'|can you find|find the (?:best|winning)|in this (?:lesson|course|study))\b'
        r'|\b(?:can|did|do|could|would) you\b[^.!]*\?'
        # an unanswered quiz question at the end; a rhetorical "Why ...?" is commentary
        r'|\b(?:what|which|how|can|could|does|did|should|is there)\b[^.!?]*\?\W{0,3}$', re.I)),
    ("other games", re.compile(
        r'\balready seen\b|\b[Gg]ame \d+\b|\b[Mm]ove \d+\b'
        r'|\b[A-Z][\w.]+ (?:vs\.?|v\.?) [A-Z]|\bwas played in\b')),
    ("hindsight", re.compile(
        r'\bwent on to\b|\brest of the game\b|\b(?:won|lost) the game\b', re.I)),
)


def ungrounded_reason(text: str) -> str | None:
    """Why `text` relies on facts outside the prompt, or None if it does not."""
    for reason, pattern in _UNGROUNDED:
        if pattern.search(text):
            return reason
    return None


def split_key(source: str, meta: dict, uid: str) -> str:
    """
    The unit that must not straddle splits: the Lichess study.

    Chapters of one study share an author, a theme and often whole passages of
    text, so a game-level split puts near-duplicates of test targets in train.
    Games without a study id fall back to their own uid.
    """
    study = meta.get("study")
    return f"{source}:study:{study}" if study else uid


def merge_duplicate_keys(pairs) -> dict[str, str]:
    """
    Map each split key to a group key, so keys that share content split together.

    `pairs` is an iterable of (split_key, signature), where the signature is
    the position plus the target text. The same game published in two studies
    (by one author, or copied by another) shares signatures; without this, 6% of
    test targets appeared verbatim in train. The group key is the smallest
    member, so the mapping does not depend on input order.
    """
    parent: dict[str, str] = {}

    def find(k: str) -> str:
        parent.setdefault(k, k)
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    first_seen: dict[str, str] = {}
    for key, signature in pairs:
        find(key)
        other = first_seen.setdefault(signature, key)
        a, b = find(key), find(other)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return {k: find(k) for k in parent}


def split_for(uid: str, val: float = 0.1, test: float = 0.1, salt: str = "kibitz") -> str:
    """
    Deterministic split assignment for a split key (see `split_key`).

    Hashing the key (rather than shuffling) means re-running gives the same
    split, and a new study moves an existing one only if it merges their groups.
    """
    digest = hashlib.sha1(f"{salt}:{uid}".encode()).hexdigest()[:8]
    position = int(digest, 16) / 0xFFFFFFFF
    if position < test:
        return "test"
    if position < test + val:
        return "val"
    return "train"


def build_record(meta: dict, move: dict, source: str, uid: str, target: str) -> dict:
    """
    One training pair, in the neutral `{"messages": [...]}` format.

    `meta` carries provenance so a generated comment can be traced back to its
    game and compared against the annotator. Pass --strict to build_dataset to
    emit only the `messages` key.
    """
    from kibitz import prompt

    return {
        "messages": prompt.build_messages(meta, move, target),
        "meta": {
            "source": source,
            "uid": uid,
            "ply": move.get("ply"),
            "san": move["san"],
            "classification": move["move_classification"],
            "phase": move["game_phase"],
            "motifs": move["tactical_motifs"],
        },
    }
