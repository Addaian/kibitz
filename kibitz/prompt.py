"""
Turning an enriched move into a prompt.

The same construction is used for training pairs and at inference time — if the
two ever drift the model sees a prompt shape it was never trained on, which is
the classic way a fine-tune quietly degrades. `build_messages` (training) and
`build_prompt` (inference) therefore share `format_context`.

Fields the record does not carry are omitted rather than rendered as "unknown",
so the model is never trained to condition on a literal "?".

Kibitz comments on every move, but most moves deserve a line, not a paragraph.
The `Length:` field is how that is controlled: in training it is read off the
annotator's target, so the model learns what "brief" means; at inference
`length_for_move` sets it from the features, so quiet moves get a brief line.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are Kibitz, an expert chess commentator. Given a position and the move "
    "played, write short, insightful commentary in the style of a grandmaster "
    "analyst. Explain what the move does and why it matters. Do not repeat the "
    "evaluation numbers back."
)

INSTRUCTION = "Write 1-3 sentences of commentary for this move."

# Targets this short train the "brief" setting. 43% of the corpus qualifies.
BRIEF_MAX_WORDS = 12



def _players(meta: dict) -> str | None:
    white, black = meta.get("white"), meta.get("black")
    if not white or not black or white == "?" or black == "?":
        return None
    year = (meta.get("date") or "")[:4]
    year = year if year.isdigit() else None      # "????", "Marc", "08.2" happen
    where = ", ".join(x for x in (meta.get("event"), year) if x and x != "?")
    return f"{white} vs. {black}" + (f" ({where})" if where else "")


def format_context(meta: dict, move: dict) -> str:
    """The structured block describing one move. Shared by training and inference."""
    lines = []

    players = _players(meta)
    if players:
        lines.append(f"Game: {players}")

    opening, eco = meta.get("opening"), meta.get("eco")
    if opening and opening != "?":
        lines.append(f"Opening: {opening}" + (f" ({eco})" if eco and eco != "?" else ""))

    lines.append(f"Phase: {move['game_phase']}")
    if move.get("previous_moves"):
        lines.append(f"Previous moves: {move['previous_moves']}")
    lines.append(f"Move {move['move_number']} ({move['color']}): {move['san']}")

    # The annotator's own !/? marks (NAGs) are deliberately absent: they are
    # hindsight from the same person who wrote the target, and a live game or an
    # unannotated PGN never has them.

    lines.append(f"Classification: {move['move_classification']}")
    lines.append(
        f"Evaluation: {move['eval_before']:+.2f} -> {move['eval_after']:+.2f} "
        f"(delta {move['eval_delta']:+.2f} for {move['color']})"
    )

    if move.get("mate_in") is not None:
        mater = "White" if move["mate_in"] > 0 else "Black"
        lines.append(f"Forced mate: {mater} mates in {abs(move['mate_in'])}")
    if not move.get("is_best") and move.get("best_move_san"):
        lines.append(f"Engine preferred: {move['best_move_san']}")
        if move.get("best_line_san"):
            lines.append(f"Engine line: {move['best_line_san']}")
    if move.get("captured"):
        lines.append(f"Captures: {move['captured']}")
    if move.get("threatens"):
        lines.append(f"Threatens to win: {', '.join(move['threatens'])}")
    if move.get("left_hanging"):
        lines.append(f"Left hanging: {', '.join(move['left_hanging'])}")
    if move.get("tactical_motifs"):
        lines.append(f"Tactical motifs: {', '.join(move['tactical_motifs'])}")

    flags = []
    if move.get("is_forcing"):
        flags.append("forcing")
    if move.get("is_sacrifice"):
        flags.append(f"sacrifices {move['material_offered']:+.1f} pawns of material")
    if flags:
        lines.append(f"Notes: {'; '.join(flags)}")

    lines.append(f"Position after the move (FEN): {move['fen_after']}")
    return "\n".join(lines)


def length_for_target(target: str) -> str:
    """Training: the length setting is whatever the annotator actually wrote."""
    return "brief" if len(target.split()) <= BRIEF_MAX_WORDS else "full"


def length_for_move(move: dict) -> str:
    """Inference: a full comment for moves with something to explain, else brief."""
    notable = (
        move.get("move_classification") in ("inaccuracy", "mistake", "blunder")
        or move.get("tactical_motifs")
        or move.get("is_sacrifice")
        or move.get("mate_in") is not None
    )
    return "full" if notable else "brief"


def build_prompt(meta: dict, move: dict, length: str = "full") -> str:
    """The user turn: context, length setting and the instruction."""
    return f"{format_context(meta, move)}\nLength: {length}\n\n{INSTRUCTION}"


def build_messages(meta: dict, move: dict, target: str | None = None) -> list[dict]:
    """
    A training pair in the neutral `{"role", "content"}` messages format.

    `target` is the cleaned human annotation. Omit it to get just the user turn,
    which is what inference needs; the length setting then comes from the move.
    """
    length = length_for_target(target) if target else length_for_move(move)
    messages = [{"role": "user", "content": build_prompt(meta, move, length)}]
    if target:
        messages.append({"role": "assistant", "content": target})
    return messages
