"""
Turning an enriched move into a prompt.

The same construction is used for training pairs and at inference time — if the
two ever drift the model sees a prompt shape it was never trained on, which is
the classic way a fine-tune quietly degrades. `build_messages` (training) and
`build_prompt` (inference) therefore share `format_context`.

Fields the record does not carry are omitted rather than rendered as "unknown",
so the model is never trained to condition on a literal "?".
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are Kibitz, an expert chess commentator. Given a position and the move "
    "played, write short, insightful commentary in the style of a grandmaster "
    "analyst. Explain what the move does and why it matters. Do not repeat the "
    "evaluation numbers back."
)

INSTRUCTION = "Write 1-3 sentences of commentary for this move."

# The symbols annotators actually use, from the PGN NAG codes.
NAG_SYMBOL = {1: "!", 2: "?", 3: "!!", 4: "??", 5: "!?", 6: "?!"}


def _players(meta: dict) -> str | None:
    white, black = meta.get("white"), meta.get("black")
    if not white or not black or white == "?" or black == "?":
        return None
    where = ", ".join(x for x in (meta.get("event"), (meta.get("date") or "")[:4])
                      if x and x not in ("?", ""))
    return f"{white} vs. {black}" + (f" ({where})" if where else "")


def format_context(meta: dict, move: dict) -> str:
    """The structured block describing one move. Shared by training and inference."""
    lines = []

    players = _players(meta)
    if players:
        lines.append(f"Game: {players}")

    opening = meta.get("opening")
    if opening:
        eco = meta.get("eco")
        lines.append(f"Opening: {opening}" + (f" ({eco})" if eco else ""))

    lines.append(f"Phase: {move['game_phase']}")
    lines.append(f"Move {move['move_number']} ({move['color']}): {move['san']}")

    symbols = "".join(NAG_SYMBOL[n] for n in sorted(move.get("nags") or []) if n in NAG_SYMBOL)
    if symbols:
        lines.append(f"Annotator's mark: {symbols}")

    lines.append(f"Classification: {move['move_classification']}")
    lines.append(
        f"Evaluation: {move['eval_before']:+.2f} -> {move['eval_after']:+.2f} "
        f"(delta {move['eval_delta']:+.2f} for {move['color']})"
    )

    if move.get("mate_in") is not None:
        lines.append(f"Forced mate in {abs(move['mate_in'])}")
    if not move.get("is_best") and move.get("best_move"):
        lines.append(f"Engine preferred: {move['best_move']}")
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


def build_prompt(meta: dict, move: dict) -> str:
    """The user turn: context plus the instruction. Used at inference time."""
    return f"{format_context(meta, move)}\n\n{INSTRUCTION}"


def build_messages(meta: dict, move: dict, target: str | None = None) -> list[dict]:
    """
    A training pair in Claude Messages format.

    `target` is the cleaned human annotation. Omit it to get just the user turn,
    which is what inference needs.
    """
    messages = [{"role": "user", "content": build_prompt(meta, move)}]
    if target:
        messages.append({"role": "assistant", "content": target})
    return messages
