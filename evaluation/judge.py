"""Rank-3 description scoring via Qwen as a text-only LLM judge."""
from __future__ import annotations

import json
import re

# Local config MUST be imported before torch/transformers (pulled in by describe).
from src import config  # noqa: F401
from src.describe import QwenDescriber

from .match import MatchPair, PredEvent

_JUDGE_PROMPT = """You are evaluating a video-analytics system that describes person-vehicle interactions.

Ground-truth note:
{gt_notes}

System prediction:
{pred_text}

Score whether the system prediction refers to the same core person-vehicle event as the ground-truth note.
Ignore wording differences. Judge the action (enter/exit/interact/load/unload/open door/etc.) and subject.
Do not require the prediction to mention every detail in the note.

Return ONLY a JSON object with this schema:
{{"score": 0|1|2, "reason": "short explanation"}}

Scoring:
- 2 = same core action and subject
- 1 = partially correct (right action but missing/vague detail, or related but incomplete)
- 0 = wrong or unrelated
"""


def pack_prediction_text(pred: PredEvent) -> str:
    return (
        f"connection: {pred.connection}\n"
        f"action_detail: {pred.action_detail}\n"
        f"person: {pred.person_description}\n"
        f"vehicle: {pred.vehicle_description}"
    )


def _extract_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    braced = re.search(r"\{.*\}", text, re.DOTALL)
    if braced:
        try:
            data = json.loads(braced.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def parse_judge_reply(raw: str) -> tuple[int, str]:
    """Parse judge output into ``(score, reason)``. Failures → score 0."""
    data = _extract_json_object(raw)
    if data is None:
        return 0, f"parse_failed: {raw[:200]}"
    try:
        score = int(data.get("score"))
    except (TypeError, ValueError):
        return 0, f"parse_failed: {raw[:200]}"
    if score not in (0, 1, 2):
        return 0, f"parse_failed_bad_score: {raw[:200]}"
    reason = str(data.get("reason") or "").strip()
    return score, reason


def judge_pair(describer: QwenDescriber, pair: MatchPair) -> MatchPair:
    """Fill ``desc_score`` / ``desc_reason`` on ``pair`` using Qwen."""
    prompt = _JUDGE_PROMPT.format(
        gt_notes=pair.gt.notes or "(empty)",
        pred_text=pack_prediction_text(pair.pred),
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    raw = describer._generate(messages)
    score, reason = parse_judge_reply(raw)
    pair.desc_score = score
    pair.desc_reason = reason
    return pair


def judge_pairs(
    describer: QwenDescriber,
    pairs: list[MatchPair],
    *,
    verbose: bool = False,
) -> list[MatchPair]:
    for i, pair in enumerate(pairs, start=1):
        judge_pair(describer, pair)
        if verbose:
            print(
                f"  judge [{i}/{len(pairs)}] "
                f"{pair.gt.clip_id} gt#{pair.gt.interaction_id} "
                f"→ score={pair.desc_score} ({pair.desc_reason[:80]})"
            )
    return pairs


def ensure_describer(
    describer: QwenDescriber | None,
) -> QwenDescriber:
    """Reuse a loaded describer or construct one for standalone eval."""
    if describer is not None:
        return describer
    return QwenDescriber()
