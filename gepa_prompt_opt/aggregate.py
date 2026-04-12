from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .io_utils import read_json, write_json


DEFAULT_FAILURE_WEIGHTS = {
    "loop_closure_failure": 1.0,
    "progressive_scene_drift": 0.8,
    "memory_loss": 0.8,
    "action_not_followed": 0.8,
    "late_action_onset": 0.5,
    "temporal_jitter": 0.4,
    "motion_discontinuity": 0.4,
    "scene_layout_drift": 0.4,
}


DEFAULT_TEXT_PATTERNS = {
    "loop_completion": [r"\breturn\b", r"\bback to (the )?start\b", r"\b(loop|loop closure)\b"],
    "drift": [r"\bdrift\b", r"\binconsistent\b", r"\bchanges over time\b"],
    "action_adherence": [r"\bnot follow\b", r"\bignored\b", r"\bwrong action\b"],
    "temporal_artifacts": [r"\bjitter\b", r"\bshaky\b", r"\bjump\b", r"\babrupt\b"],
}


@dataclass(frozen=True)
class ObjectiveConfig:
    alpha: float = 0.5
    beta: float = 0.7
    gamma: float = 0.1


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def per_video_score(video_entry: dict[str, Any]) -> float:
    report = video_entry.get("failure_report") or {}
    vlm = report.get("vlm_judgments") or {}

    a = float(vlm.get("action_adherence_score", 0.5))
    c = float(vlm.get("scene_consistency_score", 0.5))
    t = float(vlm.get("temporal_coherence_score", 0.5))

    raw = video_entry.get("raw_metrics") or {}
    drifts = []
    for k in [
        "drifting_semantic",
        "drifting_aesthetic",
        "drifting_motion_smoothness",
        "drifting_naturalness",
    ]:
        if k in raw and isinstance(raw[k], (int, float)):
            drifts.append(float(raw[k]))
    mean_drift = sum(drifts) / len(drifts) if drifts else 0.0
    drift_term = 1.0 - _clamp01(mean_drift)

    s = 0.45 * _clamp01(a) + 0.25 * _clamp01(c) + 0.20 * _clamp01(t) + 0.10 * drift_term
    return _clamp01(s)


def failure_penalty(video_entry: dict[str, Any], weights: dict[str, float] | None = None) -> float:
    weights = weights or DEFAULT_FAILURE_WEIGHTS
    report = video_entry.get("failure_report") or {}
    failures = report.get("failure_modes") or []
    total = 0.0
    for fm in failures:
        label = fm.get("label")
        sev = float(fm.get("severity", 0.0))
        w = float(weights.get(label, 0.2))
        total += w * _clamp01(sev)
    return _clamp01(total)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / (len(values) - 1)


def summarize_free_text(videos: list[dict[str, Any]], max_examples: int = 3) -> dict[str, Any]:
    counts: dict[str, int] = {k: 0 for k in DEFAULT_TEXT_PATTERNS}
    examples: dict[str, list[dict[str, Any]]] = {k: [] for k in DEFAULT_TEXT_PATTERNS}
    compiled = {k: [re.compile(pat, re.IGNORECASE) for pat in pats] for k, pats in DEFAULT_TEXT_PATTERNS.items()}

    for v in videos:
        report = v.get("failure_report") or {}
        text = str(report.get("free_text_summary", ""))
        for bucket, pats in compiled.items():
            if any(p.search(text) for p in pats):
                counts[bucket] += 1
                if len(examples[bucket]) < max_examples:
                    examples[bucket].append({"id": v.get("id"), "free_text_summary": text})

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {"counts": counts, "examples": examples, "ranked": ranked}


def aggregate_candidate(
    combined_report_path: str,
    *,
    objective_cfg: ObjectiveConfig | None = None,
    failure_weights: dict[str, float] | None = None,
) -> tuple[float, dict[str, Any]]:
    objective_cfg = objective_cfg or ObjectiveConfig()
    data = read_json(combined_report_path)
    videos = data.get("videos") or []

    scores = [per_video_score(v) for v in videos]
    penalties = [failure_penalty(v, failure_weights) for v in videos]

    mean_score = sum(scores) / len(scores) if scores else 0.0
    worst_p10 = _percentile(scores, 10.0)
    var = _variance(scores)
    mean_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    any_failure_rate = (
        sum(1 for v in videos if (v.get("failure_report") or {}).get("failure_modes")) / len(videos) if videos else 0.0
    )

    j = mean_score + objective_cfg.alpha * worst_p10 - objective_cfg.beta * mean_penalty - objective_cfg.gamma * var

    freq: dict[str, int] = {}
    sev_sum: dict[str, float] = {}
    sev_cnt: dict[str, int] = {}
    for v in videos:
        report = v.get("failure_report") or {}
        for fm in report.get("failure_modes") or []:
            label = fm.get("label")
            if not label:
                continue
            freq[label] = freq.get(label, 0) + 1
            sev_sum[label] = sev_sum.get(label, 0.0) + float(fm.get("severity", 0.0))
            sev_cnt[label] = sev_cnt.get(label, 0) + 1

    top_failures = sorted(
        [
            {
                "label": k,
                "frequency": freq[k] / len(videos) if videos else 0.0,
                "mean_severity": (sev_sum[k] / sev_cnt[k]) if sev_cnt[k] else 0.0,
            }
            for k in freq.keys()
        ],
        key=lambda x: (x["frequency"], x["mean_severity"]),
        reverse=True,
    )[:8]

    ranked = sorted(
        [{"id": v.get("id"), "score": per_video_score(v), "video": v} for v in videos],
        key=lambda x: x["score"],
        reverse=True,
    )
    good = [
        {
            "id": x["id"],
            "score": x["score"],
            "free_text_summary": (x["video"].get("failure_report") or {}).get("free_text_summary", ""),
        }
        for x in ranked[:3]
    ]
    bad = [
        {
            "id": x["id"],
            "score": x["score"],
            "failure_modes": (x["video"].get("failure_report") or {}).get("failure_modes", []),
            "free_text_summary": (x["video"].get("failure_report") or {}).get("free_text_summary", ""),
        }
        for x in ranked[-3:]
    ]

    text_summary = summarize_free_text(videos)

    side_info = {
        "objective": {
            "J": float(j),
            "mean_score": mean_score,
            "worst_p10_score": worst_p10,
            "variance": var,
            "mean_failure_penalty": mean_penalty,
            "any_failure_rate": any_failure_rate,
        },
        "top_failure_modes": top_failures,
        "free_text_patterns": text_summary,
        "representative_examples": {"good": good, "bad": bad},
    }
    return float(j), side_info


def write_aggregate_json(out_path: str, j: float, side_info: dict[str, Any]) -> None:
    write_json(out_path, {"J": j, "side_info": side_info})
