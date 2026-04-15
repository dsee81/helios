# summary_templates.py
# Deterministic summary generation from failure modes

def generate_summary(failure_modes: list, task_type: str, vlm_judgments: dict = None) -> str:
    """
    Generate a short free-text summary from the top failure modes.

    Args:
        failure_modes: List of failure mode dicts, sorted by severity
        task_type: The task type
        vlm_judgments: VLM judgments dict

    Returns:
        Short explanation string
    """
    if not failure_modes:
        return "The video appears to follow the instruction correctly with no major issues detected."

    # Take top 3 most severe failures
    top_failures = sorted(failure_modes, key=lambda x: x["severity"], reverse=True)[:3]
    failure_labels = [f["label"] for f in top_failures]

    # Template-based generation
    templates = {
        "late_action_onset": "starts the commanded action too late",
        "wrong_turn_direction": "turns in the wrong direction",
        "doorway_not_entered": "fails to enter the doorway",
        "progressive_scene_drift": "accumulates scene drift over time",
        "loop_closure_failure": "fails to return to the starting point",
        "temporal_jitter": "exhibits motion jitter and instability",
        "scene_layout_drift": "has inconsistent scene layout changes",
        "action_not_followed": "does not follow the commanded action"
    }

    issues = []
    for label in failure_labels:
        if label in templates:
            issues.append(templates[label])

    if not issues:
        return "Minor issues detected but no specific failure modes identified."

    if len(issues) == 1:
        summary = f"The model {issues[0]}."
    elif len(issues) == 2:
        summary = f"The model {issues[0]} and {issues[1]}."
    else:
        summary = f"The model {', '.join(issues[:-1])}, and {issues[-1]}."

    # Add positive notes if applicable
    positive_notes = []
    if vlm_judgments:
        scene_score = vlm_judgments.get("scene_consistency_score", 0.5)
        if scene_score > 0.7:
            positive_notes.append("preserves scene consistency well")
        temporal_score = vlm_judgments.get("temporal_coherence_score", 0.5)
        if temporal_score > 0.7:
            positive_notes.append("maintains temporal coherence")

    if positive_notes:
        positive_str = " and ".join(positive_notes)
        summary = f"The video {positive_str}, but {summary[10:].lower()}"  # Remove "The model" and adjust

    return summary