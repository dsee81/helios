# failure_explainer.py
# Main entry point for failure explanation system

from .failure_taxonomy import SUPPORTED_TASK_TYPES
from .failure_rules import apply_rules
from .vlm_judges import get_vlm_judgments
from .summary_templates import generate_summary

def infer_segment_metrics(raw_metrics: dict) -> dict:
    """
    Infer segment-wise metrics from drifting metrics.
    
    Drifting metrics measure the difference between early (first 15%) and late (last 15%) frames.
    We use these to infer segment behavior:
    - If drifting is high, early was good but late degraded
    - If drifting is low, performance was consistent
    """
    segment_metrics = {
        "early": {},
        "middle": {},
        "late": {}
    }
    
    # Map raw metrics to segment metrics
    metric_mappings = {
        "semantic": "drifting_semantic",
        "aesthetic": "drifting_aesthetic",
        "motion_smoothness": "drifting_motion_smoothness",
        "naturalness": "drifting_naturalness"
    }
    
    for base_metric, drift_metric in metric_mappings.items():
        if base_metric in raw_metrics and drift_metric in raw_metrics:
            base_score = raw_metrics[base_metric]
            drift = raw_metrics[drift_metric]
            
            # Infer early/late from base and drift
            # Early tends to be better (higher), late tends to show more drift
            early_score = min(base_score + drift / 2, 1.0)
            late_score = max(base_score - drift / 2, 0.0)
            middle_score = base_score  # Middle is approximately the average
            
            segment_metrics["early"][base_metric] = early_score
            segment_metrics["middle"][base_metric] = middle_score
            segment_metrics["late"][base_metric] = late_score
        elif base_metric in raw_metrics:
            # No drift metric, assume consistent
            score = raw_metrics[base_metric]
            segment_metrics["early"][base_metric] = score
            segment_metrics["middle"][base_metric] = score
            segment_metrics["late"][base_metric] = score
    
    # Also use motion metrics as proxy for action alignment
    if "motion_amplitude" in raw_metrics:
        # Higher motion amplitude in early = better action onset
        motion = raw_metrics["motion_amplitude"]
        segment_metrics["early"]["action_alignment"] = motion * 0.8
        segment_metrics["middle"]["action_alignment"] = motion
        segment_metrics["late"]["action_alignment"] = motion * 0.9
    
    return segment_metrics

def explain_failures(raw_metrics: dict,
                    task_type: str = "reconstruct_original",
                    prompt_text: str = "",
                    video_path: str = None) -> dict:
    """
    Main function to explain failures in Helios evaluation.
    
    This is a post-processing layer on top of standard Helios metrics.
    It requires only the raw metrics and task type from your evaluation pipeline.

    Args:
        raw_metrics: Dict of raw scalar metrics from Helios eval
                    (semantic, naturalness, motion_amplitude, motion_smoothness,
                     aesthetic, drifting_aesthetic, drifting_semantic, drifting_naturalness,
                     drifting_motion_smoothness, etc.)
        task_type: One of SUPPORTED_TASK_TYPES
                  (reconstruct_original, immediate_turn, doorway_entry, loop, minor_scene_change)
        prompt_text: Optional instruction/prompt text for context
        video_path: Optional path to generated video for VLM analysis

    Returns:
        Dict with failure analysis report including:
        - raw_metrics: Original metrics
        - inferred_segment_metrics: Derived from drifting metrics
        - vlm_judgments: Task-specific structure from VLM (if video provided)
        - failure_modes: List of detected failures with evidence
        - free_text_summary: Human-readable explanation
    """

    # Validate inputs
    if task_type not in SUPPORTED_TASK_TYPES:
        raise ValueError(f"Unsupported task_type: {task_type}. Supported: {SUPPORTED_TASK_TYPES}")
    
    if not raw_metrics:
        raise ValueError("raw_metrics cannot be empty")

    # Infer segment metrics from drifting metrics
    segment_metrics = infer_segment_metrics(raw_metrics)

    # Get VLM judgments if video is provided
    vlm_judgments = {}
    task_specific_metrics = {}
    try:
        if video_path and prompt_text:
            vlm_result = get_vlm_judgments(prompt_text, task_type, None, video_path)
            vlm_judgments = vlm_result
            # Extract task-specific metrics from VLM output
            if "observed_action_start_segment" in vlm_result:
                task_specific_metrics["observed_action_start_segment"] = vlm_result["observed_action_start_segment"]
            if "observed_turn_direction" in vlm_result:
                task_specific_metrics["observed_turn_direction"] = vlm_result["observed_turn_direction"]
            if "doorway_entered" in vlm_result:
                task_specific_metrics["doorway_entry_success"] = vlm_result["doorway_entered"]
            if "endpoint_similarity" in vlm_result:
                task_specific_metrics["endpoint_similarity"] = vlm_result["endpoint_similarity"]
            if "loop_closure_score" in vlm_result:
                task_specific_metrics["loop_closure_score"] = vlm_result["loop_closure_score"]
            if "loop_closure_achieved" in vlm_result:
                task_specific_metrics["loop_closure_achieved"] = vlm_result["loop_closure_achieved"]
            for key in (
                "cyclic_trajectory_score",
                "revisit_near_end_score",
                "seam_smoothness_score",
                "best_revisit_similarity",
                "mean_topk_revisit_similarity",
            ):
                if key in vlm_result:
                    task_specific_metrics[key] = vlm_result[key]
        else:
            vlm_judgments = {"vlm_unavailable": True}
    except Exception as e:
        vlm_judgments = {"vlm_error": str(e)}

    # Prepare metrics dict for rules engine
    metrics_for_rules = {
        "raw_metrics": raw_metrics,
        "segment_metrics": segment_metrics,
        "task_specific_metrics": task_specific_metrics
    }

    # Apply rule-based diagnosis
    failure_modes = apply_rules(metrics_for_rules, task_type, vlm_judgments)

    # Generate summary
    free_text_summary = generate_summary(failure_modes, task_type, vlm_judgments)

    # Build final report
    report = {
        "raw_metrics": raw_metrics,
        "inferred_segment_metrics": segment_metrics,
        "vlm_judgments": vlm_judgments,
        "failure_modes": failure_modes,
        "free_text_summary": free_text_summary
    }

    return report
