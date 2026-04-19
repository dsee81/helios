# failure_rules.py
# Rule-based failure diagnosis engine

import math
from .failure_taxonomy import ALL_FAILURE_LABELS

def normalize_severity(value, min_val, max_val, invert=False):
    """Normalize a value to [0,1] severity score."""
    if invert:
        # For metrics where lower is worse
        severity = (max_val - value) / (max_val - min_val) if max_val > min_val else 0
    else:
        # For metrics where higher is worse
        severity = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
    return max(0, min(1, severity))

class FailureRule:
    def __init__(self, label, condition_func, severity_func, evidence_func, rule_id):
        self.label = label
        self.condition_func = condition_func
        self.severity_func = severity_func
        self.evidence_func = evidence_func
        self.rule_id = rule_id

    def check(self, metrics, task_type, vlm_judgments=None):
        """Check if rule triggers and return failure mode dict if so."""
        if self.condition_func(metrics, task_type, vlm_judgments):
            severity = self.severity_func(metrics, task_type, vlm_judgments)
            evidence = self.evidence_func(metrics, task_type, vlm_judgments)
            return {
                "label": self.label,
                "severity": severity,
                "triggered_by": self.rule_id,
                "evidence": evidence
            }
        return None

# Define rules
RULES = []

# Late action onset rule
def late_action_onset_condition(metrics, task_type, vlm_judgments):
    if task_type not in ["immediate_turn", "doorway_entry"]:
        return False
    segment_metrics = metrics.get("segment_metrics", {})
    raw_metrics = metrics.get("raw_metrics", {})
    
    early_action = segment_metrics.get("early", {}).get("action_alignment", 1.0)
    mid_action = segment_metrics.get("middle", {}).get("action_alignment", 1.0)
    
    # Rule 1: Action alignment low early but improves in middle
    rule1 = early_action < 0.5 and mid_action > early_action + 0.15
    
    # Rule 2: Very low motion amplitude (action didn't occur or was too weak)
    motion = raw_metrics.get("motion_amplitude", 0.5)
    rule2 = motion < 0.4
    
    # Rule 3: Check VLM judgment
    vlm_check = vlm_judgments.get("action_started_early_enough") == False if vlm_judgments else False
    
    return rule1 or rule2 or vlm_check

def late_action_onset_severity(metrics, task_type, vlm_judgments):
    segment_metrics = metrics.get("segment_metrics", {})
    raw_metrics = metrics.get("raw_metrics", {})
    
    early = segment_metrics.get("early", {}).get("action_alignment", 1.0)
    motion = raw_metrics.get("motion_amplitude", 0.5)
    
    # Severity from low early alignment
    sev1 = normalize_severity(early, 0.5, 1.0, invert=True)
    
    # Severity from low motion
    sev2 = normalize_severity(motion, 0.4, 0.8, invert=True)
    
    # Combined severity
    return max(sev1, sev2)

def late_action_onset_evidence(metrics, task_type, vlm_judgments):
    segment_metrics = metrics.get("segment_metrics", {})
    task_specific = metrics.get("task_specific_metrics", {})
    evidence = {
        "expected_start_frame_max": task_specific.get("expected_action_start_frame_max", 3),
        "observed_start_frame": task_specific.get("observed_action_start_frame", None),
        "early_action_alignment": segment_metrics.get("early", {}).get("action_alignment", None),
        "mid_action_alignment": segment_metrics.get("middle", {}).get("action_alignment", None)
    }
    if vlm_judgments:
        evidence["vlm_action_started_early_enough"] = vlm_judgments.get("action_started_early_enough", None)
        evidence["observed_action_start_segment"] = vlm_judgments.get("observed_action_start_segment", None)
    return evidence

RULES.append(FailureRule(
    "late_action_onset",
    late_action_onset_condition,
    late_action_onset_severity,
    late_action_onset_evidence,
    "late_action_onset_rule_v1"
))

# Wrong turn direction rule
def wrong_turn_direction_condition(metrics, task_type, vlm_judgments):
    if task_type not in ["immediate_turn"]:
        return False
    task_specific = metrics.get("task_specific_metrics", {})
    expected = task_specific.get("expected_turn_direction")
    observed = task_specific.get("observed_turn_direction")
    if expected and observed:
        return expected.lower() != observed.lower()
    if vlm_judgments:
        vlm_expected = task_specific.get("expected_turn_direction")
        vlm_observed = vlm_judgments.get("observed_turn_direction")
        if vlm_expected and vlm_observed:
            return vlm_expected.lower() != vlm_observed.lower()
    return False

def wrong_turn_direction_severity(metrics, task_type, vlm_judgments):
    # High severity if direction is wrong
    return 1.0

def wrong_turn_direction_evidence(metrics, task_type, vlm_judgments):
    task_specific = metrics.get("task_specific_metrics", {})
    evidence = {
        "expected_turn_direction": task_specific.get("expected_turn_direction"),
        "observed_turn_direction": task_specific.get("observed_turn_direction")
    }
    if vlm_judgments:
        evidence["vlm_observed_turn_direction"] = vlm_judgments.get("observed_turn_direction")
    return evidence

RULES.append(FailureRule(
    "wrong_turn_direction",
    wrong_turn_direction_condition,
    wrong_turn_direction_severity,
    wrong_turn_direction_evidence,
    "wrong_turn_direction_rule_v1"
))

# Doorway not entered rule
def doorway_not_entered_condition(metrics, task_type, vlm_judgments):
    if task_type not in ["doorway_entry"]:
        return False
    task_specific = metrics.get("task_specific_metrics", {})
    success = task_specific.get("doorway_entry_success", True)
    if not success:
        return True
    if vlm_judgments:
        vlm_success = vlm_judgments.get("doorway_entered", True)
        return not vlm_success
    return False

def doorway_not_entered_severity(metrics, task_type, vlm_judgments):
    return 1.0

def doorway_not_entered_evidence(metrics, task_type, vlm_judgments):
    task_specific = metrics.get("task_specific_metrics", {})
    evidence = {
        "doorway_entry_success": task_specific.get("doorway_entry_success")
    }
    if vlm_judgments:
        evidence["vlm_doorway_entered"] = vlm_judgments.get("doorway_entered")
    return evidence

RULES.append(FailureRule(
    "doorway_not_entered",
    doorway_not_entered_condition,
    doorway_not_entered_severity,
    doorway_not_entered_evidence,
    "doorway_not_entered_rule_v1"
))

# Progressive scene drift rule
def progressive_scene_drift_condition(metrics, task_type, vlm_judgments):
    segment_metrics = metrics.get("segment_metrics", {})
    raw_metrics = metrics.get("raw_metrics", {})
    
    # Check semantic and aesthetic drift from raw metrics
    semantic_drift = raw_metrics.get("drifting_semantic", 0.0)
    aesthetic_drift = raw_metrics.get("drifting_aesthetic", 0.0)
    
    # Also check inferred segment metrics
    early_scene = segment_metrics.get("early", {}).get("semantic", 1.0)
    late_scene = segment_metrics.get("late", {}).get("semantic", 1.0)
    drift = early_scene - late_scene
    
    vlm_drift = vlm_judgments.get("scene_drift_visible", False) if vlm_judgments else False
    return drift > 0.15 or semantic_drift > 0.2 or aesthetic_drift > 0.15 or vlm_drift

def progressive_scene_drift_severity(metrics, task_type, vlm_judgments):
    segment_metrics = metrics.get("segment_metrics", {})
    early = segment_metrics.get("early", {}).get("scene_similarity", 1.0)
    late = segment_metrics.get("late", {}).get("scene_similarity", 1.0)
    drift = early - late
    severity = normalize_severity(drift, 0, 0.5)
    if vlm_judgments and vlm_judgments.get("scene_drift_visible"):
        severity = max(severity, 0.7)
    return severity

def progressive_scene_drift_evidence(metrics, task_type, vlm_judgments):
    segment_metrics = metrics.get("segment_metrics", {})
    evidence = {
        "early_scene_similarity": segment_metrics.get("early", {}).get("scene_similarity"),
        "late_scene_similarity": segment_metrics.get("late", {}).get("scene_similarity")
    }
    if vlm_judgments:
        evidence["vlm_scene_drift_visible"] = vlm_judgments.get("scene_drift_visible")
    return evidence

RULES.append(FailureRule(
    "progressive_scene_drift",
    progressive_scene_drift_condition,
    progressive_scene_drift_severity,
    progressive_scene_drift_evidence,
    "progressive_scene_drift_rule_v1"
))

# Loop closure failure rule
def loop_closure_failure_condition(metrics, task_type, vlm_judgments):
    if task_type not in ["loop"]:
        return False
    task_specific = metrics.get("task_specific_metrics", {})
    endpoint_sim = task_specific.get("endpoint_similarity", 1.0)
    loop_score = task_specific.get("loop_closure_score", endpoint_sim)
    cyclic = task_specific.get("cyclic_trajectory_score", loop_score)
    revisit = task_specific.get("revisit_near_end_score", loop_score)
    seam = task_specific.get("seam_smoothness_score", loop_score)
    vlm_failed = False
    if vlm_judgments:
        vlm_failed = vlm_judgments.get("loop_closure_achieved") is False
        endpoint_sim = task_specific.get("endpoint_similarity", vlm_judgments.get("endpoint_similarity", endpoint_sim))
        loop_score = task_specific.get("loop_closure_score", vlm_judgments.get("loop_closure_score", loop_score))
        cyclic = task_specific.get("cyclic_trajectory_score", vlm_judgments.get("cyclic_trajectory_score", cyclic))
        revisit = task_specific.get("revisit_near_end_score", vlm_judgments.get("revisit_near_end_score", revisit))
        seam = task_specific.get("seam_smoothness_score", vlm_judgments.get("seam_smoothness_score", seam))
    return loop_score < 0.72 or cyclic < 0.68 or revisit < 0.65 or seam < 0.60 or vlm_failed

def loop_closure_failure_severity(metrics, task_type, vlm_judgments):
    task_specific = metrics.get("task_specific_metrics", {})
    sim = task_specific.get("loop_closure_score", task_specific.get("endpoint_similarity", 1.0))
    if vlm_judgments:
        sim = task_specific.get("loop_closure_score", vlm_judgments.get("loop_closure_score", sim))
    cyclic = task_specific.get("cyclic_trajectory_score", vlm_judgments.get("cyclic_trajectory_score", sim) if vlm_judgments else sim)
    revisit = task_specific.get("revisit_near_end_score", vlm_judgments.get("revisit_near_end_score", sim) if vlm_judgments else sim)
    seam = task_specific.get("seam_smoothness_score", vlm_judgments.get("seam_smoothness_score", sim) if vlm_judgments else sim)
    weak = min(float(sim), float(cyclic), float(revisit), float(seam))
    return normalize_severity(weak, 0.6, 0.85, invert=True)

def loop_closure_failure_evidence(metrics, task_type, vlm_judgments):
    task_specific = metrics.get("task_specific_metrics", {})
    evidence = {
        "endpoint_similarity": task_specific.get("endpoint_similarity"),
        "loop_closure_score": task_specific.get("loop_closure_score"),
        "cyclic_trajectory_score": task_specific.get("cyclic_trajectory_score"),
        "revisit_near_end_score": task_specific.get("revisit_near_end_score"),
        "seam_smoothness_score": task_specific.get("seam_smoothness_score"),
    }
    if vlm_judgments:
        evidence["vlm_loop_closure_achieved"] = vlm_judgments.get("loop_closure_achieved")
        evidence["vlm_loop_closure_score"] = vlm_judgments.get("loop_closure_score")
        evidence["vlm_cyclic_trajectory_score"] = vlm_judgments.get("cyclic_trajectory_score")
        evidence["vlm_revisit_near_end_score"] = vlm_judgments.get("revisit_near_end_score")
        evidence["vlm_seam_smoothness_score"] = vlm_judgments.get("seam_smoothness_score")
    return evidence

RULES.append(FailureRule(
    "loop_closure_failure",
    loop_closure_failure_condition,
    loop_closure_failure_severity,
    loop_closure_failure_evidence,
    "loop_closure_failure_rule_v1"
))

# Temporal jitter rule
def temporal_jitter_condition(metrics, task_type, vlm_judgments):
    raw_metrics = metrics.get("raw_metrics", {})
    motion_amplitude = raw_metrics.get("motion_amplitude", 0.5)
    motion_smoothness = raw_metrics.get("motion_smoothness", 1.0)
    return motion_amplitude > 0.7 and motion_smoothness < 0.6

def temporal_jitter_severity(metrics, task_type, vlm_judgments):
    raw_metrics = metrics.get("raw_metrics", {})
    smoothness = raw_metrics.get("motion_smoothness", 1.0)
    return normalize_severity(smoothness, 0.6, 1.0, invert=True)

def temporal_jitter_evidence(metrics, task_type, vlm_judgments):
    raw_metrics = metrics.get("raw_metrics", {})
    return {
        "motion_amplitude": raw_metrics.get("motion_amplitude"),
        "motion_smoothness": raw_metrics.get("motion_smoothness")
    }

RULES.append(FailureRule(
    "temporal_jitter",
    temporal_jitter_condition,
    temporal_jitter_severity,
    temporal_jitter_evidence,
    "temporal_jitter_rule_v1"
))

def apply_rules(metrics, task_type, vlm_judgments=None):
    """Apply all rules and return list of triggered failure modes."""
    failures = []
    for rule in RULES:
        failure = rule.check(metrics, task_type, vlm_judgments)
        if failure:
            failures.append(failure)
    return failures
