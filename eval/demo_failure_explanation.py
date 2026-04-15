#!/usr/bin/env python3
"""
Integration example for the failure explanation system.

This shows how to use the failure explanation on top of existing Helios metrics.
You only need to pass raw metrics from your evaluation pipeline.
"""

import sys
import os
# Add the current directory to sys.path so we can import failure_explanation
sys.path.insert(0, os.path.dirname(__file__))

from failure_explanation import explain_failures

def demo_failure_explanation():
    """Demonstrate the failure explanation system with minimal input."""

    # Example metrics from standard Helios evaluation
    # This is what you get from running your eval pipeline
    raw_metrics = {
        "semantic": 0.82,
        "naturalness": 0.75,
        "motion_amplitude": 0.25,  # Low motion in early frames
        "motion_smoothness": 0.78,
        "aesthetic": 0.70,
        "drifting_aesthetic": 0.08,
        "drifting_semantic": 0.12,
        "drifting_naturalness": 0.09,
        "drifting_motion_smoothness": 0.05
    }

    # Run failure explanation with just the metrics and task type
    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="immediate_turn",
        prompt_text="Turn left immediately and walk forward"
        # video_path="path/to/generated_video.mp4"  # Optional, for VLM analysis
    )

    # Print results
    print("=== Helios Failure Explanation Report ===")
    print(f"Task: immediate_turn")
    print()

    print("Free-text Summary:")
    print(report["free_text_summary"])
    print()

    print("Detected Failure Modes:")
    for failure in report["failure_modes"]:
        print(f"  - {failure['label']} (severity: {failure['severity']:.2f})")
        print(f"    Rule: {failure['triggered_by']}")
        if failure['evidence']:
            print(f"    Evidence: {failure['evidence']}")
        print()

    print("Inferred Segment Metrics (derived from drifting metrics):")
    for segment in ["early", "middle", "late"]:
        print(f"  {segment}:")
        for metric, value in report["inferred_segment_metrics"][segment].items():
            print(f"    {metric}: {value:.2f}")
    print()

    print("VLM Judgments:")
    if report["vlm_judgments"].get("vlm_unavailable"):
        print("  VLM not used (no video path provided)")
    else:
        for key, value in report["vlm_judgments"].items():
            if key != "evidence":
                print(f"  {key}: {value}")
    print()

    print("Raw Metrics (input):")
    for key, value in raw_metrics.items():
        print(f"  {key}: {value:.2f}")

if __name__ == "__main__":
    demo_failure_explanation()