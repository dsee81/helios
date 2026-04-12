# examples.py
# Usage examples and test cases for failure explanation system

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from failure_explanation import explain_failures

def example_late_turn():
    """Example of late action onset failure.
    
    Signals: Low motion amplitude early, high drifting metrics indicate action started late.
    """
    raw_metrics = {
        "semantic": 0.85,
        "naturalness": 0.78,
        "motion_amplitude": 0.3,  # Low early motion = late action
        "motion_smoothness": 0.82,
        "aesthetic": 0.75,
        "drifting_semantic": 0.12,
        "drifting_naturalness": 0.08,
        "drifting_aesthetic": 0.10,
        "drifting_motion_smoothness": 0.05
    }

    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="immediate_turn",
        prompt_text="Turn left immediately"
    )

    print("Late Turn Example:")
    print(f"Summary: {report['free_text_summary']}")
    print(f"Failures: {[f['label'] for f in report['failure_modes']]}")
    return report

def example_wrong_turn_direction():
    """Example of wrong turn direction failure.
    
    Proper turn starts on time but direction is wrong.
    """
    raw_metrics = {
        "semantic": 0.75,
        "naturalness": 0.70,
        "motion_amplitude": 0.75,
        "motion_smoothness": 0.80,
        "aesthetic": 0.68,
        "drifting_semantic": 0.08,
        "drifting_naturalness": 0.07,
        "drifting_aesthetic": 0.06,
        "drifting_motion_smoothness": 0.04
    }

    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="immediate_turn",
        prompt_text="Turn left immediately"
    )

    print("Wrong Turn Direction Example:")
    print(f"Summary: {report['free_text_summary']}")
    print(f"Failures: {[f['label'] for f in report['failure_modes']]}")
    return report

def example_loop_closure_failure():
    """Example of loop closure failure.
    
    High drifting metrics + low consistency scores indicate loop was not closed properly.
    """
    raw_metrics = {
        "semantic": 0.80,
        "naturalness": 0.75,
        "motion_amplitude": 0.65,
        "motion_smoothness": 0.70,
        "aesthetic": 0.72,
        "drifting_semantic": 0.35,  # High drift = bad loop closure
        "drifting_naturalness": 0.28,
        "drifting_aesthetic": 0.25,
        "drifting_motion_smoothness": 0.20
    }

    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="loop",
        prompt_text="Walk in a loop and return to start"
    )

    print("Loop Closure Failure Example:")
    print(f"Summary: {report['free_text_summary']}")
    print(f"Failures: {[f['label'] for f in report['failure_modes']]}")
    return report

def example_progressive_scene_drift():
    """Example of progressive scene drift.
    
    High drifting metrics indicate scene consistency degrades over time.
    """
    raw_metrics = {
        "semantic": 0.70,
        "naturalness": 0.65,
        "motion_amplitude": 0.55,
        "motion_smoothness": 0.68,
        "aesthetic": 0.62,
        "drifting_semantic": 0.28,  # High drift in semantic consistency
        "drifting_naturalness": 0.22,
        "drifting_aesthetic": 0.30,  # High drift in aesthetic
        "drifting_motion_smoothness": 0.15
    }

    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="minor_scene_change",
        prompt_text="Walk forward maintaining the scene"
    )

    print("Progressive Scene Drift Example:")
    print(f"Summary: {report['free_text_summary']}")
    print(f"Failures: {[f['label'] for f in report['failure_modes']]}")
    return report

def example_doorway_not_entered():
    """Example of doorway not entered failure.
    
    Action-related metrics are low, and consistent low semantic scores indicate scene wasn't entered.
    """
    raw_metrics = {
        "semantic": 0.60,  # Low semantic consistency
        "naturalness": 0.58,
        "motion_amplitude": 0.45,  # Low motion = insufficient movement
        "motion_smoothness": 0.65,
        "aesthetic": 0.55,
        "drifting_semantic": 0.25,
        "drifting_naturalness": 0.20,
        "drifting_aesthetic": 0.18,
        "drifting_motion_smoothness": 0.10
    }

    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="doorway_entry",
        prompt_text="Walk through the doorway"
    )

    print("Doorway Not Entered Example:")
    print(f"Summary: {report['free_text_summary']}")
    print(f"Failures: {[f['label'] for f in report['failure_modes']]}")
    return report

def example_good_reconstruction():
    """Example of good reconstruction with minimal failures."""
    raw_metrics = {
        "semantic": 0.92,
        "naturalness": 0.88,
        "motion_amplitude": 0.70,
        "motion_smoothness": 0.85,
        "aesthetic": 0.86,
        "drifting_semantic": 0.05,  # Low drift = good consistency
        "drifting_naturalness": 0.04,
        "drifting_aesthetic": 0.06,
        "drifting_motion_smoothness": 0.03
    }

    report = explain_failures(
        raw_metrics=raw_metrics,
        task_type="reconstruct_original",
        prompt_text="A person walking in a park"
    )

    print("Good Reconstruction Example:")
    print(f"Summary: {report['free_text_summary']}")
    print(f"Failures: {[f['label'] for f in report['failure_modes']]}")
    return report

if __name__ == "__main__":
    # Run examples
    example_late_turn()
    print()
    example_wrong_turn_direction()
    print()
    example_loop_closure_failure()
    print()
    example_progressive_scene_drift()
    print()
    example_doorway_not_entered()
    print()
    example_good_reconstruction()