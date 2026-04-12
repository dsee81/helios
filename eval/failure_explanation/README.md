# Failure Explanation System

This module extends the Helios evaluation pipeline with Stage B: Failure Explanation, providing interpretable failure modes and supporting evidence on top of existing numeric metrics.

## Overview

The system implements a hybrid evaluator with three layers:

1. **Existing numeric metrics** - Raw scores from Helios evaluation
2. **Rule-based failure diagnosis** - Maps metric patterns to explicit failure labels
3. **Local VLM judge** - Structured semantic judgments for hard-to-measure aspects

## Key Features

- **Evidence-backed analysis** - Every failure mode includes supporting evidence
- **Task-aware diagnosis** - Different rules for different task types
- **VLM integration** - Local Qwen2.5-VL for semantic judgments
- **Extensible taxonomy** - Easy to add new failure modes and rules
- **Graceful fallbacks** - Works without VLM if unavailable

## Usage

```python
from failure_explanation import explain_failures

report = explain_failures(
    raw_metrics=raw_metrics,
    segment_metrics=segment_metrics,
    task_specific_metrics=task_specific_metrics,
    task_type="immediate_turn",
    prompt_text="Turn left immediately",
    video_path="path/to/video.mp4"  # or keyframe_paths dict
)

print(report["free_text_summary"])
for failure in report["failure_modes"]:
    print(f"{failure['label']}: {failure['severity']:.2f}")
```

## VLM Integration

To use VLM judgments, implement the `query_vlm` function in `vlm_judges.py`:

```python
def query_vlm(prompt: str, images: List[str] = None) -> str:
    # Your local Qwen2.5-VL inference code here
    # Return JSON string
    pass
```

## Supported Task Types

- `reconstruct_original`
- `immediate_turn`
- `doorway_entry`
- `loop`
- `minor_scene_change`

## Failure Taxonomy

### Action Failures
- `action_not_followed`
- `late_action_onset`
- `wrong_turn_direction`
- `insufficient_turn_magnitude`
- `overshoot_action`
- `doorway_not_entered`

### Scene Consistency Failures
- `scene_layout_drift`
- `object_persistence_failure`
- `lighting_texture_shift`
- `wrong_room_transition`

### Temporal Failures
- `temporal_jitter`
- `abrupt_scene_jump`
- `motion_discontinuity`

### Long Horizon Failures
- `progressive_scene_drift`
- `loop_closure_failure`
- `memory_loss`

## Output Format

The system returns a JSON-serializable dict with:

- `raw_metrics`: Original metrics
- `segment_metrics`: Segment-wise metrics
- `task_specific_metrics`: Task-specific signals
- `vlm_judgments`: VLM-derived judgments
- `failure_modes`: List of detected failures with severity and evidence
- `free_text_summary`: Short human-readable explanation

## Examples

See `examples.py` for test cases demonstrating:

- Late action onset
- Wrong turn direction
- Loop closure failure
- Progressive scene drift
- Doorway entry failure
- VLM-assisted diagnosis

## Integration with Existing Pipeline

This system builds on top of existing Helios metrics without replacing them. Add the failure explanation as a post-processing step after computing numeric scores.