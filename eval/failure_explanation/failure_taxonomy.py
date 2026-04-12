# failure_taxonomy.py
# Defines the failure taxonomy for Helios evaluation

FAILURE_CATEGORIES = {
    "action_failures": [
        "action_not_followed",
        "late_action_onset",
        "wrong_turn_direction",
        "insufficient_turn_magnitude",
        "overshoot_action",
        "doorway_not_entered"
    ],
    "scene_consistency_failures": [
        "scene_layout_drift",
        "object_persistence_failure",
        "lighting_texture_shift",
        "wrong_room_transition"
    ],
    "temporal_failures": [
        "temporal_jitter",
        "abrupt_scene_jump",
        "motion_discontinuity"
    ],
    "long_horizon_failures": [
        "progressive_scene_drift",
        "loop_closure_failure",
        "memory_loss"
    ]
}

ALL_FAILURE_LABELS = []
for category, labels in FAILURE_CATEGORIES.items():
    ALL_FAILURE_LABELS.extend(labels)

# Descriptions for each failure mode
FAILURE_DESCRIPTIONS = {
    "action_not_followed": "The commanded action is not performed at all.",
    "late_action_onset": "The action starts later than expected.",
    "wrong_turn_direction": "The turn direction does not match the command.",
    "insufficient_turn_magnitude": "The turn angle is too small.",
    "overshoot_action": "The action exceeds the required magnitude.",
    "doorway_not_entered": "The doorway is not successfully entered.",
    "scene_layout_drift": "The scene layout changes unexpectedly.",
    "object_persistence_failure": "Objects appear/disappear incorrectly.",
    "lighting_texture_shift": "Lighting or textures change inconsistently.",
    "wrong_room_transition": "Transition to wrong room or area.",
    "temporal_jitter": "Motion has unwanted shaking or jitter.",
    "abrupt_scene_jump": "Sudden discontinuous changes in scene.",
    "motion_discontinuity": "Motion flow breaks unexpectedly.",
    "progressive_scene_drift": "Scene accumulates drift over time.",
    "loop_closure_failure": "Loop does not return to start properly.",
    "memory_loss": "Model forgets earlier scene elements."
}

# Supported task types
SUPPORTED_TASK_TYPES = [
    "reconstruct_original",
    "immediate_turn",
    "doorway_entry",
    "loop",
    "minor_scene_change"
]