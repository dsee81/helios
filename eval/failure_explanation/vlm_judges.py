# vlm_judges.py
# Local VLM judge integration for structured semantic diagnostics

import json
import os
import math
from typing import List, Dict, Any, Optional, Union

import torch
from PIL import Image

# Local VLM model location. Override this when the model lives outside /tmp.
LOCAL_VLM_MODEL_PATH = os.environ.get("HELIOS_LOCAL_VLM_MODEL_PATH", "/tmp/Qwen2.5-VL-7B-Instruct")
LOOP_CLOSURE_FRAME_STRIDE = 6

# Global cache for loaded model and processor
_VLM_MODEL = None
_VLM_PROCESSOR = None


def _load_vlm_model():
    global _VLM_MODEL, _VLM_PROCESSOR
    if _VLM_MODEL is not None and _VLM_PROCESSOR is not None:
        return _VLM_MODEL, _VLM_PROCESSOR

    try:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    except ImportError:
        raise RuntimeError("transformers is not installed")

    if not os.path.exists(LOCAL_VLM_MODEL_PATH):
        raise FileNotFoundError(
            f"Local VLM model path not found: {LOCAL_VLM_MODEL_PATH}"
        )

    processor = AutoProcessor.from_pretrained(
        LOCAL_VLM_MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        LOCAL_VLM_MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    _VLM_MODEL = model.eval()
    _VLM_PROCESSOR = processor
    return _VLM_MODEL, _VLM_PROCESSOR




def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _prepare_inputs(prompt: str, images: List[Union[str, Image.Image]] = None):
    model, processor = _load_vlm_model()
    image_inputs = []
    if images:
        for image in images:
            try:
                if isinstance(image, Image.Image):
                    image_inputs.append(image)
                elif isinstance(image, str):
                    image_inputs.append(_load_image(image))
            except Exception:
                continue

    content = []
    for image in image_inputs:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})

    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        return_tensors="pt",
        padding=True,
    )
    return model, processor, inputs


def query_vlm(prompt: str, images: List[Union[str, Image.Image]] = None) -> str:
    """
    Query the local Qwen2.5-VL model using a hardcoded local path.

    If the local model is not available, this falls back to a deterministic placeholder response.
    """
    try:
        model, processor, inputs = _prepare_inputs(prompt, images)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256)
        generated_ids = output[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return decoded
    except Exception as exc:
        fallback = {
            "error": str(exc),
            "confidence": 0.0
        }
        return json.dumps(fallback)


def _sample_frame_indices(total_frames: int, start: int, end: int, num_samples: int) -> List[int]:
    if total_frames <= 0 or start >= end:
        return []
    start = max(0, start)
    end = min(total_frames, end)
    if end - start <= 0:
        return []
    step = max(1.0, (end - start) / float(num_samples))
    indices = [min(total_frames - 1, int(start + step * i)) for i in range(num_samples)]
    return sorted(set(indices))


def _read_frame_indices(video_path: str, indices: List[int]) -> List[Image.Image]:
    try:
        import cv2
    except ImportError:
        return []

    if not video_path or not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []

    images: List[Image.Image] = []
    for frame_index in sorted(set(i for i in indices if i >= 0)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame))

    cap.release()
    return images


def sample_frames(video_path: str = None, keyframe_paths: Dict[str, List[str]] = None,
                 num_early: int = 5, num_middle: int = 5, num_late: int = 5) -> Dict[str, List[Image.Image]]:
    """
    Sample frames from video or use provided keyframes.

    Args:
        video_path: Path to video file (optional)
        keyframe_paths: Dict with keys 'early', 'middle', 'late' containing lists of image paths
        num_early/middle/late: Number of frames to sample per segment

    Returns:
        Dict with 'early', 'middle', 'late' keys containing lists of PIL Image objects
    """
    if keyframe_paths:
        sampled = {"early": [], "middle": [], "late": []}
        for segment in ["early", "middle", "late"]:
            for path in keyframe_paths.get(segment, []):
                try:
                    sampled[segment].append(_load_image(path))
                except Exception:
                    continue
        return sampled

    if not video_path:
        return {"early": [], "middle": [], "late": []}

    try:
        import cv2
    except ImportError:
        return {"early": [], "middle": [], "late": []}

    if not os.path.exists(video_path):
        return {"early": [], "middle": [], "late": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return {"early": [], "middle": [], "late": []}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return {"early": [], "middle": [], "late": []}

    early_end = max(1, total_frames // 3)
    middle_start = early_end
    middle_end = max(middle_start + 1, 2 * total_frames // 3)
    late_start = middle_end

    segment_indices = {
        "early": _sample_frame_indices(total_frames, 0, early_end, num_early),
        "middle": _sample_frame_indices(total_frames, middle_start, middle_end, num_middle),
        "late": _sample_frame_indices(total_frames, late_start, total_frames, num_late)
    }

    sampled_frames = {"early": [], "middle": [], "late": []}
    for segment, indices in segment_indices.items():
        for frame_index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = cap.read()
            if not success or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames[segment].append(Image.fromarray(frame))

    cap.release()
    return sampled_frames


def sample_frames_every_n(video_path: str, stride: int) -> List[Image.Image]:
    """
    Sample the whole video at a fixed frame stride.
    """
    if not video_path or stride <= 0:
        return []

    try:
        import cv2
    except ImportError:
        return []

    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        return []

    indices = list(range(0, total_frames, stride))
    if (total_frames - 1) not in indices:
        indices.append(total_frames - 1)
    return _read_frame_indices(video_path, indices)


def _image_similarity(a: Image.Image, b: Image.Image) -> float:
    try:
        import numpy as np
        from skimage.metrics import structural_similarity

        aa = np.asarray(a.convert("L").resize((128, 72)), dtype="float32") / 255.0
        bb = np.asarray(b.convert("L").resize((128, 72)), dtype="float32") / 255.0
        return max(0.0, min(1.0, float(structural_similarity(aa, bb, data_range=1.0))))
    except Exception:
        import numpy as np

        aa = np.asarray(a.convert("RGB").resize((96, 54)), dtype="float32") / 255.0
        bb = np.asarray(b.convert("RGB").resize((96, 54)), dtype="float32") / 255.0
        mse = float(((aa - bb) ** 2).mean())
        return max(0.0, min(1.0, 1.0 - math.sqrt(mse)))


def _cycle_revisit_metrics(frames: List[Image.Image]) -> Dict[str, float]:
    if len(frames) < 4:
        return {
            "best_revisit_similarity": 0.5,
            "mean_topk_revisit_similarity": 0.5,
            "revisit_near_end_score": 0.5,
            "cyclic_trajectory_score": 0.5,
            "seam_smoothness_score": 0.5,
        }

    first = frames[0]
    last = frames[-1]
    late_start = max(1, int(len(frames) * 0.65))
    late_frames = frames[late_start:]
    late_sims = [_image_similarity(first, frame) for frame in late_frames] or [_image_similarity(first, last)]
    sorted_sims = sorted(late_sims, reverse=True)
    topk = sorted_sims[: min(3, len(sorted_sims))]
    best_revisit = sorted_sims[0]
    mean_topk = sum(topk) / len(topk)

    endpoint = _image_similarity(first, last)
    seam_prev = _image_similarity(frames[-2], last)
    seam_next = endpoint
    seam = 0.5 * seam_prev + 0.5 * seam_next

    early_anchor = frames[min(1, len(frames) - 1)]
    late_anchor = frames[max(0, len(frames) - 2)]
    trajectory_return = _image_similarity(early_anchor, late_anchor)
    cyclic = 0.45 * mean_topk + 0.35 * trajectory_return + 0.20 * endpoint

    return {
        "best_revisit_similarity": float(best_revisit),
        "mean_topk_revisit_similarity": float(mean_topk),
        "revisit_near_end_score": float(0.65 * mean_topk + 0.35 * endpoint),
        "cyclic_trajectory_score": float(cyclic),
        "seam_smoothness_score": float(seam),
    }

def parse_vlm_response(response: str) -> Dict[str, Any]:
    """Parse VLM JSON response, with fallback for malformed output."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON-like content
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        return {"error": "Failed to parse VLM response", "raw_response": response}

def judge_action_adherence(prompt_text: str, task_type: str,
                          keyframe_paths: Dict[str, List[str]] = None,
                          video_path: str = None) -> Dict[str, Any]:
    """
    Judge action adherence using VLM.

    Returns structured judgment for action following.
    """
    if task_type not in ["immediate_turn", "doorway_entry"]:
        return {}

    frames = sample_frames(video_path, keyframe_paths)
    early_frames = frames.get("early", [])
    mid_frames = frames.get("middle", [])

    prompt = f"""
Analyze the video frames for action adherence to the instruction: "{prompt_text}"

Task type: {task_type}

Examine the early frames and middle frames.

Return JSON with:
- action_adherence_score: float 0-1
- action_started_early_enough: boolean
- observed_action_start_segment: "early"|"middle"|"late"
- observed_turn_direction: "left"|"right"|"straight"|null
- doorway_entered: boolean (if applicable)
- confidence: float 0-1
"""

    try:
        response = query_vlm(prompt, early_frames + mid_frames)
        result = parse_vlm_response(response)
        return result
    except Exception as e:
        return {"error": str(e), "action_adherence_score": 0.5, "confidence": 0.0}

def judge_scene_consistency(prompt_text: str, task_type: str,
                           keyframe_paths: Dict[str, List[str]] = None,
                           video_path: str = None) -> Dict[str, Any]:
    """
    Judge scene consistency using VLM.
    """
    frames = sample_frames(video_path, keyframe_paths)
    early_frames = frames.get("early", [])
    late_frames = frames.get("late", [])

    prompt = f"""
Analyze scene consistency in the video frames.

Instruction: "{prompt_text}"
Task type: {task_type}

Compare early frames vs late frames for:
- Layout consistency
- Object persistence
- Lighting/texture changes

Return JSON with:
- scene_consistency_score: float 0-1
- scene_drift_visible: boolean
- confidence: float 0-1
"""

    try:
        response = query_vlm(prompt, early_frames + late_frames)
        result = parse_vlm_response(response)
        return result
    except Exception as e:
        return {"error": str(e), "scene_consistency_score": 0.5, "confidence": 0.0}

def judge_doorway_entry(prompt_text: str, task_type: str,
                       keyframe_paths: Dict[str, List[str]] = None,
                       video_path: str = None) -> Dict[str, Any]:
    """
    Judge doorway entry success using VLM.
    """
    if task_type != "doorway_entry":
        return {}

    frames = sample_frames(video_path, keyframe_paths)
    all_frames = frames.get("early", []) + frames.get("middle", []) + frames.get("late", [])

    prompt = f"""
Analyze if the doorway was successfully entered.

Instruction: "{prompt_text}"

Look for:
- Camera movement through doorway
- Change in environment indicating entry
- Doorway visible and crossed

Return JSON with:
- doorway_entered: boolean
- confidence: float 0-1
"""

    try:
        response = query_vlm(prompt, all_frames)
        result = parse_vlm_response(response)
        return result
    except Exception as e:
        return {"error": str(e), "doorway_entered": False, "confidence": 0.0}

def judge_turn_direction_and_onset(prompt_text: str, task_type: str,
                                  keyframe_paths: Dict[str, List[str]] = None,
                                  video_path: str = None) -> Dict[str, Any]:
    """
    Judge turn direction and onset timing.
    """
    if task_type != "immediate_turn":
        return {}

    frames = sample_frames(video_path, keyframe_paths)
    early_frames = frames.get("early", [])

    prompt = f"""
Analyze turn direction and onset timing.

Instruction: "{prompt_text}"

Examine early frames for:
- Direction of turn (left/right)
- Whether turn starts immediately

Return JSON with:
- turn_direction_correct: boolean
- observed_direction: "left"|"right"|"straight"
- action_started_early_enough: boolean
- confidence: float 0-1
"""

    try:
        response = query_vlm(prompt, early_frames)
        result = parse_vlm_response(response)
        return result
    except Exception as e:
        return {"error": str(e), "turn_direction_correct": False, "confidence": 0.0}

def judge_progressive_drift(prompt_text: str, task_type: str,
                           keyframe_paths: Dict[str, List[str]] = None,
                           video_path: str = None) -> Dict[str, Any]:
    """
    Judge progressive scene drift over time.
    """
    frames = sample_frames(video_path, keyframe_paths)
    early_frames = frames.get("early", [])
    late_frames = frames.get("late", [])

    prompt = f"""
Analyze progressive scene drift.

Instruction: "{prompt_text}"
Task type: {task_type}

Compare early vs late frames for accumulation of drift.

Return JSON with:
- progressive_drift_score: float 0-1 (higher = more drift)
- drift_visible: boolean
- confidence: float 0-1
"""

    try:
        response = query_vlm(prompt, early_frames + late_frames)
        result = parse_vlm_response(response)
        return result
    except Exception as e:
        return {"error": str(e), "progressive_drift_score": 0.5, "confidence": 0.0}


def judge_loop_closure(prompt_text: str, task_type: str,
                      keyframe_paths: Dict[str, List[str]] = None,
                      video_path: str = None) -> Dict[str, Any]:
    """
    Judge whether a loop task returns to the starting viewpoint at the end.
    """
    if task_type != "loop":
        return {}

    timeline_frames = sample_frames_every_n(video_path, LOOP_CLOSURE_FRAME_STRIDE)
    frames = sample_frames(video_path, keyframe_paths)
    early_frames = frames.get("early", [])
    late_frames = frames.get("late", [])
    if not timeline_frames or not early_frames or not late_frames:
        return {
            "loop_closure_score": 0.5,
            "loop_closure_achieved": False,
            "endpoint_similarity": 0.5,
            "confidence": 0.0,
            "frame_stride": LOOP_CLOSURE_FRAME_STRIDE,
        }

    cycle_metrics = _cycle_revisit_metrics(timeline_frames)

    prompt = f"""
Analyze whether this video forms a true visual loop.

Instruction: "{prompt_text}"
Task type: {task_type}

You are shown a timeline of the video sampled at one frame every {LOOP_CLOSURE_FRAME_STRIDE} frames, in chronological order.
The earliest images are the start of the video and the latest images are the end.
Judge whether the end returns to the same viewpoint and scene composition as the start, and whether the whole motion reads as a coherent loop.

Focus on the full sampled timeline, not only the first and final frame:
- whether the path revisits the starting place/viewpoint near the end
- whether the trajectory feels cyclic instead of simply drifting forward
- whether the final segment can transition smoothly back to the first frame
- whether landmarks and scene layout remain consistent during the return

Return JSON with:
- loop_closure_score: float 0-1
- loop_closure_achieved: boolean
- endpoint_similarity: float 0-1
- cyclic_trajectory_score: float 0-1
- revisit_near_end_score: float 0-1
- seam_smoothness_score: float 0-1
- confidence: float 0-1
"""

    try:
        response = query_vlm(prompt, timeline_frames)
        result = parse_vlm_response(response)
        result.update({k: float(result.get(k, v)) for k, v in cycle_metrics.items()})
        endpoint_similarity = float(result.get("endpoint_similarity", _image_similarity(timeline_frames[0], timeline_frames[-1])))
        vlm_loop = float(result.get("loop_closure_score", endpoint_similarity))
        cyclic = float(result.get("cyclic_trajectory_score", cycle_metrics["cyclic_trajectory_score"]))
        revisit = float(result.get("revisit_near_end_score", cycle_metrics["revisit_near_end_score"]))
        seam = float(result.get("seam_smoothness_score", cycle_metrics["seam_smoothness_score"]))
        score = 0.45 * cyclic + 0.35 * max(revisit, cycle_metrics["best_revisit_similarity"]) + 0.20 * seam
        score = max(score, 0.5 * vlm_loop + 0.5 * score)
        result["loop_closure_score"] = float(max(0.0, min(1.0, score)))
        result["endpoint_similarity"] = endpoint_similarity
        result["frame_stride"] = LOOP_CLOSURE_FRAME_STRIDE
        if "loop_closure_achieved" not in result:
            result["loop_closure_achieved"] = score >= 0.72 and revisit >= 0.68 and seam >= 0.62
        return result
    except Exception as e:
        return {
            "error": str(e),
            "loop_closure_score": 0.5,
            "loop_closure_achieved": False,
            "endpoint_similarity": 0.5,
            "confidence": 0.0,
            "frame_stride": LOOP_CLOSURE_FRAME_STRIDE,
            **(_cycle_revisit_metrics(timeline_frames) if timeline_frames else {}),
        }

def get_vlm_judgments(prompt_text: str, task_type: str,
                     keyframe_paths: Dict[str, List[str]] = None,
                     video_path: str = None) -> Dict[str, Any]:
    """
    Get all relevant VLM judgments for the task.
    """
    judgments = {}

    try:
        judgments.update(judge_action_adherence(prompt_text, task_type, keyframe_paths, video_path))
    except:
        pass

    try:
        judgments.update(judge_scene_consistency(prompt_text, task_type, keyframe_paths, video_path))
    except:
        pass

    if task_type == "doorway_entry":
        try:
            judgments.update(judge_doorway_entry(prompt_text, task_type, keyframe_paths, video_path))
        except:
            pass

    if task_type == "immediate_turn":
        try:
            judgments.update(judge_turn_direction_and_onset(prompt_text, task_type, keyframe_paths, video_path))
        except:
            pass

    if task_type == "loop":
        try:
            judgments.update(judge_loop_closure(prompt_text, task_type, keyframe_paths, video_path))
        except:
            pass

    try:
        judgments.update(judge_progressive_drift(prompt_text, task_type, keyframe_paths, video_path))
    except:
        pass

    # Aggregate scores
    action_score = judgments.get("action_adherence_score", 0.5)
    scene_score = judgments.get("scene_consistency_score", 0.5)
    temporal_score = 1.0 - judgments.get("progressive_drift_score", 0.5)

    judgments["action_adherence_score"] = action_score
    judgments["scene_consistency_score"] = scene_score
    judgments["temporal_coherence_score"] = temporal_score
    judgments["used_local_vlm"] = True
    judgments["vlm_model_path"] = LOCAL_VLM_MODEL_PATH

    # Judged failure modes from VLM
    judged_failures = []
    if judgments.get("action_started_early_enough") == False:
        judged_failures.append("late_action_onset")
    if judgments.get("turn_direction_correct") == False:
        judged_failures.append("wrong_turn_direction")
    if judgments.get("doorway_entered") == False:
        judged_failures.append("doorway_not_entered")
    if judgments.get("scene_drift_visible") == True:
        judged_failures.append("progressive_scene_drift")
    if judgments.get("loop_closure_achieved") == False:
        judged_failures.append("loop_closure_failure")

    judgments["judged_failure_modes"] = judged_failures

    # Evidence
    judgments["evidence"] = {
        "observed_action_start_segment": judgments.get("observed_action_start_segment"),
        "observed_turn_direction": judgments.get("observed_turn_direction"),
        "doorway_entry_success": judgments.get("doorway_entered"),
        "endpoint_similarity": judgments.get("endpoint_similarity"),
        "loop_closure_score": judgments.get("loop_closure_score"),
        "cyclic_trajectory_score": judgments.get("cyclic_trajectory_score"),
        "revisit_near_end_score": judgments.get("revisit_near_end_score"),
        "seam_smoothness_score": judgments.get("seam_smoothness_score"),
        "best_revisit_similarity": judgments.get("best_revisit_similarity"),
        "mean_topk_revisit_similarity": judgments.get("mean_topk_revisit_similarity"),
        "loop_frame_stride": judgments.get("frame_stride"),
    }

    return judgments
