from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from omegaconf import OmegaConf


@dataclass
class PrefixDataConfig:
    action_dirs: Dict[str, str] = field(default_factory=dict)
    latent_root: str = ""
    csv_root: str = ""
    prompt_fallback: str = "A driving scene from a forward-facing camera."
    video_glob: str = "**/*.mp4"
    latent_glob: str = "**/*.pt"
    csv_glob: str = "**/*.csv"
    num_frames: Optional[int] = None
    height: int = 384
    width: int = 640
    cache_metadata: bool = True
    force_rebuild: bool = False
    v_stop_thresh: float = 0.05
    w_stop_thresh: float = 0.05
    velocity_column: str = "v_calculated"
    yaw_rate_column: str = "w_calculated"


@dataclass
class PrefixModelConfig:
    pretrained_model_name_or_path: str = "BestWishYsh/Helios-Base"
    revision: Optional[str] = None
    variant: Optional[str] = None
    scheduler_config_path: str = "scripts/accelerate_configs/scheduler_config.json"
    mixed_precision: str = "bf16"
    prefix_length: int = 8
    prefix_init_std: float = 0.02
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
        "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
        "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )


@dataclass
class PrefixGenerationConfig:
    history_sizes: list[int] = field(default_factory=lambda: [16, 2, 1])
    latent_window_size: int = 9
    num_inference_steps: int = 20
    guidance_scale: float = 5.0
    use_dynamic_shifting: bool = False
    time_shift_type: str = "linear"
    is_keep_x0: bool = True
    is_enable_stage2: bool = False
    stage2_num_stages: int = 3
    stage2_num_inference_steps_list: list[int] = field(default_factory=lambda: [10, 10, 10])
    scheduler_type: str = "unipc"
    num_generation_sections: int = 1
    add_noise_to_video_latents: bool = True
    video_noise_sigma_min: float = 0.111
    video_noise_sigma_max: float = 0.135
    use_cfg_zero_star: bool = False
    use_zero_init: bool = True
    zero_steps: int = 1
    use_kv_cache: bool = False


@dataclass
class PrefixLossConfig:
    velocity_scale: float = 0.5
    yaw_scale: float = 0.4
    velocity_weight: float = 1.0
    yaw_weight: float = 1.0
    direction_weight: float = 0.5
    action_ce_weight: float = 0.5
    source_consistency_weight: float = 0.25
    source_motion_weight: float = 0.25
    temporal_smoothness_weight: float = 0.1
    drift_weight: float = 0.05
    use_non_diff_metrics: bool = True
    non_diff_metrics_every: int = 20


@dataclass
class PrefixTrainConfig:
    output_dir: str = "outputs/prefix_opt"
    logging_dir: str = "logs"
    seed: int = 43
    train_batch_size: int = 1
    eval_batch_size: int = 1
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    max_train_steps: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    save_every: int = 100
    eval_every: int = 100
    max_eval_batches: int = 8
    gradient_check_warmup_steps: int = 5
    zero_grad_tolerance_steps: int = 3
    allow_tf32: bool = False
    gradient_checkpointing: bool = False
    log_with: Optional[str] = None


@dataclass
class PrefixInferenceConfig:
    output_path: str = "outputs/prefix_opt/infer.mp4"
    fps: int = 24
    disable_prefix: bool = False


@dataclass
class PrefixOptConfig:
    data: PrefixDataConfig = field(default_factory=PrefixDataConfig)
    model: PrefixModelConfig = field(default_factory=PrefixModelConfig)
    generation: PrefixGenerationConfig = field(default_factory=PrefixGenerationConfig)
    loss: PrefixLossConfig = field(default_factory=PrefixLossConfig)
    train: PrefixTrainConfig = field(default_factory=PrefixTrainConfig)
    inference: PrefixInferenceConfig = field(default_factory=PrefixInferenceConfig)


def load_config(path: str) -> PrefixOptConfig:
    schema = OmegaConf.structured(PrefixOptConfig)
    loaded = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, loaded)
    return OmegaConf.to_object(merged)

