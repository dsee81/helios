from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import OmegaConf

from .bins import DEFAULT_TURN_BIN_ORDER


@dataclass
class TurnTIDataConfig:
    manifest_path: str = ""
    prompt_fallback: str = "A driving scene from a forward-facing camera."
    num_frames: Optional[int] = None
    height: int = 384
    width: int = 640
    cache_metadata: bool = True
    force_rebuild: bool = False
    strict_paths: bool = True


@dataclass
class TurnTIModelConfig:
    pretrained_model_name_or_path: str = "BestWishYsh/Helios-Base"
    revision: Optional[str] = None
    variant: Optional[str] = None
    scheduler_config_path: str = "scripts/accelerate_configs/scheduler_config.json"
    mixed_precision: str = "bf16"
    num_bins: int = 6
    num_vectors: int = 4
    init_scale: float = 0.05
    learnable_delta: bool = True
    bin_names: List[str] = field(default_factory=lambda: list(DEFAULT_TURN_BIN_ORDER))
    init_phrases: List[str] = field(
        default_factory=lambda: [
            "a driving scene with a gentle right turn",
            "a driving scene with a strong right turn",
            "a driving scene moving straight and stable",
            "a driving scene moving straight with slight wobble",
            "a driving scene with a gentle left turn",
            "a driving scene with a strong left turn",
        ]
    )
    negative_prompt: str = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
        "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
        "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
        "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )


@dataclass
class TurnTIGenerationConfig:
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
class TurnTILossConfig:
    reconstruction_weight: float = 1.0
    anchor_weight: float = 0.1
    neighbor_smoothness_weight: float = 0.05
    temporal_smoothness_weight: float = 0.05


@dataclass
class TurnTITrainConfig:
    output_dir: str = "outputs/turn_ti"
    logging_dir: str = "logs"
    seed: int = 43
    train_batch_size: int = 1
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    max_train_steps: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-4
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    save_every: int = 100
    max_eval_batches: int = 8
    allow_tf32: bool = False
    log_with: Optional[str] = None
    single_gpu_only: bool = True


@dataclass
class TurnTIInferenceConfig:
    output_path: str = "outputs/turn_ti/infer.mp4"
    fps: int = 24
    disable_bank: bool = False


@dataclass
class TurnTIConfig:
    data: TurnTIDataConfig = field(default_factory=TurnTIDataConfig)
    model: TurnTIModelConfig = field(default_factory=TurnTIModelConfig)
    generation: TurnTIGenerationConfig = field(default_factory=TurnTIGenerationConfig)
    loss: TurnTILossConfig = field(default_factory=TurnTILossConfig)
    train: TurnTITrainConfig = field(default_factory=TurnTITrainConfig)
    inference: TurnTIInferenceConfig = field(default_factory=TurnTIInferenceConfig)


def load_config(path: str) -> TurnTIConfig:
    schema = OmegaConf.structured(TurnTIConfig)
    loaded = OmegaConf.load(path)
    merged = OmegaConf.merge(schema, loaded)
    return OmegaConf.to_object(merged)
