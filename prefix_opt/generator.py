from __future__ import annotations

import json
import os
import time
from typing import Optional

import torch


def flatten_video_latent_sections(video_latent_sections: torch.Tensor) -> torch.Tensor:
    if video_latent_sections.ndim != 6:
        raise ValueError(
            f"Expected latent sections with shape [B, S, C, T, H, W], got {tuple(video_latent_sections.shape)}"
        )
    batch, sections, channels, frames, height, width = video_latent_sections.shape
    return video_latent_sections.permute(0, 2, 1, 3, 4, 5).contiguous().view(batch, channels, sections * frames, height, width)


def split_history_target_sections(
    video_latent_sections: torch.Tensor,
    num_generation_sections: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_latent_sections.ndim != 6:
        raise ValueError(
            f"Expected latent sections with shape [B, S, C, T, H, W], got {tuple(video_latent_sections.shape)}"
        )
    total_sections = video_latent_sections.shape[1]
    if num_generation_sections <= 0:
        raise ValueError(f"num_generation_sections must be positive, got {num_generation_sections}")
    if total_sections <= num_generation_sections:
        raise ValueError(
            f"Need more latent sections than generation targets, got total_sections={total_sections} "
            f"and num_generation_sections={num_generation_sections}"
        )
    history_sections = video_latent_sections[:, : total_sections - num_generation_sections].contiguous()
    target_sections = video_latent_sections[:, total_sections - num_generation_sections :].contiguous()
    return history_sections, target_sections


class HeliosPrefixV2VGenerator:
    def __init__(self, cfg):
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from transformers import AutoTokenizer, UMT5EncoderModel

        from helios.modules.transformer_helios import HeliosTransformer3DModel
        from helios.pipelines.pipeline_helios import HeliosPipeline

        def stage(msg: str) -> None:
            print(f"[HeliosPrefixV2VGenerator] {msg}", flush=True)

        self.cfg = cfg
        t0 = time.perf_counter()
        stage("tokenizer_start")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=cfg.model.revision,
        )
        stage(f"tokenizer_ready {time.perf_counter() - t0:.2f}s")

        weight_dtype = torch.float32
        if cfg.model.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif cfg.model.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        t0 = time.perf_counter()
        stage("text_encoder_start")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=cfg.model.revision,
            variant=cfg.model.variant,
            dtype=weight_dtype,
        )
        stage(f"text_encoder_ready {time.perf_counter() - t0:.2f}s")
        t0 = time.perf_counter()
        stage("vae_start")
        self.vae = AutoencoderKLWan.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="vae",
            revision=cfg.model.revision,
            variant=cfg.model.variant,
            torch_dtype=torch.float32,
        )
        stage(f"vae_ready {time.perf_counter() - t0:.2f}s")
        transformer_load_kwargs = {
            "subfolder": "transformer",
            "torch_dtype": weight_dtype,
            "low_cpu_mem_usage": True,
            "device_map": "cpu",
        }
        # Mirror the proven inference path when loading from a local checkpoint,
        # while keeping a fallback to the older behavior for compatibility.
        if os.path.isdir(str(cfg.model.pretrained_model_name_or_path)):
            transformer_load_kwargs["local_files_only"] = True
        t0 = time.perf_counter()
        stage("transformer_start")
        try:
            self.transformer = HeliosTransformer3DModel.from_pretrained(
                cfg.model.pretrained_model_name_or_path,
                **transformer_load_kwargs,
            )
        except TypeError:
            transformer_load_kwargs.pop("local_files_only", None)
            self.transformer = HeliosTransformer3DModel.from_pretrained(
                cfg.model.pretrained_model_name_or_path,
                **transformer_load_kwargs,
            )
        except Exception:
            transformer_load_kwargs.pop("local_files_only", None)
            transformer_load_kwargs.pop("use_default_loader", None)
            self.transformer = HeliosTransformer3DModel.from_pretrained(
                cfg.model.pretrained_model_name_or_path,
                **transformer_load_kwargs,
            )
        stage(f"transformer_ready {time.perf_counter() - t0:.2f}s")
        t0 = time.perf_counter()
        stage("transformer_backend_start")
        self._configure_transformer_attention_backend()
        stage(f"transformer_backend_ready {time.perf_counter() - t0:.2f}s")
        with open(cfg.model.scheduler_config_path, "r", encoding="utf-8") as handle:
            scheduler_config = json.load(handle)
        stage("scheduler_config_ready")
        t0 = time.perf_counter()
        stage("scheduler_start")
        self.scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
        stage(f"scheduler_ready {time.perf_counter() - t0:.2f}s")
        t0 = time.perf_counter()
        stage("pipeline_start")
        self.pipeline = HeliosPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            transformer=self.transformer,
        )
        if not hasattr(self.pipeline, "_interrupt"):
            self.pipeline._interrupt = False
        stage(f"pipeline_ready {time.perf_counter() - t0:.2f}s")
        self.pipeline.set_progress_bar_config(disable=True)
        stage("pipeline_progress_bar_ready")
        t0 = time.perf_counter()
        stage("freeze_models_start")
        self.freeze_models()
        stage(f"freeze_models_ready {time.perf_counter() - t0:.2f}s")

    def _configure_transformer_attention_backend(self) -> None:
        if not hasattr(self.transformer, "set_attention_backend") or not torch.cuda.is_available():
            return
        requested_backend = os.environ.get("HELIOS_ATTENTION_BACKEND", "").strip().lower()
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        try:
            if requested_backend:
                self.transformer.set_attention_backend(requested_backend)
            elif (cuda_major, cuda_minor) >= (10, 0):
                self.transformer.set_attention_backend("native")
            elif cuda_major >= 9:
                try:
                    self.transformer.set_attention_backend("_flash_3_hub")
                except Exception:
                    self.transformer.set_attention_backend("flash_hub")
            else:
                self.transformer.set_attention_backend("flash_hub")
        except Exception:
            # Preserve old functionality if backend selection fails on a given host.
            return

    def freeze_models(self) -> None:
        for module in [self.text_encoder, self.vae, self.transformer]:
            module.requires_grad_(False)
            module.eval()
        if getattr(self.cfg.train, "gradient_checkpointing", False) and hasattr(
            self.transformer, "enable_gradient_checkpointing"
        ):
            self.transformer.enable_gradient_checkpointing()

    def offload_text_encoder(self) -> None:
        self.text_encoder.to("cpu")
        if hasattr(self.pipeline, "text_encoder"):
            self.pipeline.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def to(self, device: torch.device) -> "HeliosPrefixV2VGenerator":
        self.text_encoder.to(device)
        self.vae.to(device)
        self.transformer.to(device, dtype=self.weight_dtype)
        self.pipeline.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    def encode_prompt_text(self, prompts: list[str]) -> torch.Tensor:
        prompt_embeds, _, _, _ = self.pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=self.device,
            dtype=self.weight_dtype,
        )
        return prompt_embeds.to(self.transformer.dtype)

    def _prepare_image_latents_proxy(self, video_latents: torch.Tensor) -> torch.Tensor:
        return video_latents[:, :, :1].contiguous()

    def _latents_mean_std(self):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.vae.device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            self.vae.device, self.vae.dtype
        )
        return latents_mean, latents_std

    def _sync_scheduler_device_state(self, device: torch.device) -> None:
        if hasattr(self.scheduler, "timesteps") and isinstance(self.scheduler.timesteps, torch.Tensor):
            self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        if hasattr(self.scheduler, "sigmas") and isinstance(self.scheduler.sigmas, torch.Tensor):
            self.scheduler.sigmas = self.scheduler.sigmas.to(device)

    def generate_training_video(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        video_latent_sections: torch.Tensor,
        num_generation_sections: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generated_latents = self.generate_training_latents(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video_latent_sections=video_latent_sections,
            num_generation_sections=num_generation_sections,
        )
        latents_mean, latents_std = self._latents_mean_std()
        decode_input = generated_latents.to(self.vae.dtype) / latents_std + latents_mean
        generated_video = self.vae.decode(decode_input, return_dict=False)[0]
        return generated_latents, generated_video

    def generate_training_latents(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        video_latent_sections: torch.Tensor,
        num_generation_sections: int,
    ) -> torch.Tensor:
        cfg = self.cfg.generation
        batch_size = video_latent_sections.shape[0]
        prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        negative_prompt_embeds = (
            negative_prompt_embeds.to(self.transformer.dtype) if negative_prompt_embeds is not None else None
        )
        video_latents = flatten_video_latent_sections(video_latent_sections).to(device=self.device, dtype=torch.float32)
        image_latents = self._prepare_image_latents_proxy(video_latents)
        # `stage1_sample()` reads several pipeline-private fields that are
        # normally initialized in `HeliosPipeline.__call__`, but this training
        # path invokes the stage sampler directly.
        self.pipeline._guidance_scale = cfg.guidance_scale
        self.pipeline._attention_kwargs = None
        self.pipeline._current_timestep = None
        self.pipeline._interrupt = False
        self.pipeline._num_timesteps = cfg.num_inference_steps

        history_sizes = sorted(cfg.history_sizes, reverse=True)
        latent_window_size = cfg.latent_window_size
        num_channels_latents = self.transformer.config.in_channels
        height = video_latents.shape[-2] * self.pipeline.vae_scale_factor_spatial
        width = video_latents.shape[-1] * self.pipeline.vae_scale_factor_spatial
        window_num_frames = (latent_window_size - 1) * self.pipeline.vae_scale_factor_temporal + 1

        history_latents = self._pad_history_latents(video_latents, sum(history_sizes))
        generated_sections = []
        for _ in range(num_generation_sections):
            indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]), device=self.device)
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
            indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

            latents_prefix = image_latents
            latents_history_long, latents_history_mid, latents_history_1x = history_latents[:, :, -sum(history_sizes) :].split(
                history_sizes, dim=2
            )
            latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)

            latents = self.pipeline.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=window_num_frames,
                dtype=torch.float32,
                device=self.device,
                generator=None,
                latents=None,
            )

            self.scheduler.set_timesteps(cfg.num_inference_steps, mu=1, device=self.device)
            self._sync_scheduler_device_state(self.device)
            timesteps = self.scheduler.timesteps

            latents = self.pipeline.stage1_sample(
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                timesteps=timesteps,
                guidance_scale=cfg.guidance_scale,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                attention_kwargs=None,
                device=self.device,
                transformer_dtype=self.transformer.dtype,
                generator=None,
                use_cfg_zero_star=cfg.use_cfg_zero_star,
                use_zero_init=cfg.use_zero_init,
                zero_steps=cfg.zero_steps,
                use_dmd=False,
                dmd_sigmas=None,
                dmd_timesteps=None,
                is_amplify_first_chunk=False,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["latents"],
                progress_bar=_NullProgressBar(),
            )
            generated_sections.append(latents)
            history_latents = torch.cat([history_latents, latents], dim=2)

        generated_latents = torch.cat(generated_sections, dim=2)
        return generated_latents

    @torch.no_grad()
    def run_inference(self, prompt_embeds: torch.Tensor, negative_prompt_embeds: Optional[torch.Tensor], video_latent_sections: torch.Tensor):
        _, generated_video = self.generate_training_video(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video_latent_sections=video_latent_sections,
            num_generation_sections=self.cfg.generation.num_generation_sections,
        )
        return generated_video

    def _pad_history_latents(self, history_latents: torch.Tensor, required_frames: int) -> torch.Tensor:
        if history_latents.shape[2] >= required_frames:
            return history_latents
        pad_frames = required_frames - history_latents.shape[2]
        padding = torch.zeros(
            history_latents.shape[0],
            history_latents.shape[1],
            pad_frames,
            history_latents.shape[3],
            history_latents.shape[4],
            device=history_latents.device,
            dtype=history_latents.dtype,
        )
        return torch.cat([padding, history_latents], dim=2)


class _NullProgressBar:
    def update(self, *args, **kwargs):
        return None
