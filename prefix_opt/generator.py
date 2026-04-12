from __future__ import annotations

import json
from typing import Optional

import torch


def flatten_video_latent_sections(video_latent_sections: torch.Tensor) -> torch.Tensor:
    if video_latent_sections.ndim != 6:
        raise ValueError(
            f"Expected latent sections with shape [B, S, C, T, H, W], got {tuple(video_latent_sections.shape)}"
        )
    batch, sections, channels, frames, height, width = video_latent_sections.shape
    return video_latent_sections.permute(0, 2, 1, 3, 4, 5).contiguous().view(batch, channels, sections * frames, height, width)


class HeliosPrefixV2VGenerator:
    def __init__(self, cfg):
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
        from transformers import AutoTokenizer, UMT5EncoderModel

        from helios.modules.transformer_helios import HeliosTransformer3DModel
        from helios.pipelines.pipeline_helios import HeliosPipeline

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=cfg.model.revision,
        )

        weight_dtype = torch.float32
        if cfg.model.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif cfg.model.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.weight_dtype = weight_dtype

        self.text_encoder = UMT5EncoderModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=cfg.model.revision,
            variant=cfg.model.variant,
            dtype=weight_dtype,
        )
        self.vae = AutoencoderKLWan.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="vae",
            revision=cfg.model.revision,
            variant=cfg.model.variant,
            torch_dtype=torch.float32,
        )
        self.transformer = HeliosTransformer3DModel.from_pretrained(
            cfg.model.pretrained_model_name_or_path,
            subfolder="transformer",
        )
        with open(cfg.model.scheduler_config_path, "r", encoding="utf-8") as handle:
            scheduler_config = json.load(handle)
        self.scheduler = UniPCMultistepScheduler.from_config(scheduler_config)
        self.pipeline = HeliosPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            scheduler=self.scheduler,
            transformer=self.transformer,
        )
        self.pipeline.set_progress_bar_config(disable=True)
        self.freeze_models()

    def freeze_models(self) -> None:
        for module in [self.text_encoder, self.vae, self.transformer]:
            module.requires_grad_(False)
            module.eval()

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

    def generate_training_video(
        self,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        video_latent_sections: torch.Tensor,
        num_generation_sections: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.generation
        batch_size = video_latent_sections.shape[0]
        prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        negative_prompt_embeds = (
            negative_prompt_embeds.to(self.transformer.dtype) if negative_prompt_embeds is not None else None
        )
        video_latents = flatten_video_latent_sections(video_latent_sections).to(device=self.device, dtype=torch.float32)
        image_latents = self._prepare_image_latents_proxy(video_latents)
        latents_mean, latents_std = self._latents_mean_std()

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
        decode_input = generated_latents.to(self.vae.dtype) / latents_std + latents_mean
        generated_video = self.vae.decode(decode_input, return_dict=False)[0]
        return generated_latents, generated_video

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
