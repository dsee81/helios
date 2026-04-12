import unittest

import pandas as pd
import torch

from prefix_opt.actions import ACTION_TO_ID
from prefix_opt.conditioning import ActionPrefixBank, build_conditioned_prompt_embeds
from prefix_opt.dataset import infer_action_id_from_motion
from prefix_opt.generator import flatten_video_latent_sections


class PrefixOptTests(unittest.TestCase):
    def test_stop_inference_matches_thresholds(self):
        motion_df = pd.DataFrame({"v_calculated": [0.001, 0.002, 0.01], "w_calculated": [0.001, 0.002, 0.01]})
        action_id = infer_action_id_from_motion("w", motion_df, "v_calculated", "w_calculated", 0.05, 0.05)
        self.assertEqual(action_id, ACTION_TO_ID["stop"])

    def test_prefix_concat_extends_sequence(self):
        prefix_bank = ActionPrefixBank(prefix_length=4, hidden_size=8, init_std=0.01)
        prompt_embeds = torch.randn(2, 6, 8)
        action_ids = torch.tensor([0, 3])
        conditioned = build_conditioned_prompt_embeds(prompt_embeds, action_ids, prefix_bank)
        self.assertEqual(tuple(conditioned.prompt_embeds.shape), (2, 10, 8))

    def test_gradient_reaches_prefix_only(self):
        prefix_bank = ActionPrefixBank(prefix_length=2, hidden_size=4, init_std=0.01)
        frozen = torch.nn.Linear(4, 1)
        frozen.requires_grad_(False)
        prompt_embeds = torch.randn(1, 3, 4)
        conditioned = build_conditioned_prompt_embeds(prompt_embeds, torch.tensor([1]), prefix_bank)
        output = frozen(conditioned.prompt_embeds).sum()
        output.backward()
        self.assertIsNotNone(prefix_bank.prefix.grad)
        self.assertGreater(prefix_bank.prefix.grad.abs().sum().item(), 0.0)
        self.assertTrue(all(param.grad is None for param in frozen.parameters()))

    def test_flatten_video_latents(self):
        sections = torch.randn(2, 3, 16, 9, 8, 8)
        flattened = flatten_video_latent_sections(sections)
        self.assertEqual(tuple(flattened.shape), (2, 16, 27, 8, 8))


if __name__ == "__main__":
    unittest.main()
