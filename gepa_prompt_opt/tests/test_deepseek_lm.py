import unittest
from pathlib import Path
import itertools

from gepa_prompt_opt.deepseek_lm import _normalize_model_name
from gepa_prompt_opt.gepa_driver import OptimizeConfig, normalize_config
from gepa_prompt_opt.io_utils import sha256_json


class TestDeepSeekLM(unittest.TestCase):
    def test_normalize_model_name(self):
        self.assertEqual(_normalize_model_name("deepseek/deepseek-chat"), "deepseek-chat")
        self.assertEqual(_normalize_model_name("deepseek-chat"), "deepseek-chat")
        self.assertEqual(_normalize_model_name(""), "deepseek-chat")

    def test_normalize_config_preserves_split_and_disable_vlm(self):
        repo_root = Path(__file__).resolve().parents[2]
        cfg = OptimizeConfig(
            repo_root=repo_root,
            work_dir=Path("runs/x"),
            dataset_manifest=Path("gepa_prompt_opt/examples/loop_dataset_example.json"),
            seed_template=Path("gepa_prompt_opt/examples/seed_loop_template.json"),
            eval_split="dev",
            disable_vlm=True,
            run_naturalness=False,
            fps=30,
        )
        out = normalize_config(cfg)
        self.assertEqual(out.eval_split, "dev")
        self.assertTrue(out.disable_vlm)
        self.assertFalse(out.run_naturalness)
        self.assertEqual(out.fps, 30)

    def test_candidate_names_are_unique_per_evaluation_attempt(self):
        template = {"components": {"a": "x"}}
        counter = itertools.count()
        name1 = f"eval_{next(counter):04d}_cand_{sha256_json(template)[:16]}"
        name2 = f"eval_{next(counter):04d}_cand_{sha256_json(template)[:16]}"
        self.assertNotEqual(name1, name2)
        self.assertTrue(name1.endswith(name2.split("_cand_")[1]))


if __name__ == "__main__":
    unittest.main()
