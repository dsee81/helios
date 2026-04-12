import unittest
from pathlib import Path

from gepa_prompt_opt.helios_infer import HeliosV2VConfig, build_infer_command


class TestInferParse(unittest.TestCase):
    def test_build_infer_command(self):
        repo_root = Path(__file__).resolve().parents[2]
        script = repo_root / "scripts" / "inference" / "helios-distilled_v2v.sh"
        cfg = HeliosV2VConfig(repo_root=repo_root, inference_script_path=script)
        env, argv = build_infer_command(
            cfg,
            input_video_path=repo_root / "example" / "car.mp4",
            prompt="hello",
            output_folder=repo_root / "output_helios" / "tmp_test",
            num_frames=240,
        )
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertIn("--video_path", argv)
        self.assertIn("--prompt", argv)
        self.assertIn("--output_folder", argv)


if __name__ == "__main__":
    unittest.main()

