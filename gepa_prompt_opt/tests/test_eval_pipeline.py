import tempfile
import unittest
from pathlib import Path

from gepa_prompt_opt.eval_pipeline import EvalPipelineConfig, run_eval_pipeline


class TestEvalPipeline(unittest.TestCase):
    def test_run_eval_pipeline_passes_env_and_finds_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            eval_dir = root / "eval"
            eval_dir.mkdir()
            script = eval_dir / "fake_run_metrics.sh"
            script.write_text(
                "#!/bin/bash\n"
                "set -euo pipefail\n"
                "mkdir -p \"$VIDEO_DIR\"\n"
                "mkdir -p \"$BASE_OUTPUT_DIR/$EXPERIMENT_NAME/$(basename \"$VIDEO_DIR\")\"\n"
                "test \"$INPUT_CSV\" = \"" + str((root / "input.csv").resolve()) + "\"\n"
                "test \"$TASK_TYPE\" = \"loop\"\n"
                "test \"$VIDEO_PATH_COLUMN\" = \"video_path\"\n"
                "test \"$DISABLE_VLM\" = \"1\"\n"
                "test \"$RUN_NATURALNESS\" = \"0\"\n"
                "printf '{}' > \"$BASE_OUTPUT_DIR/$EXPERIMENT_NAME/$(basename \"$VIDEO_DIR\")/combined_video_report.json\"\n",
                encoding="utf-8",
            )

            input_csv = root / "input.csv"
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "x.mp4").write_text("", encoding="utf-8")
            input_csv.write_text(f"id,prompt,video_path\n1,hello,{(video_dir / 'x.mp4').resolve()}\n", encoding="utf-8")

            out = run_eval_pipeline(
                EvalPipelineConfig(
                    repo_root=root,
                    eval_script_path=script,
                    video_path_column="video_path",
                    task_type="loop",
                    disable_vlm=True,
                    run_naturalness=False,
                ),
                input_csv=input_csv,
                base_output_dir=root / "results",
                experiment_name="candidate",
                dry_run=False,
            )
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
