import unittest

from gepa_prompt_opt.template import PromptTemplate, render_prompt


class TestTemplate(unittest.TestCase):
    def test_render_prompt_substitution(self):
        t = PromptTemplate(
            raw={
                "template_version": "1.0",
                "task": "loop",
                "components": {
                    "role_instruction": "Role {x}",
                    "action_description": "Do {y}",
                    "loop_completion_requirement": "Return {z}",
                    "temporal_consistency_constraints": "Temporal",
                    "scene_preservation_constraints": "Scene",
                    "negative_constraints": "Neg",
                },
                "render": {
                    "order": [
                        "role_instruction",
                        "action_description",
                        "loop_completion_requirement",
                        "temporal_consistency_constraints",
                        "scene_preservation_constraints",
                        "negative_constraints",
                    ],
                    "separator": "\n",
                    "labels": False,
                },
            }
        )
        out = render_prompt(t, {"x": "A", "y": "B", "z": "C"})
        self.assertIn("Role A", out)
        self.assertIn("Do B", out)
        self.assertIn("Return C", out)


if __name__ == "__main__":
    unittest.main()

