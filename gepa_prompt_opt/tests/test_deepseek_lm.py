import unittest

from gepa_prompt_opt.deepseek_lm import _normalize_model_name


class TestDeepSeekLM(unittest.TestCase):
    def test_normalize_model_name(self):
        self.assertEqual(_normalize_model_name("deepseek/deepseek-chat"), "deepseek-chat")
        self.assertEqual(_normalize_model_name("deepseek-chat"), "deepseek-chat")
        self.assertEqual(_normalize_model_name(""), "deepseek-chat")


if __name__ == "__main__":
    unittest.main()

