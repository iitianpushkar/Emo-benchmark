import unittest

from scripts.extraction.extract_qwen_shared_embeddings import make_prompt


class EmotionCodePromptTests(unittest.TestCase):
    def test_video_text_prompt_has_fixed_code_mapping_and_answer_suffix(self) -> None:
        prompt = make_prompt("That was amazing.", "emotion_code", "video_text")

        for code, emotion in zip("ABCDEFG", [
            "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
        ]):
            self.assertIn(f"{code} = {emotion}", prompt)
        self.assertIn("Utterance: That was amazing.", prompt)
        self.assertTrue(prompt.endswith("Answer with only one code:"))

    def test_code_prompt_supports_each_modality_mode(self) -> None:
        for mode in ["video_text", "video_only", "text_only"]:
            with self.subTest(mode=mode):
                prompt = make_prompt("Test utterance.", "emotion_code", mode)
                self.assertTrue(prompt.endswith("Answer with only one code:"))


if __name__ == "__main__":
    unittest.main()
