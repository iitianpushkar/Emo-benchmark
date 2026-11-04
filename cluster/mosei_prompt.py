"""Prompt text for frozen Qwen CMU-MOSEI embedding extraction."""


def make_mosei_prompt(utterance: str, style: str) -> str:
    if style == "utterance_only":
        return utterance
    if style == "emotion_task":
        return (
            "Task: infer the speaker's overall emotion from the video and utterance "
            "as exactly one of: happy, sad, anger, surprise, disgust, fear, neutral.\n"
            f"Utterance: {utterance}\n"
            "Use facial expression, body cues, and wording as evidence."
        )
    raise ValueError(f"Unknown prompt style: {style}")
