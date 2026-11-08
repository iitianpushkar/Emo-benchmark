"""Seven-class MELD emotion prompt for frozen Qwen embeddings."""


def make_meld_prompt(utterance: str) -> str:
    return (
        "Task: infer the speaker's overall emotion from the video and utterance "
        "as exactly one of: anger, disgust, fear, joy, neutral, sadness, surprise.\n"
        f"Utterance: {utterance}\n"
        "Use facial expression, body cues, and wording as evidence."
    )
