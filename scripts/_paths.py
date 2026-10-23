"""Resolve CLI helpers independently of the current working directory."""
from pathlib import Path

SCRIPT_PATHS = {
    "prepare_meld.py": "data/prepare_meld.py",
    "download_mer2024.py": "data/download_mer2024.py",
    "prepare_mer2024.py": "data/prepare_mer2024.py",
    "export_mosei_csd.py": "data/export_mosei_csd.py",
    "merge_mosei_labels.py": "data/merge_mosei_labels.py",
    "create_meld_conflict_dataset.py": "data/create_meld_conflict_dataset.py",
    "extract_qwen_embeddings.py": "extraction/extract_qwen_embeddings.py",
    "extract_qwen_shared_embeddings.py": "extraction/extract_qwen_shared_embeddings.py",
    "extract_internvl3_embeddings.py": "extraction/extract_internvl3_embeddings.py",
    "extract_smolvlm2_embeddings.py": "extraction/extract_smolvlm2_embeddings.py",
    "run_qwen_shared_chunks.py": "extraction/run_qwen_shared_chunks.py",
    "run_internvl3_chunks.py": "extraction/run_internvl3_chunks.py",
    "run_smolvlm2_chunks.py": "extraction/run_smolvlm2_chunks.py",
    "merge_embedding_chunks.py": "extraction/merge_embedding_chunks.py",
    "train_mlp.py": "classification/train_mlp.py",
    "evaluate_mlp.py": "classification/evaluate_mlp.py",
    "divprune.py": "pruning/divprune.py",
    "random_prune.py": "pruning/random_prune.py",
    "frame_diversity.py": "pruning/frame_diversity.py",
    "fuse_modality_embeddings.py": "fusion/fuse_modality_embeddings.py",
    "run_alpha_sweep.py": "fusion/run_alpha_sweep.py",
    "summarize_alpha_sweep.py": "fusion/summarize_alpha_sweep.py",
    "compare_visual_token_ablation.py": "ablation/compare_visual_token_ablation.py",
    "compare_modality_token_ablation.py": "ablation/compare_modality_token_ablation.py",
    "run_video_only_divprune.py": "ablation/run_video_only_divprune.py",
    "run_meld_conflict_dev.py": "conflicts/run_meld_conflict_dev.py",
    "evaluate_meld_conflicts.py": "conflicts/evaluate_meld_conflicts.py",
    "information_horizon.py": "information_horizon/information_horizon.py",
    "run_information_horizon.py": "information_horizon/run_information_horizon.py",
    "summarize_information_horizon.py": "information_horizon/summarize_information_horizon.py",
    "compile_aligned_results.py": "analysis/compile_aligned_results.py"
}

def resolve_script(name: str) -> Path:
    return Path(__file__).resolve().parent / SCRIPT_PATHS[name]
