#!/usr/bin/env python3
"""Training-free diverse-frame selection with frozen image embeddings."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_RETAIN_RATIO = 0.20
DEFAULT_ENCODER_MODEL = "facebook/dinov2-base"
IMPLEMENTATION_NAME = "centroid_seeded_frame_max_min_cosine"


def retained_frame_count(num_frames: int, retain_ratio: float = DEFAULT_RETAIN_RATIO) -> int:
    """Return the bounded number of decoded frames to retain."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if not 0.0 < retain_ratio <= 1.0:
        raise ValueError("retain_ratio must be in the interval (0, 1]")
    return min(num_frames, max(1, int(round(num_frames * retain_ratio))))


def select_diverse_frame_indices(
    frame_embeddings: torch.Tensor,
    retain_ratio: float = DEFAULT_RETAIN_RATIO,
) -> torch.Tensor:
    """Select centroid-seeded max-min cosine-diverse frames in temporal order."""
    if frame_embeddings.ndim != 2:
        raise ValueError(
            "Expected frame_embeddings shaped [num_frames, hidden_dim], "
            f"got {tuple(frame_embeddings.shape)}"
        )
    if frame_embeddings.shape[0] == 0 or frame_embeddings.shape[1] == 0:
        raise ValueError("frame_embeddings must have non-zero frame and feature dimensions")
    if not torch.isfinite(frame_embeddings).all():
        raise ValueError("frame_embeddings contain NaN or Inf values")

    num_frames = frame_embeddings.shape[0]
    target_count = retained_frame_count(num_frames, retain_ratio)
    device = frame_embeddings.device
    if target_count == num_frames:
        return torch.arange(num_frames, device=device, dtype=torch.long)

    normalized = F.normalize(frame_embeddings.float(), p=2, dim=-1, eps=1e-12)
    centroid = F.normalize(normalized.mean(dim=0, keepdim=True), p=2, dim=-1, eps=1e-12)
    first = int((normalized @ centroid.transpose(0, 1)).squeeze(1).argmax().item())

    distances = (1.0 - normalized @ normalized.transpose(0, 1)).clamp_(0.0, 2.0)
    selected = torch.zeros(num_frames, dtype=torch.bool, device=device)
    selected[first] = True
    min_distance = distances[first].clone()
    min_distance[selected] = float("-inf")

    for _ in range(1, target_count):
        next_index = int(min_distance.argmax().item())
        selected[next_index] = True
        min_distance = torch.minimum(min_distance, distances[next_index])
        min_distance[selected] = float("-inf")

    return torch.nonzero(selected, as_tuple=False).squeeze(1).sort().values.long()


def frame_tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    """Convert one CHW RGB frame to PIL without assuming its numeric range."""
    if frame.ndim != 3 or frame.shape[0] not in {1, 3, 4}:
        raise ValueError(f"Expected a CHW frame, got {tuple(frame.shape)}")
    pixels = frame.detach().cpu()
    if pixels.dtype.is_floating_point:
        if float(pixels.max().item()) <= 1.0:
            pixels = pixels * 255.0
        pixels = pixels.round()
    pixels = pixels.clamp(0, 255).to(torch.uint8)
    if pixels.shape[0] == 1:
        pixels = pixels.expand(3, -1, -1)
    if pixels.shape[0] == 4:
        pixels = pixels[:3]
    return Image.fromarray(pixels.permute(1, 2, 0).contiguous().numpy(), mode="RGB")


class FrameDiversitySelector:
    """Encode decoded frames with a frozen image model and retain a diverse subset."""

    def __init__(
        self,
        model_id: str = DEFAULT_ENCODER_MODEL,
        device: str = "cpu",
        batch_size: int = 32,
        trust_remote_code: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        from transformers import AutoImageProcessor, AutoModel

        self.model_id = model_id
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _pool_model_output(output: Any) -> torch.Tensor:
        pooler_output = getattr(output, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        last_hidden_state = getattr(output, "last_hidden_state", None)
        if last_hidden_state is None or last_hidden_state.ndim != 3:
            raise ValueError(
                "Frame encoder must return pooler_output or a 3D last_hidden_state"
            )
        return last_hidden_state[:, 0]

    def encode(self, video_frames: torch.Tensor) -> torch.Tensor:
        if video_frames.ndim != 4:
            raise ValueError(
                "Expected video_frames shaped [num_frames, channels, height, width], "
                f"got {tuple(video_frames.shape)}"
            )
        images = [frame_tensor_to_pil(frame) for frame in video_frames]
        embeddings: list[torch.Tensor] = []
        for start in range(0, len(images), self.batch_size):
            batch = self.processor(
                images=images[start : start + self.batch_size],
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.inference_mode():
                output = self.model(**batch)
            embeddings.append(self._pool_model_output(output).detach().float().cpu())
        return torch.cat(embeddings, dim=0)

    def select(
        self,
        video_frames: torch.Tensor,
        retain_ratio: float = DEFAULT_RETAIN_RATIO,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encode(video_frames)
        indices = select_diverse_frame_indices(embeddings, retain_ratio=retain_ratio)
        selected_frames = video_frames.index_select(0, indices.to(video_frames.device))
        return selected_frames, indices.cpu()
