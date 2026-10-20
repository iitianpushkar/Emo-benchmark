"""Physical decoder-token removal for single-sample, cache-free Qwen2.5-VL."""

# Allow both `python scripts/<group>/<file>.py` and `python -m scripts...`.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from contextlib import contextmanager
import torch


def sequence_indices(visual_mask, token=None):
    """Retain all text/special tokens and optionally one visual-token ordinal."""
    if visual_mask.ndim != 1 or visual_mask.dtype != torch.bool:
        raise ValueError('Expected a one-dimensional boolean visual mask')
    visual = visual_mask.nonzero().flatten()
    keep = ~visual_mask.clone()
    if token is not None:
        if not 0 <= token < len(visual):
            raise ValueError('Visual token ordinal out of range')
        keep[visual[token]] = True
    indices = keep.nonzero().flatten()
    if not len(indices):
        raise ValueError('Cannot remove the entire sequence')
    return indices


@contextmanager
def remove_visual_tokens(language_model, visual_mask, depth, token=None):
    """Remove after `depth` completed blocks; preserve original RoPE positions.

    Supports eager attention, batch size one, no KV cache. Every subsequent
    block receives sliced masks/positions because the parent constructs these
    once using the original sequence. Hidden states are shortened only once.
    """
    layers = language_model.layers
    if not 0 <= depth < len(layers):
        raise ValueError(f'depth must be in [0, {len(layers)-1}]')
    indices = sequence_indices(visual_mask, token)
    original_length = len(visual_mask)
    handles, calls = [], [0] * len(layers)

    def hook_for(index):
        def hook(module, args, kwargs):
            calls[index] += 1
            kwargs = dict(kwargs)
            hidden = args[0] if args else kwargs['hidden_states']
            if hidden.shape[0] != 1:
                raise ValueError('Only batch size one is supported')
            idx = indices.to(hidden.device)
            if index == depth:
                if hidden.shape[1] != original_length:
                    raise ValueError('Unexpected input sequence length')
                hidden = hidden.index_select(1, idx)
            if args:
                args = (hidden,) + args[1:]
            else:
                kwargs['hidden_states'] = hidden
            mask = kwargs.get('attention_mask')
            if mask is None or mask.ndim != 4 or mask.shape[-2:] != (original_length, original_length):
                raise ValueError('Expected original square 4D eager causal attention mask; unsupported transformers version/backend')
            kwargs['attention_mask'] = mask.index_select(-2, idx.to(mask.device)).index_select(-1, idx.to(mask.device))
            for key in ('position_ids', 'cache_position'):
                value = kwargs.get(key)
                if value is not None:
                    kwargs[key] = value.index_select(-1, idx.to(value.device))
            rotary = kwargs.get('position_embeddings')
            if rotary is None:
                raise ValueError('Expected precomputed Qwen rotary embeddings')
            kwargs['position_embeddings'] = tuple(v.index_select(-2, idx.to(v.device)) for v in rotary)
            return args, kwargs
        return hook
    try:
        for i in range(depth, len(layers)):
            handles.append(layers[i].register_forward_pre_hook(hook_for(i), with_kwargs=True))
        yield indices
        if any(calls[i] != 1 for i in range(depth, len(layers))):
            raise RuntimeError('Expected each decoder block to execute exactly once')
    finally:
        for handle in handles:
            handle.remove()


def information_scores(p_single, p_none):
    """Signed probability differences; do not clamp negative evidence."""
    return p_single - p_none
