"""Track cluster run settings while gating resume only on chunk boundaries."""

from __future__ import annotations

import json
from pathlib import Path


def config_differences(previous: dict, current: dict) -> dict:
    differences = {}
    missing = "<missing>"

    def compare(old: dict, new: dict, prefix: str = "") -> None:
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            before, after = old.get(key, missing), new.get(key, missing)
            if isinstance(before, dict) and isinstance(after, dict):
                compare(before, after, path)
            elif before != after:
                differences[path] = {"previous": before, "current": after}

    compare(previous, current)
    return differences


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def record_resume_config(config_path: Path, current: dict,
                         legacy_chunk_size: int | None = None) -> dict:
    """Accept other changed settings, saving every distinct job configuration."""
    if not config_path.is_file():
        _write_json(config_path, current)
        return {}
    previous = json.loads(config_path.read_text())
    saved_chunk_size = previous.get("chunk_size", legacy_chunk_size)
    if saved_chunk_size != current["chunk_size"]:
        raise ValueError(
            f"Cannot resume {config_path}: saved chunk_size={saved_chunk_size} "
            f"but current --chunk-size={current['chunk_size']}. "
            "Use the original chunk size to reuse completed chunks."
        )
    if previous == current:
        return {}
    differences = config_differences(previous, current)
    print(f"Resuming with changed settings: {json.dumps(differences, sort_keys=True)}",
          flush=True)
    history_path = config_path.with_name("resume_configs.json")
    history = (json.loads(history_path.read_text()) if history_path.is_file()
               else [previous])
    history.append(current)
    _write_json(history_path, history)
    _write_json(config_path, current)
    return differences
