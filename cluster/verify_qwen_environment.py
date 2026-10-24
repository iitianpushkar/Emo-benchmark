"""Check Qwen experiment dependencies without loading weights or using a GPU."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "torch": ">=2.5.1",
    "torchvision": "",
    "transformers": "==4.57.1",
    "accelerate": "",
    "huggingface_hub": "",
    "qwen-vl-utils": ">=0.0.14",
    "decord": "",
    "numpy": "",
    "mpmath": "",
    "scipy": "",
    "scikit-learn": "",
    "Pillow": "",
}


def require_full_h100(torch) -> str:
    """Reject H100 MIG slices for the full 80 GB experiment protocol."""
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an allocated full H100 GPU")
    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if "H100" not in name or memory_gib < 70:
        raise RuntimeError(
            f"This experiment requires a full H100 80 GB GPU; allocated device is "
            f"{name} with {memory_gib:.1f} GiB. On Rorqual, request --gpus=h100:1 "
            "instead of a 1g/2g/3g MIG slice."
        )
    return name


def verify_qwen_environment(*, curve: bool = False) -> dict:
    """Report package versions and import compatibility on a login node."""
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env and Path(sys.prefix).resolve() != Path(virtual_env).expanduser().resolve():
        raise RuntimeError(
            f"VIRTUAL_ENV={virtual_env}, but {sys.executable} is running with "
            f"sys.prefix={sys.prefix}. Load cluster modules before activating the "
            "virtual environment, then use its bin/python explicitly."
        )
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError as exc:
        raise RuntimeError("Install the Python requirements, including packaging, before verification") from exc

    packages = {}
    errors = []
    requirements = {**REQUIRED_PACKAGES, **({"matplotlib": ""} if curve else {})}
    print(f"Checking {len(requirements)} Qwen experiment packages in {sys.executable}", flush=True)
    for name, specifier in requirements.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            errors.append(f"{name}{specifier}: not installed")
            print(f"  {name}: MISSING", flush=True)
            continue
        packages[name] = version
        valid = not specifier or Version(version) in SpecifierSet(specifier)
        print(f"  {name}: {version}{'' if valid else f' (requires {specifier})'}", flush=True)
        if not valid:
            errors.append(f"{name}: installed {version}, requires {specifier}")

    imports = {
        "torch": ("torch", None),
        "torchvision": ("torchvision", None),
        "PyTorch FLOP counter": ("torch.utils.flop_counter", "FlopCounterMode"),
        "accelerate": ("accelerate", None),
        "NumPy": ("numpy", None),
        "mpmath": ("mpmath", None),
        "SciPy": ("scipy", None),
        "Pillow": ("PIL.Image", None),
        "Qwen2.5-VL model": ("transformers", "Qwen2_5_VLForConditionalGeneration"),
        "Qwen processor": ("transformers", "AutoProcessor"),
        "Qwen video utilities": ("qwen_vl_utils", "process_vision_info"),
        "decord video reader": ("decord", "VideoReader"),
        "OpenCV": ("cv2", None),
        "scikit-learn metrics": ("sklearn.metrics", "f1_score"),
    }
    if curve:
        imports["Matplotlib plotting"] = ("matplotlib.pyplot", None)
    import_status = {}
    print("Checking runtime imports (this may take a moment)", flush=True)
    for label, (module_name, attribute) in imports.items():
        try:
            module = importlib.import_module(module_name)
            if attribute is not None:
                getattr(module, attribute)
            import_status[label] = "ok"
            if label == "OpenCV":
                opencv_version = getattr(module, "__version__", "unknown")
                packages["OpenCV (cv2)"] = opencv_version
                print(f"  OpenCV: {opencv_version} (cv2 import OK)", flush=True)
            else:
                print(f"  {label}: OK", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if label == "OpenCV" and isinstance(exc, ModuleNotFoundError):
                message += "; on Alliance, load the OpenCV module before activating the virtual environment"
            import_status[label] = message
            errors.append(f"{label}: {message}")
            print(f"  {label}: FAIL ({message})", flush=True)

    cuda_build = None
    if import_status["torch"] == "ok":
        cuda_build = importlib.import_module("torch").version.cuda
        print(f"  PyTorch CUDA build: {cuda_build or 'NONE'}", flush=True)
        if cuda_build is None:
            errors.append("PyTorch is a CPU-only build; install a CUDA-enabled build for the H100 job")
    if errors:
        raise RuntimeError("Qwen environment verification failed:\n- " + "\n- ".join(errors))
    print("Qwen environment verification passed; GPU availability is checked in the compute-node job", flush=True)
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "prefix": sys.prefix, "virtual_env": virtual_env,
            "packages": packages, "imports": import_status, "torch_cuda_build": cuda_build}
