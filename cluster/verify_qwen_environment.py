"""Check Qwen experiment dependencies without loading weights or using a GPU."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys


REQUIRED_PACKAGES = {
    "torch": ">=2.5.1",
    "transformers": "==4.57.1",
    "accelerate": "",
    "huggingface_hub": "",
    "qwen-vl-utils": ">=0.0.14",
    "decord": "",
    "opencv-python": "",
    "numpy": "",
    "scikit-learn": "",
    "Pillow": "",
}


def verify_qwen_environment(*, curve: bool = False) -> dict:
    """Report package versions and import compatibility on a login node."""
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
        "PyTorch FLOP counter": ("torch.utils.flop_counter", "FlopCounterMode"),
        "accelerate": ("accelerate", None),
        "NumPy": ("numpy", None),
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
            print(f"  {label}: OK", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
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
            "packages": packages, "imports": import_status, "torch_cuda_build": cuda_build}
