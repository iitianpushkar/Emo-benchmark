"""Check Qwen experiment dependencies without loading weights or using a GPU."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "packaging": "",
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


def require_cuda_gpu(torch) -> str:
    """Confirm CUDA is available and report the allocated device capacity."""
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires an allocated CUDA GPU")
    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Allocated CUDA device: {name} ({memory_gib:.1f} GiB)", flush=True)
    return name


def verify_qwen_environment(*, curve: bool = False) -> dict:
    """Check dependencies installed inside the running virtual environment."""
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError(
            f"{sys.executable} is not running inside a Python virtual environment. "
            "Activate the experiment environment after loading cluster modules, "
            "or invoke its bin/python explicitly."
        )
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

    venv_root = Path(sys.prefix).resolve()
    packages = {}
    package_locations = {}
    errors = []
    requirements = {**REQUIRED_PACKAGES, **({"matplotlib": ""} if curve else {})}
    print(f"Checking {len(requirements)} Qwen experiment packages in {sys.executable}", flush=True)
    for name, specifier in requirements.items():
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            errors.append(f"{name}{specifier}: not installed")
            print(f"  {name}: MISSING", flush=True)
            continue
        version = distribution.version
        location = Path(distribution.locate_file("")).resolve()
        packages[name] = version
        package_locations[name] = str(location)
        valid = not specifier or Version(version) in SpecifierSet(specifier)
        inside_venv = location.is_relative_to(venv_root)
        print(f"  {name}: {version}{'' if valid else f' (requires {specifier})'}"
              f"{' (outside virtual environment)' if not inside_venv else ''}", flush=True)
        if not valid:
            errors.append(f"{name}: installed {version}, requires {specifier}")
        if not inside_venv:
            errors.append(f"{name}: installed at {location}, outside {venv_root}")

    imports = {
        "packaging": ("packaging", None),
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
            if label != "OpenCV":
                module_file = getattr(module, "__file__", None)
                if not module_file or not Path(module_file).resolve().is_relative_to(venv_root):
                    raise RuntimeError(f"imported from {module_file}, outside {venv_root}")
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
            "packages": packages, "package_locations": package_locations,
            "imports": import_status, "torch_cuda_build": cuda_build}
