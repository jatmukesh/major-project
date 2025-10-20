"""Utilities to help diagnose TensorFlow GPU visibility issues for this project."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

WSL_LIBCUDA_PATH = Path("/usr/lib/wsl/lib/libcuda.so.1")

CUDA_EXPECTED_VERSION = os.environ.get("TF_CUDA_VERSION", "12.3")
CUDNN_EXPECTED_VERSION = os.environ.get("TF_CUDNN_VERSION", "9.1")

IS_WINDOWS = os.name == "nt"

def _default_cuda_root() -> Path:
    if IS_WINDOWS:
        return Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3")
    # On Linux/WSL the canonical location is /usr/local/cuda (symlink to the
    # versioned directory such as /usr/local/cuda-12.3).
    return Path("/usr/local/cuda")


def _candidate_cuda_roots() -> Iterable[Path]:
    env_vars = ["CUDA_PATH", "CUDA_HOME"]
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            yield Path(value)
    yield _default_cuda_root()


CUDA_ROOT = next((root for root in _candidate_cuda_roots() if Path(root).exists()), _default_cuda_root())


def _run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and capture output without raising if it fails."""
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def check_python_version() -> str:
    version = sys.version.replace("\n", " ")
    return f"Python interpreter: {version}"


def check_tensorflow_version() -> str:
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"TensorFlow import failed: {exc}"
    devices = tf.config.list_physical_devices("GPU")
    built_with_cuda = getattr(tf.test, "is_built_with_cuda", lambda: False)()
    details = [
        f"TensorFlow version: {tf.__version__}",
        f"Built with CUDA support: {built_with_cuda}",
        f"Discovered GPUs: {devices!r}",
    ]
    if not devices and sys.platform.startswith("win"):
        details.append(
            "The official TensorFlow wheels for Windows are CPU-only. "
            "Use WSL2 (Ubuntu) or a Linux host for CUDA GPU training, or install "
            "tensorflow-directml for a DirectML-based GPU backend."
        )
    return "\n".join(details)


def check_nvidia_smi() -> str:
    code, out, err = _run_command(["nvidia-smi"])
    if code != 0:
        return f"nvidia-smi failed (code {code}): {err or out}"
    return "nvidia-smi output:\n" + out


def check_cuda_directories(root: Path = CUDA_ROOT) -> str:
    if not root.exists():
        return f"CUDA root not found at {root}"
    if IS_WINDOWS:
        expected_lib = "lib"
    else:
        # Most CUDA Linux distributions install to lib64, but keep the legacy
        # symlink for tooling that expects "lib".
        expected_lib = "lib64"

    missing = [
        name
        for name in ("bin", "include", expected_lib)
        if not (root / name).exists()
    ]
    lines = [f"CUDA root: {root}"]
    if missing:
        lines.append("Missing subdirectories: " + ", ".join(missing))
    else:
        lines.append("All expected CUDA subdirectories present.")
    return "\n".join(lines)


def check_cudnn_files(root: Path = CUDA_ROOT) -> str:
    if IS_WINDOWS:
        cudnn_bin = list((root / "bin").glob("cudnn*.dll"))
        cudnn_lib = list((root / "lib" / "x64").glob("cudnn*.lib"))
    else:
        lib_dirs = [root / "lib64", root / "lib"]
        cudnn_bin = []
        cudnn_lib = []
        for lib_dir in lib_dirs:
            cudnn_bin.extend(lib_dir.glob("libcudnn*.so"))
            cudnn_lib.extend(lib_dir.glob("libcudnn*.so.*"))
    if not cudnn_bin and not cudnn_lib:
        return "cuDNN binaries not found in CUDA installation."
    lines = ["cuDNN files detected:"]
    for path in sorted({p.name for p in cudnn_bin + cudnn_lib}):
        lines.append(f"  - {path}")
    return "\n".join(lines)


def check_environment_path() -> str:
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    cuda_bin = str(CUDA_ROOT / "bin")
    cudnn_hint = [
        entry
        for entry in path_entries
        if entry.lower().startswith(cuda_bin.lower())
    ]
    if cudnn_hint:
        status = "CUDA bin directory present in PATH."
    else:
        status = f"CUDA bin directory missing from PATH: expected {cuda_bin}"

    if not IS_WINDOWS:
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        if "/usr/lib/wsl/lib" not in ld_library_path.split(":"):
            status += (
                "\nWSL shim libraries not found in LD_LIBRARY_PATH. Add "
                "'/usr/lib/wsl/lib' so TensorFlow can load libcuda.so.1."
            )
    return status


def check_libcuda_visibility() -> str:
    if IS_WINDOWS:
        return "libcuda.dll is provided by the Windows driver; no ldconfig check performed."
    if WSL_LIBCUDA_PATH.exists():
        return f"WSL libcuda shim present at {WSL_LIBCUDA_PATH}."
    code, out, err = _run_command(["ldconfig", "-p"])
    if code != 0:
        return f"ldconfig -p failed (code {code}): {err or out}"
    if "libcuda.so" in out:
        return "libcuda.so present in dynamic loader cache."
    return (
        "libcuda.so not found. Ensure the NVIDIA driver is installed on the Windows host "
        "and that /usr/lib/wsl/lib is on LD_LIBRARY_PATH."
    )


def check_nvcc_version() -> str:
    code, out, err = _run_command(["nvcc", "--version"])
    if code != 0:
        return f"nvcc --version failed (code {code}): {err or out}"

    match = re.search(r"release\s+(\d+\.\d+)", out)
    version = match.group(1) if match else "unknown"
    message = ["nvcc --version output:", out]
    if version != "unknown" and version != CUDA_EXPECTED_VERSION:
        message.append(
            "WARNING: TensorFlow 2.16 expects CUDA "
            f"{CUDA_EXPECTED_VERSION}, but nvcc reports {version}."
        )
        message.append(
            "Install cuda-toolkit-12-3 inside WSL or adjust TF_CUDA_VERSION if "
            "you intentionally use a different stack."
        )
    return "\n".join(message)


def run_all_checks() -> str:
    sections = [
        check_python_version(),
        check_tensorflow_version(),
        check_nvidia_smi(),
        check_nvcc_version(),
        check_cuda_directories(),
        check_cudnn_files(),
        check_environment_path(),
        check_libcuda_visibility(),
        f"Expected CUDA version: {CUDA_EXPECTED_VERSION}",
        f"Expected cuDNN version: {CUDNN_EXPECTED_VERSION}",
    ]
    return "\n\n".join(sections)


if __name__ == "__main__":
    print(run_all_checks())
