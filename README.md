# Major Project

## Overview
This repository contains assets for the prescription medicine recognition project, including the main research notebook.

## Environment Setup
1. **Install a supported Python version.** TensorFlow 2.16 publishes wheels for
   Python 3.10–3.12 only. Windows users can download the latest 3.11.x
   installer from <https://www.python.org/downloads/windows/> (tick “Add
   python.exe to PATH” during installation). After installing, confirm that the
   interpreter is available from both the launcher and the current shell:
   ```powershell
   python --version   # should print Python 3.11.x
   py --list          # look for an entry such as 3.11-64
   ```
   Seeing `Python 3.11.5` (or similar) means the installation succeeded.
2. **Re-create the virtual environment with the supported interpreter.** If you
   previously created `.venv` with Python 3.13, remove it first:
   ```powershell
   Remove-Item -Recurse -Force .venv
   ```
   Then create and activate a fresh environment that pins Python 3.11:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate
   ```
   Confirm the correct interpreter is active (the prompt should start with
   `(.venv)` and `python --version` should report Python 3.11.x).
3. **Install project dependencies inside the activated environment:**
   ```powershell
   python -m pip install --upgrade pip
   pip install numpy pandas pillow tensorflow==2.16.1
   ```
   *If you plan to use GPU acceleration, note that the official TensorFlow wheels for Windows are CPU-only.* Use WSL2 (Ubuntu) or a Linux host for CUDA-based GPU training, or install `tensorflow-directml` if you specifically need a DirectML backend on Windows.

### Why we are not using "latest everything"
- **TensorFlow wheels lag behind Python releases.** At the moment, TensorFlow 2.16 is the newest stable build for Windows, and it only publishes wheels for Python 3.10–3.12. Python 3.13 environments therefore cannot install TensorFlow without building from source, which is outside the scope of this project.
- **Pre-release TensorFlow builds target newer CUDA stacks.** Experimental releases such as 2.20.0 require CUDA 12.4+ and matching cuDNN binaries that TensorFlow’s own compatibility table still treats as preview. Mixing those with the rest of this project increases the chance of runtime linker errors (missing DLLs, mismatched compute capability) and driver incompatibilities.
- **API churn breaks notebooks unexpectedly.** Jumping to the newest major versions of TensorFlow or its transitive dependencies (e.g., keras-core) often deprecates symbols or changes default behaviors. Staying on the documented stack keeps the training notebook reproducible across collaborators’ machines.

If you decide to experiment with newer versions, check TensorFlow’s [official compatibility matrix](https://www.tensorflow.org/install/source#gpu) first and be prepared to realign CUDA, cuDNN, and any code that depends on removed APIs.

## GPU Verification
After installing TensorFlow, activate `.venv` (the prompt should start with `(.venv)`) and run:
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
A detected GPU appears as `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`.

> **Note:** TensorFlow may print informational messages (for example, warnings about oneDNN optimizations) before the device list.
> That is expected—wait for the final line that shows the actual `PhysicalDevice(...)` entries.

If the command prints `[]`, TensorFlow did **not** detect a GPU. On native Windows, this is expected because the official pip
packages are built without CUDA. To train on a GPU, switch to a WSL2 Ubuntu environment (or another Linux host) and follow
TensorFlow’s [GPU requirements](https://www.tensorflow.org/install/source#gpu), or install `tensorflow-directml` if you want the
DirectML backend that Microsoft maintains for Windows.

### WSL2 (Ubuntu) GPU setup checklist
Follow these steps from an Ubuntu shell (for example, one opened with `wsl -d Ubuntu`):

1. **Update the distro and install build prerequisites**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y build-essential git curl wget ca-certificates
   ```
2. **Install the CUDA toolkit that matches TensorFlow 2.16 (12.3)**
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   sudo apt install -y cuda-toolkit-12-3
   ```
   After installation, run `wsl --shutdown` from Windows PowerShell and reopen Ubuntu so the driver bridge is refreshed.

   > **Ubuntu 24.04 / WSL2 note:** On Noble, `cuda-toolkit-12-3` often
   > attempts to pull in `nsight-systems`, which still depends on the
   > legacy `libtinfo5`. Use one of the following fixes before retrying
   > the install:
   > - Skip the optional Nsight components:
   >   ```bash
   >   sudo apt install -y cuda-toolkit-12-3 --no-install-recommends
   >   ```
   > - If you need Nsight Systems, temporarily enable the Jammy archive,
   >   install `libtinfo5`, then remove the extra source:
   >   ```bash
   >   echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu jammy main" | \
   >     sudo tee /etc/apt/sources.list.d/jammy-compat.list
   >   sudo apt update
   >   sudo apt install -y libtinfo5
   >   sudo rm /etc/apt/sources.list.d/jammy-compat.list
   >   sudo apt update
   >   ```
   >   (If the mirror above is slow or unreachable, pick an alternative
   >   from <https://packages.ubuntu.com/jammy/libtinfo5>.)
   > Then rerun `sudo apt install -y cuda-toolkit-12-3`.
   Verify that `nvcc` reports the expected toolkit version:
   ```bash
   nvcc --version
   ```
   The output should include `release 12.3`. If it shows an older toolkit (for
   example, 12.0 installed from the default Ubuntu repository), remove it and
   reinstall the 12.3 packages:
   ```bash
   sudo apt remove --purge -y "cuda-toolkit-*" "cuda" "nvidia-cuda-toolkit"
   sudo apt autoremove -y
   sudo apt install -y cuda-toolkit-12-3 --no-install-recommends
   ```
3. **Verify NVIDIA detects the GPU inside WSL**
   ```bash
   nvidia-smi
   ```
   You should see your RTX 4060 listed. If the command fails, update the Windows NVIDIA driver (version 566.26 or newer) and reboot.
4. **Install Python 3.11 and create the project virtual environment**
   ```bash
   sudo apt install -y python3.11 python3.11-venv python3-pip
   cd /mnt/c/Users/mukes/Documents/Workspace/major-project
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python --version  # confirm the prompt now uses Python 3.11.x
   ```
   If the activated shell still reports Python 3.12 (or another version), run
   `deactivate`, remove `.venv`, and recreate it with `python3.11 -m venv .venv`
   to ensure TensorFlow installs against a supported interpreter.
5. **Install the project dependencies with GPU support**
   ```bash
   pip install numpy pandas pillow tensorflow[and-cuda]==2.16.1
   ```
   On Ubuntu/WSL, also expose the CUDA runtime libraries shipped with the
   Windows driver bridge to TensorFlow by adding the WSL shim directory to your
   environment:
   ```bash
   echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```
6. **Confirm TensorFlow can see the GPU**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
   ```
   A healthy setup prints TensorFlow 2.16.1 and at least one `PhysicalDevice` entry.

> **Tip:** Add an alias such as `alias proj='cd /mnt/c/Users/mukes/Documents/Workspace/major-project && source .venv/bin/activate'`
> to your `~/.bashrc` so future WSL sessions jump straight into the activated environment.

## Troubleshooting
- If you encounter `ModuleNotFoundError: No module named 'tensorflow'`, ensure the commands above were executed **after** activating `.venv`.
- For PowerShell execution policy issues, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, close the terminal, and re-open it before activating the environment again.
