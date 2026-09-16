"""
setup.py
--------
Run once after creating a fresh conda environment.
Automatically detects GPU, installs the correct PyTorch build,
then installs all remaining dependencies from requirements.txt.

Usage:
    python setup.py
"""

import subprocess
import sys
import os


def run(cmd):
    """Run a shell command and exit on failure."""
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: Command failed: {' '.join(cmd)}")
        sys.exit(1)


def detect_gpu():
    """
    Returns True if an NVIDIA GPU is present.
    Uses nvidia-smi which is available as long as the NVIDIA driver is installed,
    independent of Python or conda.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def install_torch(has_gpu):
    if not has_gpu:
        print("  No GPU detected — skipping, requirements.txt will handle CPU torch.")
        return

    print("  GPU detected — installing CUDA PyTorch (cu128)...")
    run([
        sys.executable, "-m", "pip", "install", "torch",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    ])


def install_requirements():
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_path):
        print("ERROR: requirements.txt not found next to setup.py")
        sys.exit(1)
    run([sys.executable, "-m", "pip", "install", "-r", req_path])


def install_spacy_model():
    run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def verify_torch():
    subprocess.run([
        sys.executable, "-c",
        "import torch; "
        "print(f'  torch version : {torch.__version__}'); "
        "print(f'  CUDA available : {torch.cuda.is_available()}'); "
        "print(f'  Device         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
    ])


if __name__ == "__main__":
    print("=" * 55)
    print("  Environment Setup")
    print("=" * 55)

    print("\n[1/4] Detecting hardware...")
    has_gpu = detect_gpu()

    print("\n[2/4] Installing PyTorch...")
    install_torch(has_gpu)

    print("\n[3/4] Installing requirements.txt...")
    install_requirements()

    print("\n[4/4] Downloading spaCy model...")
    install_spacy_model()

    print("\n" + "=" * 55)
    print("  Setup complete. Verifying PyTorch install:")
    verify_torch()
    print("=" * 55)