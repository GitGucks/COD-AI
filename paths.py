"""Central path configuration for COD-AI.

When running as Windows Python (which all three scripts do), data and
models must live on the native Windows filesystem — h5py and PyTorch
cannot create files on WSL UNC paths (\\wsl.localhost\...).

All files go under C:\Users\<user>\COD-AI\
"""
import sys
from pathlib import Path

if sys.platform == "win32":
    BASE_DIR = Path.home() / "COD-AI"
else:
    BASE_DIR = Path(__file__).parent

DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
