# Central path configuration for COD-AI.
# When running as Windows Python, data and models must live on the native
# Windows filesystem (h5py and PyTorch cannot write to WSL UNC paths).
# Files go under C:/Users/<user>/COD-AI/
import sys
from pathlib import Path

if sys.platform == "win32":
    BASE_DIR = Path.home() / "COD-AI"
else:
    BASE_DIR = Path(__file__).parent

DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
