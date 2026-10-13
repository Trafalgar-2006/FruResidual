import os
import sys
from pathlib import Path

# torch and conda-forge numpy ship separate OpenMP runtimes on Windows; without
# this the process aborts when numpy runs after torch (same fix as FruResidual).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
