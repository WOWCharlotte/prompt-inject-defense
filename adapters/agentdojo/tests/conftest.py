"""Pytest configuration: ensure project root and src/ are on sys.path."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"

for p in (str(PROJECT_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
