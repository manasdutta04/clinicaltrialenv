#!/usr/bin/env python3
"""Wrapper to run the root inference agent from the nested app directory."""
from pathlib import Path
import runpy
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

runpy.run_path(str(ROOT_DIR / "inference.py"), run_name="__main__")
