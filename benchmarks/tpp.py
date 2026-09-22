#!/usr/bin/env python3
"""Compatibility entry point for the benchmark command suite."""

from __future__ import annotations

import sys
from pathlib import Path


INTERNAL_DIR = Path(__file__).resolve().parent / "_internal"
if str(INTERNAL_DIR) not in sys.path:
	sys.path.insert(0, str(INTERNAL_DIR))

from tpp import main


if __name__ == "__main__":
	raise SystemExit(main())
