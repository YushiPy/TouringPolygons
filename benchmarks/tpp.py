#!/usr/bin/env python3
"""Compatibility entry point for the benchmark command suite."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPTS_DIR))

from tpp import main


if __name__ == "__main__":
	raise SystemExit(main())
