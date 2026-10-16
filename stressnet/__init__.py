"""Shared library for the stressnet-tests project.

The training scripts (`train_*.py`) at the repository root are thin CLI
entrypoints; reusable training, data, plotting, reporting and configuration
logic lives here so it can be imported and reused (including by downstream
fine-tuning code).
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
