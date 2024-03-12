# Copyright (c) 2024, NVIDIA CORPORATION.

import os
from functools import lru_cache

import tomli


# Avoid unnecessary I/O by caching.
@lru_cache(1)
def _get_pyproject(dirname="."):
    """Parse and return the pyproject.toml file."""
    with open(os.path.join(dirname, "pyproject.toml"), "rb") as f:
        return tomli.load(f)
