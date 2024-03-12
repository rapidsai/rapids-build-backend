# Copyright (c) 2024, NVIDIA CORPORATION.

from functools import lru_cache

import tomli


# Avoid unnecessary I/O by caching.
@lru_cache(1)
def _get_pyproject():
    """Parse and return the pyproject.toml file."""
    with open("pyproject.toml", "rb") as f:
        return tomli.load(f)
