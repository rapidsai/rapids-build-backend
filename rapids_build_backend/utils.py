# Copyright (c) 2024, NVIDIA CORPORATION.

import os

import tomli


def _get_pyproject(dirname="."):
    """Parse and return the pyproject.toml file."""
    with open(os.path.join(dirname, "pyproject.toml"), "rb") as f:
        return tomli.load(f)
