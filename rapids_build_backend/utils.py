# Copyright (c) 2024, NVIDIA CORPORATION.

import os

import tomlkit


def _get_pyproject(dirname: str = ".") -> tomlkit.toml_document.TOMLDocument:
    """Parse and return the pyproject.toml file."""
    with open(os.path.join(dirname, "pyproject.toml")) as f:
        return tomlkit.load(f)
