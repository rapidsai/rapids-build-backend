# Copyright (c) 2024, NVIDIA CORPORATION.

import os

import tomlkit


def _get_pyproject(dirname: str = ".") -> tomlkit.toml_document.TOMLDocument:
    """Parse and return the pyproject.toml file."""
    with open(os.path.join(dirname, "pyproject.toml")) as f:
        return tomlkit.load(f)


def _get_setup_py() -> str:
    """
    Returns a string with the contents of setup.py,
    or empty string if it doesn't exist.
    """
    # setuptools.build_meta.get_requires_for_wheel() assumes that "setup.py" is directly
    # relative to the current working directly, so rapids-build-backend can too.
    #
    # ref: https://github.com/pypa/setuptools/blob/f91fa3d9fc7262e0467e4b2f84fe463f8f8d23cf/setuptools/build_meta.py#L304
    setup_py_file = "setup.py"

    if not os.path.isfile(setup_py_file):
        return ""

    with open(setup_py_file) as f:
        return f.read()
