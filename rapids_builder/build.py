# Copyright (c) 2024, NVIDIA CORPORATION.

import re
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module

import tomli
import tomli_w


# Avoid unnecessary I/O by caching.
@lru_cache(1)
def _get_pyproject():
    """Parse and return the pyproject.toml file."""
    with open("pyproject.toml", "rb") as f:
        return tomli.load(f)


@lru_cache(1)
def _get_backend():
    """Get the wrapped build backend specified in pyproject.toml."""
    pyproject = _get_pyproject()

    try:
        build_backend = pyproject["tool"]["rapids_builder"]["build-backend"]
    except KeyError:
        raise ValueError(
            "No build backend specified in pyproject.toml's tool.rapids_builder table"
        )

    try:
        return import_module(build_backend)
    except ImportError:
        raise ValueError(
            "Could not import build backend specified in pyproject.toml's "
            "tool.rapids_builder table. Make sure you specified the right optional "
            "dependency in your build-system.requires entry for rapids_builder."
        )


@lru_cache(1)
def _get_cuda_major():
    """Get the CUDA suffix from the pyproject.toml file."""
    try:
        process_output = subprocess.run(["nvcc", "--version"], capture_output=True)
        output_lines = process_output.stdout.decode().splitlines()
        match = re.search(r"release (\d+)", output_lines[3])
        return match.group(1)
    except BaseException as e:
        raise ValueError(
            "Could not determine the CUDA version. Make sure nvcc is in your PATH."
        ) from e


@lru_cache(1)
def _get_cuda_suffix():
    """Get the CUDA suffix from the pyproject.toml file."""
    return f"-cu{_get_cuda_major()}"


_VERSIONED_RAPIDS_WHEELS = [
    "rmm",
    "pylibcugraphops",
    "pylibcugraph",
    "nx-cugraph",
    "dask-cudf",
    "cuspatial",
    "cuproj",
    "cuml",
    "cugraph",
    "cudf",
    "ptxcompiler",
    "cubinlinker",
    "cugraph-dgl",
    "cugraph-pyg",
    "cugraph-equivariant",
    "raft-dask",
    "pylibwholegraph",
    "pylibraft",
    "cuxfilter",
    "cucim",
    "ucx-py",
    "ucxx",
    "pynvjitlink",
    "distributed-ucxx",
]


def _suffix_requires(requires):
    """Add the CUDA suffix to any versioned RAPIDS wheels in requires."""
    new_requires = []
    suffix = _get_cuda_suffix()
    for req in requires:
        for known_req in _VERSIONED_RAPIDS_WHEELS:
            if known_req in req:
                new_requires.append(req.replace(known_req, known_req + suffix))
                break
        else:
            # cupy is a special case because it's not a RAPIDS wheel.
            if "cupy" in req:
                new_requires.append(
                    req.replace("cupy", f"cupy-cuda{_get_cuda_major()}x")
                )
            new_requires.append(req)
    return new_requires


def _supplement_requires(getter, config_settings):
    """Add to the list of requirements for the build backend.

    This is used to add the requirements specified in the rapids_builder table.
    """
    pyproject = _get_pyproject()

    try:
        requires = _suffix_requires(pyproject["tool"]["rapids_builder"]["requires"])
    except KeyError:
        requires = []

    if (getter := getattr(_get_backend(), getter, None)) is not None:
        requires.extend(getter(config_settings))
    return requires


# The hooks in this file could be defined more programmatically by iterating over the
# backend's attributes, but it's simpler to just define them explicitly and avoids any
# potential issues with assuming the right pyproject.toml is readable at import time (we
# need to load pyproject.toml to know what the build backend is). Note that this also
# prevents us from using something like functools.wraps to copy the docstrings from the
# backend's hooks to the rapids_builder hooks, but that's not a big deal because these
# functions only executed by the build frontend and are not user-facing.
def get_requires_for_build_wheel(config_settings):
    return _supplement_requires("get_requires_for_build_wheel", config_settings)


def get_requires_for_build_sdist(config_settings):
    return _supplement_requires("get_requires_for_build_sdist", config_settings)


def get_requires_for_build_editable(config_settings):
    return _supplement_requires("get_requires_for_build_editable", config_settings)


@contextmanager
def _modify_name_and_requirements():
    """
    Temporarily modify the name and dependencies of the package being built.

    This is used to allow the backend to modify the name of the package
    being built. This is useful for projects that want to build wheels
    with a different name than the package name.
    """
    pyproject = _get_pyproject()
    project_data = pyproject["project"]
    project_data["name"] += _get_cuda_suffix()

    dependencies = pyproject["project"].get("dependencies")
    if dependencies is not None:
        project_data["dependencies"] = _suffix_requires(project_data["dependencies"])

    optional_dependencies = pyproject["project"].get("optional-dependencies")
    if optional_dependencies is not None:
        project_data["optional-dependencies"] = {
            extra: _suffix_requires(deps)
            for extra, deps in optional_dependencies.items()
        }

    pyproject_file = "pyproject.toml"
    bkp_pyproject_file = ".pyproject.toml.rapids_builder.bak"
    try:
        shutil.move(pyproject_file, bkp_pyproject_file)
        with open(pyproject_file, "wb") as f:
            tomli_w.dump(pyproject, f)
        yield
    finally:
        # Restore by moving rather than writing to avoid any formatting changes.
        shutil.move(bkp_pyproject_file, pyproject_file)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _modify_name_and_requirements():
        return _get_backend().build_wheel(
            wheel_directory, config_settings, metadata_directory
        )


def build_sdist(sdist_directory, config_settings=None):
    with _modify_name_and_requirements():
        return _get_backend().build_sdist(sdist_directory, config_settings)


# The three hooks below are optional and may not be implemented by the wrapped backend.
# These definitions assume that they will only be called if the wrapped backend
# implements them by virtue of the logic in __init__.py.
def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    with _modify_name_and_requirements():
        return _get_backend().build_editable(
            wheel_directory, config_settings, metadata_directory
        )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _modify_name_and_requirements():
        return _get_backend().prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    with _modify_name_and_requirements():
        return _get_backend().prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )
