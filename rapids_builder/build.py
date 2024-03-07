# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module

import tomli
import tomli_w
from packaging.requirements import Requirement, SpecifierSet


# Avoid unnecessary I/O by caching.
@lru_cache(1)
def _get_pyproject():
    """Parse and return the pyproject.toml file."""
    with open("pyproject.toml", "rb") as f:
        return tomli.load(f)


@lru_cache(1)
def _get_tool_table():
    """Get the rapids_builder tool table from pyproject.toml."""
    pyproject = _get_pyproject()
    try:
        return pyproject["tool"]["rapids_builder"]
    except KeyError as e:
        raise ValueError("No rapids_builder table in pyproject.toml") from e


@lru_cache(1)
def _get_backend():
    """Get the wrapped build backend specified in pyproject.toml."""
    try:
        build_backend = _get_tool_table()["build-backend"]
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
    """Get the CUDA suffix based on nvcc.

    Returns None if nvcc is not in the PATH.
    """
    nvcc_exists = subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0
    if not nvcc_exists:
        if not _get_tool_table().get("allow-no-cuda", False):
            raise ValueError(
                "Could not determine the CUDA version. Make sure nvcc is in your PATH."
            )
        return None

    try:
        process_output = subprocess.run(["nvcc", "--version"], capture_output=True)
    except subprocess.CalledProcessError as e:
        raise ValueError("Failed to get version from nvcc.") from e

    output_lines = process_output.stdout.decode().splitlines()

    try:
        match = re.search(r"release (\d+)", output_lines[3])
        return match.group(1)
    except BaseException as e:
        raise ValueError(
            "Failed to parse CUDA version from nvcc --version output."
        ) from e


def _get_cuda_suffix():
    """Get the CUDA suffix based on nvcc.

    Returns an empty string if nvcc is not in the PATH.
    """
    if (major := _get_cuda_major()) is None:
        return ""
    return f"-cu{major}"


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

_UNVERSIONED_RAPIDS_WHEELS = [
    "dask-cuda",
    "rapids-dask-dependency",
    "ptxcompiler",
    "cubinlinker",
]


def _suffix_requires(requires):
    """Add the CUDA suffix to any versioned RAPIDS wheels in requires."""
    new_requires = []
    suffix = _get_cuda_suffix()
    for req in requires:
        req = Requirement(req)

        is_versioned_wheel = any(req.name == w for w in _VERSIONED_RAPIDS_WHEELS)
        is_unversioned_wheel = any(req.name == w for w in _UNVERSIONED_RAPIDS_WHEELS)
        only_release_deps = "RAPIDS_ONLY_RELEASE_DEPS" in os.environ

        # cupy is a special case because it's not a RAPIDS wheel. If we can't
        # determine the local CUDA version, then we fall back to making the sdist of
        # cupy on PyPI the dependency.
        if req.name == "cupy" and (major := _get_cuda_major()) is not None:
            req.name += f"-cuda{major}x"
        else:
            if is_versioned_wheel:
                req.name += suffix

            # Allow nightlies of RAPIDS packages except in release builds.
            if (is_versioned_wheel or is_unversioned_wheel) and not only_release_deps:
                req.specifier &= SpecifierSet(">=0.0.0a0")

        new_requires.append(str(req))
    return new_requires


def _supplement_requires(getter, config_settings):
    """Add to the list of requirements for the build backend.

    This is used to add the requirements specified in the rapids_builder table.
    """
    try:
        requires = _suffix_requires(_get_tool_table()["requires"])
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


@lru_cache(1)
def _get_git_commit():
    """Get the current git commit.

    Returns None if git is not in the PATH or if it fails to find the commit.
    """
    git_exists = subprocess.run(["which", "git"], capture_output=True).returncode == 0
    if git_exists:
        try:
            process_output = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True
            )
            return process_output.stdout.decode().strip()
        except subprocess.CalledProcessError:
            pass
    return None


@contextmanager
def _edit_git_commit():
    """
    Temporarily modify the git commit of the package being built.

    This is useful for projects that want to embed the current git commit in the package
    at build time.
    """
    commit_file = _get_tool_table().get("commit-file")
    commit = _get_git_commit()

    if commit_file is not None and commit is not None:
        bkp_commit_file = f".{os.path.basename(commit_file)}.rapids_builder.bak"
        try:
            with open(commit_file) as f:
                lines = f.readlines()

            shutil.move(commit_file, bkp_commit_file)

            with open(commit_file, "w") as f:
                wrote = False
                for line in lines:
                    if "__git_commit__" in line:
                        f.write(f'__git_commit__ = "{commit}"\n')
                        wrote = True
                    else:
                        f.write(line)
                # If no git commit line was found, write it at the end of the file.
                if not wrote:
                    f.write(f'__git_commit__ = "{commit}"\n')

            yield
        finally:
            # Restore by moving rather than writing to avoid any formatting changes.
            shutil.move(bkp_commit_file, commit_file)
    else:
        yield


@contextmanager
def _edit_pyproject():
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
    with _edit_pyproject(), _edit_git_commit():
        return _get_backend().build_wheel(
            wheel_directory, config_settings, metadata_directory
        )


def build_sdist(sdist_directory, config_settings=None):
    with _edit_pyproject(), _edit_git_commit():
        return _get_backend().build_sdist(sdist_directory, config_settings)


# The three hooks below are optional and may not be implemented by the wrapped backend.
# These definitions assume that they will only be called if the wrapped backend
# implements them by virtue of the logic in __init__.py.
def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    with _edit_pyproject(), _edit_git_commit():
        return _get_backend().build_editable(
            wheel_directory, config_settings, metadata_directory
        )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _edit_pyproject():
        return _get_backend().prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    with _edit_pyproject():
        return _get_backend().prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )
