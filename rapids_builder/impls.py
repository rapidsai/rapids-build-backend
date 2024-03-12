# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module

import tomli_w
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from .config import Config
from .utils import _get_pyproject


@lru_cache
def _get_backend(build_backend):
    """Get the wrapped build backend specified in pyproject.toml."""
    try:
        return import_module(build_backend)
    except ImportError:
        raise ValueError(
            "Could not import build backend specified in pyproject.toml's "
            "tool.rapids_builder table. Make sure you specified the right optional "
            "dependency in your build-system.requires entry for rapids_builder."
        )


@lru_cache
def _get_cuda_major(require_cuda=False):
    """Get the CUDA suffix based on nvcc.

    Parameters
    ----------
    require_cuda : bool
        If True, raise an exception if nvcc is not in the PATH. If False, return None.
    """
    try:
        nvcc_exists = (
            subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0
        )
        if not nvcc_exists:
            raise ValueError(
                "Could not determine the CUDA version. Make sure nvcc is in your PATH."
            )

        try:
            process_output = subprocess.run(["nvcc", "--version"], capture_output=True)
        except subprocess.CalledProcessError as e:
            raise ValueError("Failed to get version from nvcc.") from e

        output_lines = process_output.stdout.decode().splitlines()

        match = re.search(r"release (\d+)", output_lines[3])
        if match is None:
            raise ValueError("Failed to parse CUDA version from nvcc output.")
        return match.group(1)
    except Exception:
        if not require_cuda:
            return None
        raise


def _get_cuda_suffix(require_cuda=False):
    """Get the CUDA suffix based on nvcc.

    Parameters
    ----------
    require_cuda : bool
        If True, raise an exception if CUDA could not be detected. If False, return an
        empty string.
    """
    if (major := _get_cuda_major(require_cuda)) is None:
        return ""
    return f"-cu{major}"


# Wheels with a CUDA suffix.
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

# Wheels without a CUDA suffix.
_UNVERSIONED_RAPIDS_WHEELS = [
    "dask-cuda",
    "rapids-dask-dependency",
]

# Wheels that don't release regular alpha versions
_CUDA_11_ONLY_WHEELS = (
    "ptxcompiler",
    "cubinlinker",
)


def _suffix_dependencies(config, dependencies=None):
    """Add the CUDA suffix to any versioned RAPIDS wheels in dependencies.

    If dependencies is None, then config.requires is used.

    Parameters
    ----------
    config : Config
        The project's configuration.
    dependencies : list of str, optional
        The dependencies to suffix. If None, then config.requires is used.

    Returns
    -------
    list of str
        The dependencies with the CUDA suffix added to any versioned RAPIDS wheels.
    """
    dependencies = dependencies or config.requires
    new_dependencies = []
    suffix = _get_cuda_suffix(config.require_cuda)
    for req in dependencies:
        req = Requirement(req)

        # cupy is a special case because it's not a RAPIDS wheel. If we can't
        # determine the local CUDA version, then we fall back to making the sdist of
        # cupy on PyPI the dependency.
        major = _get_cuda_major(config.require_cuda)
        if req.name == "cupy" and major is not None:
            req.name += f"-cuda{major}x"
        else:
            is_cuda_11_wheel = any(req.name == w for w in _CUDA_11_ONLY_WHEELS)
            if is_cuda_11_wheel:
                # These wheels only exist for CUDA 11.
                if major != "11":
                    continue

            is_versioned_wheel = any(req.name == w for w in _VERSIONED_RAPIDS_WHEELS)
            is_unversioned_wheel = any(
                req.name == w for w in _UNVERSIONED_RAPIDS_WHEELS
            )

            if is_versioned_wheel:
                req.name += suffix

            # Allow nightlies of RAPIDS packages except in release builds. Also,
            # ptxcompiler and cubinlinker don't release regular alpha versions.
            if (
                (is_versioned_wheel or is_unversioned_wheel)
                and not config.only_release_deps
                and not is_cuda_11_wheel
            ):
                req.specifier &= SpecifierSet(">=0.0.0a0")

        new_dependencies.append(str(req))
    return new_dependencies


@lru_cache
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
def _edit_git_commit(config):
    """
    Temporarily modify the git commit of the package being built.

    This is useful for projects that want to embed the current git commit in the package
    at build time.
    """
    commit_file = config.commit_file
    commit = _get_git_commit()

    if commit_file != "" and commit is not None:
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
def _edit_pyproject(config):
    """
    Temporarily modify the name and dependencies of the package being built.

    This is used to allow the backend to modify the name of the package
    being built. This is useful for projects that want to build wheels
    with a different name than the package name.
    """
    pyproject = _get_pyproject()
    project_data = pyproject["project"]
    project_data["name"] += _get_cuda_suffix(config.require_cuda)

    dependencies = pyproject["project"].get("dependencies")
    if dependencies is not None:
        project_data["dependencies"] = _suffix_dependencies(
            config, project_data["dependencies"]
        )

    optional_dependencies = pyproject["project"].get("optional-dependencies")
    if optional_dependencies is not None:
        project_data["optional-dependencies"] = {
            extra: _suffix_dependencies(config, deps)
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


# The hooks in this file could be defined more programmatically by iterating over the
# backend's attributes, but it's simpler to just define them explicitly and avoids any
# potential issues with assuming the right pyproject.toml is readable at import time (we
# need to load pyproject.toml to know what the build backend is). Note that this also
# prevents us from using something like functools.wraps to copy the docstrings from the
# backend's hooks to the rapids_builder hooks, but that's not a big deal because these
# functions only executed by the build frontend and are not user-facing. This approach
# also ignores the possibility that the backend may not define certain optional hooks
# because these definitions assume that they will only be called if the wrapped backend
# implements them by virtue of the logic in rapids_builder's build module (the actual
# build backend, which conditionally imports these functions).
def get_requires_for_build_wheel(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        requires = _suffix_dependencies(config)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_wheel",
        ):
            requires.extend(backend.get_requires_for_build_wheel(config_settings))

        return requires


def get_requires_for_build_sdist(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        requires = _suffix_dependencies(config)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_sdist",
        ):
            requires.extend(backend.get_requires_for_build_sdist(config_settings))

        return requires


def get_requires_for_build_editable(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        requires = _suffix_dependencies(config)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_editable",
        ):
            requires.extend(backend.get_requires_for_build_editable(config_settings))

        return requires


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        return _get_backend(config.build_backend).build_wheel(
            wheel_directory, config_settings, metadata_directory
        )


def build_sdist(sdist_directory, config_settings=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        return _get_backend(config.build_backend).build_sdist(
            sdist_directory, config_settings
        )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        return _get_backend(config.build_backend).build_editable(
            wheel_directory, config_settings, metadata_directory
        )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        return _get_backend(config.build_backend).prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        return _get_backend(config.build_backend).prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )
