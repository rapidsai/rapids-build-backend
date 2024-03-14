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
            "tool.rapids-build-backend table. Make sure you specified the right "
            "optional dependency in your build-system.requires entry for "
            "rapids-build-backend."
        )


@lru_cache
def _get_cuda_major(require_cuda=False):
    """Get the CUDA suffix based on nvcc.

    Parameters
    ----------
    require_cuda : bool
        If True, raise an exception if nvcc is not in the PATH. If False, return None.

    Returns
    -------
    str or None
        The CUDA major version (e.g., "11") or None if CUDA could not be detected.
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


@lru_cache
def _get_cuda_suffix(require_cuda=False):
    """Get the CUDA suffix based on nvcc.

    Parameters
    ----------
    require_cuda : bool
        If True, raise an exception if CUDA could not be detected. If False, return an
        empty string.

    Returns
    -------
    str
        The CUDA suffix (e.g., "-cu11") or an empty string if CUDA could not be
        detected.
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


def _add_cuda_suffix(req, cuda_suffix, cuda_major):
    req = Requirement(req)
    if req.name == "cupy" and cuda_major is not None:
        req.name += f"-cuda{cuda_major}x"
    elif req.name in _VERSIONED_RAPIDS_WHEELS:
        req.name += cuda_suffix

    return str(req)


def _add_alpha_specifier(req):
    req = Requirement(req)
    if (
        req.name in _VERSIONED_RAPIDS_WHEELS or req.name in _UNVERSIONED_RAPIDS_WHEELS
    ) and req.name not in _CUDA_11_ONLY_WHEELS:
        req.specifier &= SpecifierSet(">=0.0.0a0")
    return str(req)


def _process_dependencies(config, dependencies=None):
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
    # Note that this implementation is currently suboptimal, in each step to allow the
    # steps to be more freely composable based on options. We could optimize by using a
    # single loop with multiple nested conditionals, but that would make the logic
    # harder to understand and modify. The performance of this code should be negligible
    # in the context of a build anyway.
    dependencies = dependencies or config.requires

    # Step 1: Filter out CUDA 11-only wheels if we're not in a CUDA 11 build. Skip this
    # step if if we were unable to detect a CUDA version.
    major = _get_cuda_major(config.require_cuda)
    if major is not None and major != "11":
        dependencies = filter(
            lambda dep: dep not in _CUDA_11_ONLY_WHEELS,
            dependencies,
        )

    # Step 2: Allow nightlies of RAPIDS packages except in release builds. Do this
    # before suffixing the names so that lookups in _add_alpha_specifier are accurate.
    if config.allow_nightly_deps:
        dependencies = map(
            _add_alpha_specifier,
            dependencies,
        )

    # Step 3: Add the CUDA suffix to any versioned RAPIDS wheels. This step may be
    # explicitly skipped by setting the disable_cuda_suffix option to True, or
    # implicitly skipped if we were unable to detect a CUDA version and require_cuda was
    # False.
    if not config.disable_cuda_suffix:
        suffix = _get_cuda_suffix(config.require_cuda)
        # If we can't determine the local CUDA version then we can skip this step
        if suffix:
            dependencies = map(
                lambda dep: _add_cuda_suffix(dep, suffix, major),
                dependencies,
            )

    return list(dependencies)


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
        bkp_commit_file = os.path.join(
            os.path.dirname(commit_file),
            f".{os.path.basename(commit_file)}.rapids-build-backend.bak",
        )
        try:
            try:
                shutil.move(commit_file, bkp_commit_file)
            except FileNotFoundError:
                bkp_commit_file = None

            with open(commit_file, "w") as f:
                f.write(f"{commit}\n")

            yield
        finally:
            # Restore by moving rather than writing to avoid any formatting changes.
            if bkp_commit_file:
                shutil.move(bkp_commit_file, commit_file)
            else:
                os.unlink(commit_file)
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
        project_data["dependencies"] = _process_dependencies(
            config, project_data["dependencies"]
        )

    optional_dependencies = pyproject["project"].get("optional-dependencies")
    if optional_dependencies is not None:
        project_data["optional-dependencies"] = {
            extra: _process_dependencies(config, deps)
            for extra, deps in optional_dependencies.items()
        }

    pyproject_file = "pyproject.toml"
    bkp_pyproject_file = ".pyproject.toml.rapids-build-backend.bak"
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
# backend's hooks to the rapids-build-backend hooks, but that's not a big deal because
# these functions only executed by the build frontend and are not user-facing. This
# approach also ignores the possibility that the backend may not define certain optional
# hooks because these definitions assume that they will only be called if the wrapped
# backend implements them by virtue of the logic in rapids-build-backend's build module
# (the actual build backend, which conditionally imports these functions).
def get_requires_for_build_wheel(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        requires = _process_dependencies(config)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_wheel",
        ):
            requires.extend(backend.get_requires_for_build_wheel(config_settings))

        return requires


def get_requires_for_build_sdist(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        requires = _process_dependencies(config)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_sdist",
        ):
            requires.extend(backend.get_requires_for_build_sdist(config_settings))

        return requires


def get_requires_for_build_editable(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        requires = _process_dependencies(config)

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
    with _edit_pyproject(config):
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
