# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module

import tomli_w
import yaml
from rapids_dependency_file_generator.cli import generate_matrix
from rapids_dependency_file_generator.constants import default_pyproject_dir
from rapids_dependency_file_generator.rapids_dependency_file_generator import (
    get_requested_output_types,
    make_dependency_files,
)

from . import utils
from .config import Config


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
def _get_cuda_version(require_cuda=False):
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

        match = re.search(r"release (\d+)\.(\d+)", output_lines[3])
        if match is None:
            raise ValueError("Failed to parse CUDA version from nvcc output.")
        return match.group(1), match.group(2)
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
    if (version := _get_cuda_version(require_cuda)) is None:
        return ""
    return f"-cu{version[0]}"


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
    pyproject_file = "pyproject.toml"
    bkp_pyproject_file = ".pyproject.toml.rapids-build-backend.bak"

    cuda_version = _get_cuda_version(config.require_cuda)

    with open(config.dependencies_file) as f:
        parsed_config = yaml.load(f, Loader=yaml.FullLoader)
    files = {}
    for file_key, file_config in parsed_config["files"].items():
        if "pyproject" not in get_requested_output_types(file_config["output"]):
            continue
        pyproject_dir = os.path.join(
            os.path.dirname(config.dependencies_file),
            file_config.get("pyproject_dir", default_pyproject_dir),
        )
        if not os.path.exists(pyproject_dir):
            continue
        if not os.path.samefile(pyproject_dir, "."):
            continue
        file_config["output"] = ["pyproject"]
        if config.matrix:
            file_config["matrix"] = generate_matrix(config.matrix)
        if cuda_version is not None:
            file_config.setdefault("matrix", {})["cuda"] = [
                f"{cuda_version[0]}.{cuda_version[1]}"
            ]
        files[file_key] = file_config
    parsed_config["files"] = files

    try:
        shutil.copyfile(pyproject_file, bkp_pyproject_file)
        make_dependency_files(parsed_config, config.dependencies_file, False)
        pyproject = utils._get_pyproject()
        project_data = pyproject["project"]
        project_data["name"] += _get_cuda_suffix(config.require_cuda)
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
        # Reload the config for a possibly updated tool.rapids-build-backend.requires
        reloaded_config = Config(config_settings=config_settings)
        requires = list(reloaded_config.requires)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_wheel",
        ):
            requires.extend(backend.get_requires_for_build_wheel(config_settings))

        return requires


def get_requires_for_build_sdist(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config), _edit_git_commit(config):
        # Reload the config for a possibly updated tool.rapids-build-backend.requires
        reloaded_config = Config(config_settings=config_settings)
        requires = list(reloaded_config.requires)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_sdist",
        ):
            requires.extend(backend.get_requires_for_build_sdist(config_settings))

        return requires


def get_requires_for_build_editable(config_settings):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        # Reload the config for a possibly updated tool.rapids-build-backend.requires
        reloaded_config = Config(config_settings=config_settings)
        requires = list(reloaded_config.requires)

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
