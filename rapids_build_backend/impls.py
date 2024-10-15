# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import platform
import re
import shutil
import subprocess
import typing
import warnings
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module

import rapids_dependency_file_generator
import tomlkit

from . import utils
from .config import Config


def _remove_rapidsai_from_config(
    config_settings: typing.Union[dict[str, typing.Any], None],
) -> typing.Union[dict[str, typing.Any], None]:
    """Filter out settings that begin with ``rapidsai.`` to be passed down to the
    underlying backend, because some backends get confused if you pass them options that
    they don't recognize.
    """
    if not config_settings:
        return None
    return {k: v for k, v in config_settings.items() if not k.startswith("rapidsai.")}


def _parse_matrix(matrix):
    if not matrix:
        return None
    return {
        key: [value] for key, value in (item.split("=") for item in matrix.split(";"))
    }


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
def _get_arch() -> str:
    """Get the arch of the current machine.

    Returns
    -------
    str
        The arch (e.g. "x86_64" or "aarch64")
    """
    plat = platform.machine()
    # RAPIDS projects all use "aarch64" to indicate arm architectures,
    # but arm some systems (like the M1/M2/M3 Macs) report "arm64"
    if plat == "arm64":
        return "aarch64"
    return plat


@lru_cache
def _get_cuda_version():
    """Get the CUDA suffix based on nvcc.

    Returns
    -------
    str or None
        The CUDA major version (e.g., "11")
    """
    nvcc_exists = subprocess.run(["which", "nvcc"], capture_output=True).returncode == 0
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
    return match.groups()


@lru_cache
def _get_cuda_suffix() -> str:
    """Get the CUDA suffix based on nvcc.

    Returns
    -------
    str
        The CUDA suffix (e.g., "-cu11") or an empty string if CUDA could not be
        detected.
    """
    if (version := _get_cuda_version()) is None:
        return ""
    return f"-cu{version[0]}"


@lru_cache
def _get_git_commit() -> typing.Union[str, None]:
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
def _write_git_commits(config, project_name: str):
    """
    Temporarily write the git commit files for the package being built. If the
    `commit-files` config option is not specified, write to `<project_name>/GIT_COMMIT`.

    This is useful for projects that want to embed the current git commit in the package
    at build time.
    """
    commit_files = config.commit_files
    if commit_files is None:
        commit_files = [os.path.join(project_name.replace("-", "_"), "GIT_COMMIT")]
    commit = _get_git_commit() if commit_files else None

    if commit is not None:
        for commit_file in commit_files:
            with open(commit_file, "w") as f:
                f.write(f"{commit}\n")
        try:
            yield
        finally:
            for commit_file in commit_files:
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

    if not config.disable_cuda:
        cuda_version_major, cuda_version_minor = _get_cuda_version()

    # "dependencies.yaml" might not exist in sdists and wouldn't need to... so don't
    # raise an exception if that file can't be found when this runs
    try:
        parsed_config = rapids_dependency_file_generator.load_config_from_file(
            config.dependencies_file
        )
    except FileNotFoundError:
        msg = (
            f"File not found: '{config.dependencies_file}'. If you want "
            "rapids-build-backend to consider dependencies from a dependencies file, "
            "supply an existing file via config setting 'dependencies-file'."
        )
        warnings.warn(msg, stacklevel=2)
        parsed_config = None

    try:
        shutil.copyfile(pyproject_file, bkp_pyproject_file)
        if parsed_config:
            for file_key, file_config in parsed_config.files.items():
                if (
                    rapids_dependency_file_generator.Output.PYPROJECT
                    not in file_config.output
                ):
                    continue
                pyproject_dir = os.path.join(
                    os.path.dirname(config.dependencies_file),
                    file_config.pyproject_dir,
                )
                if not (
                    os.path.exists(pyproject_dir)
                    and os.path.samefile(pyproject_dir, ".")
                ):
                    continue
                matrix = _parse_matrix(config.matrix_entry) or dict(file_config.matrix)
                if not config.disable_cuda:
                    matrix["cuda"] = [f"{cuda_version_major}.{cuda_version_minor}"]
                matrix["arch"] = [_get_arch()]
                rapids_dependency_file_generator.make_dependency_files(
                    parsed_config=parsed_config,
                    file_keys=[file_key],
                    output={rapids_dependency_file_generator.Output.PYPROJECT},
                    matrix=matrix,
                    prepend_channels=[],
                    to_stdout=False,
                )
        if not config.disable_cuda:
            pyproject = utils._get_pyproject()
            project_data = pyproject["project"]
            project_data["name"] += _get_cuda_suffix()
            with open(pyproject_file, "w") as f:
                tomlkit.dump(pyproject, f)
        yield
    finally:
        # Restore by moving rather than writing to avoid any formatting changes.
        shutil.move(bkp_pyproject_file, pyproject_file)


def _check_setup_py(setup_py_contents: str) -> None:
    """
    ``setuptools.get_requires_for_build_wheel()`` executes setup.py if it exists,
    to check for dependencies in ``setup_requires`` (passed to ``setuptools.setup()``).

    That's a problem for rapids-build-backend, as at the point where that's invoked,
    its recalculated list of build dependencies (modified in ``_edit_pyproject()``)
    haven't yet been installed.

    If any of them are imported in ``setup.py``, those imports will fail.

    This function raises an exception if it detects ``setup_requires`` being used in
    a ``setup.py``, to clarify that ``rapids-build-backend`` can't support that case.

    ref: https://github.com/rapidsai/rapids-build-backend/issues/39
    """

    # pattern = "any use of 'setup_requires' on a line that isn't a comment"
    setup_requires_pat = r"^(?!\s*#+).*setup_requires"

    if re.search(setup_requires_pat, setup_py_contents, re.M) is not None:
        raise ValueError(
            "Detected use of 'setup_requires' in a setup.py file. "
            "rapids-build-backend does not support this pattern. Try moving "
            "that list of dependencies into the 'requires' list in the "
            "[tool.rapids-build-backend] table in pyproject.toml."
        )


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
    pyproject = utils._get_pyproject()
    project_name = pyproject["project"]["name"]
    with _edit_pyproject(config), _write_git_commits(config, project_name):
        # Reload the config for a possibly updated tool.rapids-build-backend.requires
        reloaded_config = Config(config_settings=config_settings)
        requires = list(reloaded_config.requires)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_wheel",
        ):
            if config.build_backend == "setuptools.build_meta":
                _check_setup_py(setup_py_contents=utils._get_setup_py())
                # prior to https://github.com/pypa/setuptools/pull/4369 (May 2024),
                # setuptools.build_meta.get_requires_for_build_wheel() automatically
                # added 'wheel' to the build requirements. Adding that manually here,
                # since this code block skips running
                # setuptools.build_meta.get_requires_for_build_wheel().
                #
                # Without this, running 'pip wheel' might result in an error like
                # "error: invalid command 'bdist_wheel'".
                requires.extend(["wheel"])
            else:
                requires.extend(
                    backend.get_requires_for_build_wheel(
                        _remove_rapidsai_from_config(config_settings)
                    )
                )

        return requires


def get_requires_for_build_sdist(config_settings):
    config = Config(config_settings=config_settings)
    pyproject = utils._get_pyproject()
    project_name = pyproject["project"]["name"]
    with _edit_pyproject(config), _write_git_commits(config, project_name):
        # Reload the config for a possibly updated tool.rapids-build-backend.requires
        reloaded_config = Config(config_settings=config_settings)
        requires = list(reloaded_config.requires)

        if hasattr(
            backend := _get_backend(config.build_backend),
            "get_requires_for_build_sdist",
        ):
            if config.build_backend == "setuptools.build_meta":
                _check_setup_py(setup_py_contents=utils._get_setup_py())
            else:
                requires.extend(
                    backend.get_requires_for_build_sdist(
                        _remove_rapidsai_from_config(config_settings)
                    )
                )

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
            if config.build_backend == "setuptools.build_meta":
                _check_setup_py(setup_py_contents=utils._get_setup_py())
            else:
                requires.extend(
                    backend.get_requires_for_build_editable(
                        _remove_rapidsai_from_config(config_settings)
                    )
                )

        return requires


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    config = Config(config_settings=config_settings)
    pyproject = utils._get_pyproject()
    project_name = pyproject["project"]["name"]
    with _edit_pyproject(config), _write_git_commits(config, project_name):
        return _get_backend(config.build_backend).build_wheel(
            wheel_directory,
            _remove_rapidsai_from_config(config_settings),
            metadata_directory,
        )


def build_sdist(sdist_directory, config_settings=None):
    config = Config(config_settings=config_settings)
    pyproject = utils._get_pyproject()
    project_name = pyproject["project"]["name"]
    with _edit_pyproject(config), _write_git_commits(config, project_name):
        return _get_backend(config.build_backend).build_sdist(
            sdist_directory, _remove_rapidsai_from_config(config_settings)
        )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        return _get_backend(config.build_backend).build_editable(
            wheel_directory,
            _remove_rapidsai_from_config(config_settings),
            metadata_directory,
        )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        return _get_backend(config.build_backend).prepare_metadata_for_build_wheel(
            metadata_directory, _remove_rapidsai_from_config(config_settings)
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    config = Config(config_settings=config_settings)
    with _edit_pyproject(config):
        return _get_backend(config.build_backend).prepare_metadata_for_build_editable(
            metadata_directory, _remove_rapidsai_from_config(config_settings)
        )
