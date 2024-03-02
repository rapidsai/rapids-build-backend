# Copyright (c) 2024, NVIDIA CORPORATION.

from contextlib import contextmanager
from importlib import import_module

import tomli
import tomli_w


def _get_backend():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

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


def _supplement_requires():
    """
    Add to the list of requirements for the build backend.

    This is used to add requirements that are not defined in the PEP 517
    backend's pyproject.toml file.
    """
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    # TODO: These should all probably not be optional keys.
    rapids_builder_config = pyproject.get("tool", {}).get("rapids_builder", {})
    # Default to using scikit-build-core for RAPIDS projects.
    return rapids_builder_config.get("requires", [])


def get_requires_for_build_wheel(config_settings):
    backend = _get_backend()
    requires = _supplement_requires()
    if hasattr(backend, "get_requires_for_build_wheel"):
        requires.extend(backend.get_requires_for_build_wheel(config_settings))
    return requires


def get_requires_for_build_sdist(config_settings):
    backend = _get_backend()
    requires = _supplement_requires()
    if hasattr(backend, "get_requires_for_build_sdist"):
        requires.extend(backend.get_requires_for_build_sdist(config_settings))
    return requires


def get_requires_for_build_editable(config_settings):
    backend = _get_backend()
    requires = _supplement_requires()
    if hasattr(backend, "get_requires_for_build_editable"):
        requires.extend(backend.get_requires_for_build_editable(config_settings))
    return requires


@contextmanager
def _modify_name():
    """
    Temporarily modify the name of the package being built.

    This is used to allow the backend to modify the name of the package
    being built. This is useful for projects that want to build wheels
    with a different name than the package name.
    """
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    original_pyproject = pyproject.copy()
    original_name = pyproject["project"]["name"]
    pyproject["project"]["name"] = original_name + "-cu11"

    try:
        with open("pyproject.toml", "wb") as f:
            tomli_w.dump(pyproject, f)
        yield
    finally:
        with open("pyproject.toml", "wb") as f:
            tomli_w.dump(original_pyproject, f)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    backend = _get_backend()
    with _modify_name():
        return backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    backend = _get_backend()
    with _modify_name():
        return backend.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    backend = _get_backend()
    if hasattr(backend, "prepare_metadata_for_build_wheel"):
        return backend.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


# TODO: Check if this also needs to be in the __getattr__ block.
def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    backend = _get_backend()
    if hasattr(backend, "prepare_metadata_for_build_editable"):
        return backend.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )


# Use __getattr__ to delay lookup on build_editable until after we know the build
# backend has already been installed. This is necessary because unlike the other
# optional hooks, just having this method defined changes whether the build frontend
# think the backend supports editable installs, whereas the prepare_* functions can be
# defined as no-ops and have no effect on behavior.
# TODO: Verify whether the above is true.
# TODO: Cleaner solution: define all hooks in a separate module, then just return those
# conditionally in this __getattr__ depending on whether the build backend supports
# them.
def __getattr__(name):
    if name == "build_editable":
        backend = _get_backend()

        if hasattr(backend, "build_editable"):

            def build_editable(
                wheel_directory, config_settings=None, metadata_directory=None
            ):
                with _modify_name("rapids-build-wheel"):
                    return backend.build_editable(
                        wheel_directory, config_settings, metadata_directory
                    )

            return build_editable

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
