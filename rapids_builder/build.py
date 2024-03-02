# Copyright (c) 2024, NVIDIA CORPORATION.

import shutil
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module

import tomli
import tomli_w


# Avoid unnecessary I/O. All modifications of the file are for the wrapped backend.
@lru_cache(1)
def _get_pyproject():
    with open("pyproject.toml", "rb") as f:
        return tomli.load(f)


@lru_cache(1)
def _get_backend():
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


def _supplement_requires(backend_getter, config_settings):
    """
    Add to the list of requirements for the build backend.

    This is used to add requirements that are not defined in the PEP 517
    backend's pyproject.toml file.
    """
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)

    try:
        requires = pyproject["tool"]["rapids_builder"]["requires"]
    except KeyError:
        requires = []

    backend = _get_backend()
    if (getter := getattr(backend, backend_getter, None)) is not None:
        requires.extend(getter(config_settings))
    return requires


def get_requires_for_build_wheel(config_settings):
    return _supplement_requires("get_requires_for_build_wheel", config_settings)


def get_requires_for_build_sdist(config_settings):
    return _supplement_requires("get_requires_for_build_sdist", config_settings)


def get_requires_for_build_editable(config_settings):
    return _supplement_requires("get_requires_for_build_editable", config_settings)


@contextmanager
def _modify_name():
    """
    Temporarily modify the name of the package being built.

    This is used to allow the backend to modify the name of the package
    being built. This is useful for projects that want to build wheels
    with a different name than the package name.
    """
    pyproject_file = "pyproject.toml"
    with open(pyproject_file, "rb") as f:
        pyproject = tomli.load(f)

    pyproject["project"]["name"] = pyproject["project"]["name"] + "-cu11"

    try:
        shutil.move(pyproject_file, pyproject_file + ".bak")
        with open(pyproject_file, "wb") as f:
            tomli_w.dump(pyproject, f)
        yield
    finally:
        # Restore by moving rather than writing to avoid any formatting changes.
        shutil.move(pyproject_file + ".bak", pyproject_file)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    backend = _get_backend()
    with _modify_name():
        return backend.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    backend = _get_backend()
    with _modify_name():
        return backend.build_sdist(sdist_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    backend = _get_backend()
    with _modify_name():
        return backend.build_editable(
            wheel_directory, config_settings, metadata_directory
        )


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    backend = _get_backend()
    with _modify_name():
        return backend.prepare_metadata_for_build_wheel(
            metadata_directory, config_settings
        )


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    backend = _get_backend()
    with _modify_name():
        return backend.prepare_metadata_for_build_editable(
            metadata_directory, config_settings
        )
