# Copyright (c) 2024, NVIDIA CORPORATION.

import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest
import tomli
import tomli_w
from packaging.version import parse as parse_version

BASE = Path(__file__).parent.parent.resolve()


class VEnv:
    """Manage a virtual environment.

    This class is a thin wrapper around the `venv` module that provides a more
    convenient interface for creating environments and installing packages into them.
    It also provides a way to build wheels in the environment. Most of the time this
    class will not really be necessary since pip builds are isolated by default, but
    it provides an easy way to create an isolated environment in which to perform builds
    with isolation disabled if necessary.
    """

    def __init__(self, *, env_dir, wheelhouse, cache_dir):
        self.env_dir = env_dir
        self.cache_dir = cache_dir
        self.executable = env_dir / "bin" / "python"
        self.wheelhouse = wheelhouse
        venv.create(
            env_dir,
            clear=True,
            with_pip=True,
        )

    def install(self, *args):
        subprocess.run(
            [
                self.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--find-links",
                str(self.wheelhouse),
                "--cache-dir",
                self.cache_dir,
                *args,
            ],
            check=True,
        )

    def wheel(self, package_dir, *args):
        subprocess.run(
            [
                self.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--wheel-dir",
                package_dir,
                "--find-links",
                str(self.wheelhouse),
                "--cache-dir",
                self.cache_dir,
                package_dir,
                *args,
            ],
            check=True,
        )


@pytest.fixture(scope="session")
def pip_cache(tmp_path_factory):
    """A shared cache for all pip calls across environments."""
    return tmp_path_factory.mktemp("pip_cache")


@pytest.fixture(scope="session")
def wheelhouse(tmp_path_factory, pip_cache):
    """A PEP 517 wheelhouse containing the local copy of rapids_builder."""
    wheelhouse = tmp_path_factory.mktemp("wheelhouse")

    # Build the rapids-builder wheel in a temporary directory where we can bump the
    # version to ensure that it is preferred to any other available wheels.
    rapids_builder_build_dir = tmp_path_factory.mktemp("rapids_builder_build_dir")

    shutil.copytree(
        BASE,
        rapids_builder_build_dir,
        ignore=shutil.ignore_patterns("tests*"),
        dirs_exist_ok=True,
    )

    pyproject_file = rapids_builder_build_dir / "pyproject.toml"
    with open(pyproject_file, "rb") as f:
        pyproject = tomli.load(f)
    project_data = pyproject["project"]

    version = parse_version(project_data["version"])
    project_data["version"] = f"{version.major + 1}.{version.minor}.{version.micro}"
    with open(pyproject_file, "wb") as f:
        tomli_w.dump(pyproject, f)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--wheel-dir",
            str(wheelhouse),
            "--cache-dir",
            pip_cache,
            f"{rapids_builder_build_dir}",
        ],
        check=True,
    )

    return wheelhouse


@pytest.fixture(scope="session")
def env(tmp_path_factory, wheelhouse, pip_cache):
    """The standard environment that should be used by most tests."""
    path = tmp_path_factory.mktemp("env")
    return VEnv(env_dir=path, wheelhouse=wheelhouse, cache_dir=pip_cache)


@pytest.fixture()
def isolated_env(tmp_path, wheelhouse, pip_cache):
    """An environment that should be used when build isolation is disabled.

    This fixture will create a new environment with the same wheelhouse and cache each
    time that it is used, ensuring that we don't accidentally leak state between tests.
    """
    return VEnv(env_dir=tmp_path, wheelhouse=wheelhouse, cache_dir=pip_cache)
