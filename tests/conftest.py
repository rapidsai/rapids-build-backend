# Copyright (c) 2024, NVIDIA CORPORATION.

import os
import shutil
import subprocess
import sys
import tempfile
import venv
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import pytest
import tomlkit
from jinja2 import Environment, FileSystemLoader
from packaging.version import parse as parse_version

from rapids_build_backend.impls import _get_cuda_version

DIR = Path(__file__).parent.parent.resolve()


_NVCC = """\
#!/usr/bin/env python
# Spoof nvcc to return a version of choice

# All we care about is the fourth line of the nvcc output containing the version.
print('''\



Cuda compilation tools, release {version}.0, V{version}.0.0
''')
"""


@lru_cache
def _nvcc_tmp_dir():
    """Temporary directory where spoofed nvcc executables will be placed."""
    return tempfile.mkdtemp()


@pytest.fixture(scope="session")
def _cleanup_nvcc(request):
    """Tell pytest to clean up the temporary nvcc files after the session."""

    def delete_patch_files():
        _nvcc_tmp_dir().cleanup()

    request.addfinalizer(delete_patch_files)


@lru_cache
def _create_nvcc(nvcc_version):
    """Generate a Python script that spoofs the output of nvcc for a desired version."""
    fn = os.path.join(_nvcc_tmp_dir(), f"nvcc{nvcc_version}", "nvcc")
    os.makedirs(os.path.dirname(fn))
    with open(fn, "w") as f:
        f.write(_NVCC.format(version=nvcc_version))
    os.chmod(fn, 0o755)
    return fn

@pytest.fixture
def examples_dir():
    """Directory with test project files."""
    return Path(__file__).parent / "examples"


@contextmanager
def patch_nvcc_if_needed(nvcc_version):
    """Patch the PATH to insert a spoofed nvcc that returns the desired version."""
    path = os.environ["PATH"]
    try:
        # Only create a patch if one is required. In addition to reducing overhead, this
        # also ensures that we test the real nvcc and don't mask any relevant errors.
        try:
            if _get_cuda_version()[0] != nvcc_version:
                raise ValueError
        except ValueError:
            nvcc = _create_nvcc(nvcc_version)
            os.environ["PATH"] = os.pathsep.join(
                [os.path.dirname(nvcc), os.environ["PATH"]]
            )
        yield
    finally:
        os.environ["PATH"] = path


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
        return subprocess.run(
            [
                self.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple",
                "--find-links",
                str(self.wheelhouse),
                "--cache-dir",
                self.cache_dir,
                *args,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
        )

    def wheel(self, package_dir, *args):
        try:
            return subprocess.run(
                [
                    self.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--disable-pip-version-check",
                    "--extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple",
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
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print("Subprocess failed with output:")
            print(e.stdout.decode("utf-8"))
            raise


@pytest.fixture(scope="session")
def pip_cache(tmp_path_factory):
    """A shared cache for all pip calls across environments."""
    return tmp_path_factory.mktemp("pip_cache")


@pytest.fixture(scope="session")
def wheelhouse(tmp_path_factory, pip_cache):
    """A PEP 517 wheelhouse containing the local copy of rapids-build-backend."""
    wheelhouse = tmp_path_factory.mktemp("wheelhouse")

    # Build the rapids-builder wheel in a temporary directory where we can bump the
    # version to ensure that it is preferred to any other available wheels.
    rapids_build_backend_build_dir = tmp_path_factory.mktemp(
        "rapids_build_backend_build_dir"
    )

    shutil.copytree(
        DIR,
        rapids_build_backend_build_dir,
        ignore=shutil.ignore_patterns("tests*"),
        dirs_exist_ok=True,
    )

    pyproject_file = rapids_build_backend_build_dir / "pyproject.toml"
    with open(pyproject_file) as f:
        pyproject = tomlkit.load(f)
    project_data = pyproject["project"]

    version = parse_version(project_data["version"])
    project_data["version"] = f"{version.major + 1}.{version.minor}.{version.micro}"
    with open(pyproject_file, "w") as f:
        tomlkit.dump(pyproject, f)

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
            f"{rapids_build_backend_build_dir}",
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


@lru_cache
def jinja_environment():
    template_dir = os.path.join(
        os.path.dirname(__file__),
        "templates/",
    )
    return Environment(
        loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
    )


_TEMPLATE_DEFAULTS = {
    "pyproject.toml": {
        "name": "pkg",
        "dependencies": {},
        "extras": {},
        "build_requires": {},
        "flags": {},
        "build_backend": "setuptools.build_meta",
    }
}


def generate_from_template(package_dir, template_name, template_args=None):
    default_template_args = _TEMPLATE_DEFAULTS.get(template_name, {})
    template = jinja_environment().get_template(template_name)

    template_args = default_template_args | (template_args or {})
    content = template.render(**template_args)
    output_file = os.path.join(package_dir, template_name)
    with open(output_file, mode="w", encoding="utf-8") as f:
        f.write(content)
