# Copyright (c) 2024, NVIDIA CORPORATION.

import glob
import os
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from conftest import patch_nvcc_if_needed, setup_project

DIR = Path(__file__).parent.resolve()


def _generate_wheel(tmp_path, jinja_environment, env, template, template_args=None):
    """Produce a wheel and extract its metadata for testing."""
    package_dir = setup_project(
        tmp_path,
        jinja_environment,
        template,
        template_args,
    )
    if template_args["build_backend"] == "scikit_build_core.build":
        # Temp hack, also copy CMakeLists.txt manually
        shutil.copyfile(
            DIR / "config_packages" / "templates" / "CMakeLists.txt",
            package_dir / "CMakeLists.txt",
        )

    output = env.wheel(str(package_dir), "-v")

    # Parse the build dependencies from the pip output since they won't be encoded
    # anywhere in the wheel itself. Since this also means we will capture all transitive
    # dependencies, checks of the build dependencies can only rely on checking that the
    # known requirements are a subset of the installed build dependencies.
    lines = output.stdout.decode().splitlines()
    build_requires = set()
    started = False
    for line in lines:
        if "Running command pip subprocess to install backend dependencies" in line:
            started = True
        if started:
            if "Installing backend dependencies: finished" in line:
                break
            if "Collecting" in line:
                build_requires.add(line.replace("Collecting ", "").replace(" ", ""))

    # Extract metadata from the wheel file.
    wheel = glob.glob(str(package_dir / "*.whl"))[0]
    p = BytesParser()
    with zipfile.ZipFile(wheel, "r") as z:
        metadata = next(f for f in z.namelist() if os.path.basename(f) == "METADATA")
        with z.open(metadata) as f:
            data = p.parse(f)

    # Split requirements by extras.
    all_requirements = data.get_all("Requires-Dist")
    requirements = set()
    extras = {e: set() for e in data.get_all("Provides-Extra", [])}
    if all_requirements is not None:
        for req in all_requirements:
            if ";" in req:
                req, extra = req.split(";")
                extra = extra.split("==")[1].replace("'", "").replace('"', "").strip()
                extras[extra].add(req.replace(" ", ""))
            else:
                requirements.add(req.replace(" ", ""))
    return data["Name"], build_requires, requirements, extras


@pytest.mark.parametrize("nvcc_version", ["11", "12"])
def test_simple_setuptools(tmp_path, jinja_environment, env, nvcc_version):
    template_args = {
        "name": "simple_setuptools",
        "dependencies": ["rmm"],
        "extras": {"test": ["dask-cuda==24.4.*"]},
        "build_requires": ["rmm"],
        "build_backend": "setuptools.build_meta",
        "rapids_builder_extra": "setuptools",
    }
    with patch_nvcc_if_needed(nvcc_version):
        name, build_requires, requirements, extras = _generate_wheel(
            tmp_path,
            jinja_environment,
            env,
            "pyproject.toml",
            template_args,
        )

    assert name == f"simple_setuptools-cu{nvcc_version}"
    assert {f"rmm-cu{nvcc_version}>=0.0.0a0"}.issubset(build_requires)
    assert requirements == {f"rmm-cu{nvcc_version}>=0.0.0a0"}
    assert extras == {"test": {"dask-cuda==24.4.*,>=0.0.0a0"}}


@pytest.mark.parametrize("nvcc_version", ["11", "12"])
def test_simple_scikit_build_core(tmp_path, jinja_environment, env, nvcc_version):
    template_args = {
        "name": "simple_scikit_build_core",
        "dependencies": ["cupy>=12.0.0"],
        "extras": {"jit": ["ptxcompiler"]},
        "build_requires": ["rmm==24.4.*"],
        "build_backend": "scikit_build_core.build",
        "rapids_builder_extra": "scikit-build-core",
    }
    with patch_nvcc_if_needed(nvcc_version):
        name, build_requires, requirements, extras = _generate_wheel(
            tmp_path,
            jinja_environment,
            env,
            "pyproject.toml",
            template_args,
        )

    assert name == f"simple_scikit_build_core-cu{nvcc_version}"
    assert {f"rmm-cu{nvcc_version}==24.4.*,>=0.0.0a0"}.issubset(build_requires)
    assert requirements == {f"cupy-cuda{nvcc_version}x>=12.0.0"}
    if nvcc_version == "11":
        assert extras == {"jit": {"ptxcompiler-cu11"}}
    else:
        assert extras == {}
