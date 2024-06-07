# Copyright (c) 2024, NVIDIA CORPORATION.

import glob
import os
import zipfile
from email.parser import BytesParser

import pytest
from conftest import generate_from_template, patch_nvcc_if_needed


def _generate_wheel(env, package_dir):
    """Produce a wheel and extract its metadata for testing."""
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
def test_simple_setuptools(tmp_path, env, nvcc_version):
    template_args = {
        "name": "simple_setuptools",
        "dependencies": {"cu11": ["rmm-cu11>=0.0.0a0"], "cu12": ["rmm-cu12>=0.0.0a0"]},
        "extras": {"test": {"common": ["dask-cuda==24.4.*,>=0.0.0a0"]}},
        "build_requires": {
            "cu11": ["rmm-cu11>=0.0.0a0"],
            "cu12": ["rmm-cu12>=0.0.0a0"],
        },
        "build_backend": "setuptools.build_meta",
        "build_backend_package": "setuptools",
    }

    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir / "simple_setuptools")

    generate_from_template(package_dir, "dependencies.yaml", template_args)
    generate_from_template(package_dir, "pyproject.toml", template_args)

    with patch_nvcc_if_needed(nvcc_version):
        name, build_requires, requirements, extras = _generate_wheel(env, package_dir)

    assert name == f"simple_setuptools-cu{nvcc_version}"
    assert {f"rmm-cu{nvcc_version}>=0.0.0a0"}.issubset(build_requires)
    assert requirements == {f"rmm-cu{nvcc_version}>=0.0.0a0"}
    assert extras == {"test": {"dask-cuda==24.4.*,>=0.0.0a0"}}


@pytest.mark.parametrize("nvcc_version", ["11", "12"])
def test_simple_scikit_build_core(tmp_path, env, nvcc_version):
    template_args = {
        "name": "simple_scikit_build_core",
        "dependencies": {
            "cu11": ["cupy-cuda11x>=12.0.0"],
            "cu12": ["cupy-cuda12x>=12.0.0"],
        },
        "extras": {"jit": {"cu11": ["ptxcompiler-cu11"]}},
        # having multiple >= constraints is weird, but it's here to avoid
        # https://github.com/rapidsai/rapids-build-backend/pull/40#issuecomment-2152949912
        # (by always pulling from from pypi.anaconda.org/rapidsai-wheels-nightly)
        # while still testing that rapids-build-backend preserves all the dependency
        # specifiers
        "build_requires": {
            "cu11": ["rmm-cu11>=24.4.0,>=0.0.0a0"],
            "cu12": ["rmm-cu12>=24.4.0,>=0.0.0a0"],
        },
        "build_backend": "scikit_build_core.build",
        "build_backend_package": "scikit-build-core",
    }

    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir / "simple_scikit_build_core")

    generate_from_template(package_dir, "dependencies.yaml", template_args)
    generate_from_template(package_dir, "pyproject.toml", template_args)
    generate_from_template(package_dir, "CMakeLists.txt")

    with patch_nvcc_if_needed(nvcc_version):
        name, build_requires, requirements, extras = _generate_wheel(env, package_dir)

    assert name == f"simple_scikit_build_core-cu{nvcc_version}"
    # note: this is also testing that the dependency specifiers were rearranged
    #       (i.e. that >=0.0.0a0 comes before >=24.4.0)
    assert {f"rmm-cu{nvcc_version}>=0.0.0a0,>=24.4.0"}.issubset(build_requires)
    assert requirements == {f"cupy-cuda{nvcc_version}x>=12.0.0"}
    if nvcc_version == "11":
        assert extras == {"jit": {"ptxcompiler-cu11"}}
    else:
        assert extras == {"jit": set()}
