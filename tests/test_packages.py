# Copyright (c) 2024, NVIDIA CORPORATION.

import glob
import os
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from conftest import patch_nvcc_if_needed

DIR = Path(__file__).parent.resolve()


def _generate_wheel(tmp_path, env, package_name):
    """Produce a wheel and extract its metadata for testing."""
    package_dir = tmp_path / "pkg"
    shutil.copytree(DIR / "packages" / package_name, package_dir)

    env.wheel(str(package_dir))

    wheel = glob.glob(str(package_dir / "*.whl"))[0]
    p = BytesParser()
    with zipfile.ZipFile(wheel, "r") as z:
        metadata = next(f for f in z.namelist() if os.path.basename(f) == "METADATA")
        with z.open(metadata) as f:
            data = p.parse(f)

    # Split requirements by extras
    all_requirements = data.get_all("Requires-Dist")
    requirements = set()
    extras = {e: set() for e in data.get_all("Provides-Extra", [])}
    if all_requirements is not None:
        for req in all_requirements:
            if ";" in req:
                req, extra = req.split(";")
                extra = extra.split("==")[1].replace("'", "").replace('"', "").strip()
                extras[extra].add(req.strip())
            else:
                requirements.add(req)
    return data["Name"], requirements, extras


@pytest.mark.parametrize("nvcc_version", ["11", "12"])
def test_simple_setuptools(tmp_path, env, nvcc_version):
    with patch_nvcc_if_needed(nvcc_version):
        name, requirements, extras = _generate_wheel(tmp_path, env, "simple_setuptools")

    assert name == f"simple_setuptools-cu{nvcc_version}"
    assert requirements == {f"rmm-cu{nvcc_version} >=0.0.0a0"}
    assert extras == {"test": {"dask-cuda ==24.4.*,>=0.0.0a0"}}


@pytest.mark.parametrize("nvcc_version", ["11", "12"])
def test_simple_scikit_build_core(tmp_path, env, nvcc_version):
    with patch_nvcc_if_needed(nvcc_version):
        name, requirements, extras = _generate_wheel(
            tmp_path, env, "simple_scikit_build_core"
        )

    assert name == f"simple_scikit_build_core-cu{nvcc_version}"
    assert requirements == {f"cupy-cuda{nvcc_version}x>=12.0.0"}
    if nvcc_version == "11":
        assert extras == {"jit": {"ptxcompiler-cu11"}}
    else:
        assert extras == {}
