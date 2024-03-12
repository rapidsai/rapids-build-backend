# Copyright (c) 2024, NVIDIA CORPORATION.

import glob
import os
import shutil
import zipfile
from email.parser import BytesParser
from pathlib import Path

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
                extra = extra.split("==")[1].replace("'", "").strip()
                extras[extra].add(req.strip())
            else:
                requirements.add(req)
    return data["Name"], requirements, extras


def test_simple_setuptools(tmp_path, env):
    name, requirements, extras = _generate_wheel(tmp_path, env, "simple_setuptools")

    assert name == "simple_setuptools-cu12"
    assert requirements == {"rmm-cu12 >=0.0.0a0"}
    assert extras == {"test": {"dask-cuda ==24.4.*,>=0.0.0a0"}}


def test_simple_scikit_build_core(tmp_path, env):
    name, requirements, extras = _generate_wheel(
        tmp_path, env, "simple_scikit_build_core"
    )

    assert name == "simple_scikit_build_core-cu12"
    assert requirements == {"cupy-cuda12x>=12.0.0"}
    assert extras == {}
