# Copyright (c) 2024, NVIDIA CORPORATION.

import glob
import os
import shutil
import zipfile
from collections import defaultdict
from email.parser import BytesParser
from pathlib import Path

DIR = Path(__file__).parent.resolve()


def test_simple_setuptools(tmp_path, builder_env):
    package_dir = tmp_path / "pkg"
    shutil.copytree(DIR / "packages" / "simple_setuptools", package_dir)

    builder_env.wheel(str(package_dir))

    wheel = glob.glob(str(package_dir / "*.whl"))[0]
    p = BytesParser()
    with zipfile.ZipFile(wheel, "r") as z:
        metadata = next(f for f in z.namelist() if os.path.basename(f) == "METADATA")
        with z.open(metadata) as f:
            data = p.parse(f)

    assert data["Name"] == "simple_setuptools-cu12"

    # Split requirements by extras
    all_requirements = data.get_all("Requires-Dist")
    requirements = set()
    optionals = defaultdict(set)
    if all_requirements is not None:
        for req in all_requirements:
            if ";" in req:
                req, extra = req.split(";")
                extra = extra.split("==")[1].replace("'", "").strip()
                optionals[extra].add(req.strip())
            else:
                requirements.add(req)

    assert requirements == {"rmm-cu12 >=0.0.0a0"}
    assert set(optionals.keys()) == {"test"}
    assert optionals["test"] == {"dask-cuda ==24.4.*,>=0.0.0a0"}
