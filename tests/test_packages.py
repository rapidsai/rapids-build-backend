# Copyright (c) 2024, NVIDIA CORPORATION.

import glob
import os
import re
import subprocess
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
            elif re.search(
                r"Processing .*/rapids_test_dummy-0\.0\.1-py3-none-any\.whl$", line
            ):
                build_requires.add("rapids-test-dummy")

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


# rapids-build-backend should support projects using setuptools whose setup.py
# file has 'import' statements depending on some project(s) that need to be extracted
# from dependencies.yaml at build time
def test_setuptools_with_imports_in_setup_py_works(
    tmp_path,
    isolated_env,
):
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)

    template_args = {
        "name": "setuptools-with-imports-in-setup-py",
        "build_backend": "setuptools.build_meta",
        "build_backend_package": "setuptools",
        "flags": {
            "commit-files": "[]",
            "dependencies-file": '"dependencies-rbb-only.yaml"',
        },
        "setup_py_lines": [
            "import rapids_test_dummy",
            "",
            "print(rapids_test_dummy.__version__)",
            "",
            "setup()",
        ],
    }
    generate_from_template(package_dir, "dependencies-rbb-only.yaml", template_args)
    generate_from_template(package_dir, "pyproject.toml", template_args)
    generate_from_template(package_dir, "setup.py", template_args)

    with patch_nvcc_if_needed(nvcc_version="85"):
        name, build_requires, requirements, extras = _generate_wheel(
            env=isolated_env, package_dir=package_dir
        )

    assert name == "setuptools-with-imports-in-setup-py-cu85"
    assert {"rapids-test-dummy"}.issubset(build_requires)
    assert requirements == set()


def test_setuptools_with_imports_in_setup_py_fails_on_missing_imports(
    tmp_path, isolated_env, capfd
):
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)

    template_args = {
        "name": "setuptools-with-imports-in-setup-py",
        "build_backend": "setuptools.build_meta",
        "build_backend_package": "setuptools",
        "flags": {
            "commit-files": "[]",
            "dependencies-file": '"dependencies-rbb-only.yaml"',
        },
        "setup_py_lines": [
            "import rapids_test_dummy",
            "",
            "print(rapids_test_dummy.__version__)",
            "",
            "setup()",
        ],
    }
    generate_from_template(package_dir, "dependencies-rbb-only.yaml", template_args)
    generate_from_template(package_dir, "pyproject.toml", template_args)
    generate_from_template(package_dir, "setup.py", template_args)

    # only the CUDA '85.*' in this example provides required build dependency
    # 'rapids-test-dummy', so it won't be found if using some other matrix.
    #
    # This test confirms that rapids-build-backend fails loudly in that case, instead of
    # silently ignoring it.
    #
    # It'd also catch the case where other tests accidentally pass because
    # 'rapids-test-dummy' already existed in the environment where tests run.
    with patch_nvcc_if_needed(nvcc_version="25"):
        with pytest.raises(subprocess.CalledProcessError, match=".*pip.*"):
            _generate_wheel(env=isolated_env, package_dir=package_dir)

    captured_output = capfd.readouterr()
    assert (
        "ModuleNotFoundError: No module named 'rapids_test_dummy'"
        in captured_output.out
    )


def test_setuptools_with_setup_requires_fails_with_informative_error(
    tmp_path, isolated_env, capfd
):
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)
    template_args = {
        "name": "setuptools-with-imports-in-setup-py",
        "build_backend": "setuptools.build_meta",
        "build_backend_package": "setuptools",
        "flags": {
            "commit-files": "[]",
            "dependencies-file": '"dependencies-rbb-only.yaml"',
        },
        "setup_py_lines": [
            "import rapids_test_dummy",
            "",
            "print(rapids_test_dummy.__version__)",
            "",
            "setup(",
            "    setup_requires=['rapids-test-dummy'],",
            ")",
        ],
    }
    generate_from_template(package_dir, "dependencies-rbb-only.yaml", template_args)
    generate_from_template(package_dir, "pyproject.toml", template_args)
    generate_from_template(package_dir, "setup.py", template_args)

    with patch_nvcc_if_needed(nvcc_version="85"):
        with pytest.raises(subprocess.CalledProcessError, match=".*pip.*"):
            _generate_wheel(env=isolated_env, package_dir=package_dir)

    captured_output = capfd.readouterr()
    assert (
        "ValueError: Detected use of 'setup_requires' in a setup.py file"
        in captured_output.out
    )


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
