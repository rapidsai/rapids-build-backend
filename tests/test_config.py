# Copyright (c) 2024, NVIDIA CORPORATION.

import os

import pytest
from jinja2 import Environment, FileSystemLoader

from rapids_builder.config import Config


@pytest.fixture(scope="module")
def jinja_environment():
    template_dir = os.path.join(
        os.path.dirname(__file__),
        "config_packages",
        "templates/",
    )
    return Environment(loader=FileSystemLoader(template_dir))


@pytest.mark.parametrize(
    "flag, config_value, expected",
    [
        ("commit-file", '"pkg/_version.py"', "pkg/_version.py"),
        ("commit-file", None, ""),
        ("disable-cuda-suffix", "true", True),
        ("disable-cuda-suffix", "false", False),
        ("disable-cuda-suffix", None, False),
        ("only-release-deps", "true", True),
        ("only-release-deps", "false", False),
        ("only-release-deps", None, False),
        ("require-cuda", "true", True),
        ("require-cuda", "false", False),
        ("require-cuda", None, True),
    ],
)
def test_config(tmp_path, jinja_environment, flag, config_value, expected):
    template = jinja_environment.get_template("pyproject.toml")
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)

    flags = (
        {
            flag: config_value,
        }
        if config_value
        else {}
    )
    requires = []

    content = template.render(requires=requires, flags=flags)
    pyproject_file = os.path.join(package_dir, "pyproject.toml")
    with open(pyproject_file, mode="w", encoding="utf-8") as f:
        f.write(content)

    config = Config(package_dir)
    assert getattr(config, flag.replace("-", "_")) == expected


@pytest.mark.parametrize(
    "flag, config_value, expected",
    [
        ("disable-cuda-suffix", "true", True),
        ("disable-cuda-suffix", "false", False),
        ("only-release-deps", "true", True),
        ("only-release-deps", "false", False),
        ("require-cuda", "true", True),
        ("require-cuda", "false", False),
    ],
)
@pytest.mark.parametrize("capitalized", [True, False])
def test_config_env_var(
    tmp_path, jinja_environment, flag, config_value, expected, capitalized
):
    template = jinja_environment.get_template("pyproject.toml")
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)

    # Create a pyproject.toml file with the given flag set to the opposite value that
    # will be provided with the environment variable.
    flags = {
        flag: "true" if config_value == "false" else "false",
    }
    requires = []

    content = template.render(requires=requires, flags=flags)
    pyproject_file = os.path.join(package_dir, "pyproject.toml")
    with open(pyproject_file, mode="w", encoding="utf-8") as f:
        f.write(content)

    env_var = f"RAPIDS_{flag.upper().replace('-', '_')}"
    env_value = config_value.capitalize() if capitalized else config_value
    python_var = flag.replace("-", "_")
    try:
        os.environ[env_var] = env_value
        config = Config(package_dir)
        if capitalized:
            with pytest.raises(ValueError):
                getattr(config, python_var)
        else:
            assert getattr(config, python_var) == expected
    finally:
        del os.environ[env_var]
