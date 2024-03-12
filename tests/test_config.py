# Copyright (c) 2024, NVIDIA CORPORATION.

import os

import pytest
from conftest import generate_from_template

from rapids_builder.config import Config


def setup_config_project(tmp_path, flag, config_value):
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)

    generate_from_template(
        package_dir,
        "pyproject.toml",
        {
            "flags": {flag: config_value} if config_value else {},
        },
    )
    return package_dir


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
def test_config(tmp_path, flag, config_value, expected):
    config = Config(setup_config_project(tmp_path, flag, config_value))
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
def test_config_env_var(tmp_path, flag, config_value, expected):
    config = Config(setup_config_project(tmp_path, flag, None))
    env_var = f"RAPIDS_{flag.upper().replace('-', '_')}"
    python_var = flag.replace("-", "_")
    try:
        os.environ[env_var] = config_value
        assert getattr(config, python_var) == expected

        # Ensure that alternative spellings are not allowed.
        os.environ[env_var] = config_value.capitalize()
        with pytest.raises(ValueError):
            getattr(config, python_var)
    finally:
        del os.environ[env_var]


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
def test_config_config_settings(tmp_path, flag, config_value, expected):
    config = Config(
        setup_config_project(tmp_path, flag, None),
        {flag: config_value},
    )
    assert getattr(config, flag.replace("-", "_")) == expected
