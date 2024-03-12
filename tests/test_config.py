# Copyright (c) 2024, NVIDIA CORPORATION.

import os

import pytest
from jinja2 import Environment, FileSystemLoader

from rapids_builder.config import Config
from rapids_builder.utils import _get_pyproject


@pytest.fixture(scope="module")
def jinja_environment():
    template_dir = os.path.join(
        os.path.dirname(__file__),
        "config_packages",
        "templates/",
    )
    return Environment(loader=FileSystemLoader(template_dir))


def test_config(tmp_path, jinja_environment):
    template = jinja_environment.get_template("pyproject.toml")
    package_dir = tmp_path / "pkg"
    os.makedirs(package_dir)

    flags = {
        "allow-no-cuda": "false",
    }

    content = template.render(flags=flags)
    pyproject_file = os.path.join(package_dir, "pyproject.toml")
    with open(pyproject_file, mode="w", encoding="utf-8") as f:
        f.write(content)

    pyproject_data = _get_pyproject(package_dir)
    config = Config(pyproject_data)
    assert not config.allow_no_cuda
    assert config.commit_file == ""
    assert not config.disable_cuda_suffix
    assert not config.only_release_deps
    assert config.requires == []
