# Copyright (c) 2024, NVIDIA CORPORATION.

import os.path
import tempfile
from contextlib import contextmanager
from textwrap import dedent
from unittest.mock import Mock, patch

import pytest

from rapids_build_backend.impls import (
    _edit_git_commit,
    _edit_pyproject,
    _get_cuda_suffix,
)


@contextmanager
def set_cwd(cwd):
    old_cwd = os.getcwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@pytest.mark.parametrize(
    "initial_contents",
    [
        "def456\n",
        "",
        None,
    ],
)
@patch("rapids_build_backend.impls._get_git_commit", Mock(return_value="abc123"))
def test_edit_git_commit(initial_contents):
    def check_initial_contents(filename):
        if initial_contents is not None:
            with open(filename) as f:
                assert f.read() == initial_contents
        else:
            assert not os.path.exists(filename)

    with tempfile.TemporaryDirectory() as d:
        commit_file = os.path.join(d, "commit-file")
        bkp_commit_file = os.path.join(d, ".commit-file.rapids-build-backend.bak")
        if initial_contents is not None:
            with open(commit_file, "w") as f:
                f.write(initial_contents)

        config = Mock(
            commit_file=commit_file,
        )
        with _edit_git_commit(config):
            with open(commit_file) as f:
                assert f.read() == "abc123\n"
            check_initial_contents(bkp_commit_file)

        assert not os.path.exists(bkp_commit_file)
        check_initial_contents(commit_file)


@pytest.mark.parametrize(
    [
        "pyproject_dir",
        "dependencies_file",
        "write_dependencies_file",
        "cuda_version",
        "cuda_suffix",
        "cuda_python_requirement",
        "matrix",
        "arch_requirement",
    ],
    [
        (
            ".",
            "dependencies.yaml",
            True,
            ("11", "5"),
            "-cu11",
            "cuda-python>=11.5,<11.6.dev0",
            "",
            "some-x86-package",
        ),
        (
            ".",
            "dependencies.yaml",
            True,
            ("11", "5"),
            "-cu11",
            "cuda-python>=11.5,<11.6.dev0",
            "arch=aarch64",
            "some-arm-package",
        ),
        (
            "python",
            "../dependencies.yaml",
            True,
            ("12", "1"),
            "-cu12",
            "cuda-python>=12.1,<12.2.dev0",
            "",
            "some-x86-package",
        ),
        (
            ".",
            "dependencies.yaml",
            False,
            ("11", "5"),
            "-cu11",
            None,
            "",
            None,
        ),
    ],
)
def test_edit_pyproject(
    pyproject_dir,
    dependencies_file,
    write_dependencies_file,
    cuda_version,
    cuda_suffix,
    cuda_python_requirement,
    matrix,
    arch_requirement,
):
    with tempfile.TemporaryDirectory() as d:
        original_contents = dedent(
            """\
            [project]
            name = "test-project"
            dependencies = []

            [build-system]
            requires = []
            """
        )
        full_pyproject_dir = os.path.join(d, pyproject_dir)
        if not os.path.exists(full_pyproject_dir):
            os.mkdir(full_pyproject_dir)

        with set_cwd(full_pyproject_dir):
            with open("pyproject.toml", "w") as f:
                f.write(original_contents)

            if write_dependencies_file:
                with open(dependencies_file, "w") as f:
                    f.write(
                        dedent(
                            f"""\
                            files:
                              project:
                                output: pyproject
                                includes:
                                  - project
                                  - arch
                                pyproject_dir: {pyproject_dir}
                                matrix:
                                  cuda: ["11.5"]
                                  arch: ["x86_64"]
                                extras:
                                  table: project
                              build_system:
                                output: pyproject
                                includes:
                                  - build_system
                                pyproject_dir: {pyproject_dir}
                                extras:
                                  table: build-system
                              other_project:
                                output: pyproject
                                includes:
                                  - bad
                                pyproject_dir: python_bad
                                extras:
                                  table: project
                              conda:
                                output: conda
                                includes:
                                  - bad
                            dependencies:
                              project:
                                common:
                                  - output_types: [pyproject]
                                    packages:
                                      - tomli
                                specific:
                                  - output_types: [pyproject]
                                    matrices:
                                      - matrix:
                                          cuda: "11.5"
                                        packages:
                                          - cuda-python>=11.5,<11.6.dev0
                                      - matrix:
                                          cuda: "12.1"
                                        packages:
                                          - cuda-python>=12.1,<12.2.dev0
                              build_system:
                                common:
                                  - output_types: [pyproject]
                                    packages:
                                      - scikit-build-core
                              arch:
                                specific:
                                  - output_types: [pyproject]
                                    matrices:
                                      - matrix:
                                          arch: x86_64
                                        packages:
                                          - some-x86-package
                                      - matrix:
                                          arch: aarch64
                                        packages:
                                          - some-arm-package
                              bad:
                                common:
                                  - output_types: [pyproject, conda]
                                    packages:
                                      - bad-package
                            """
                        )
                    )
            config = Mock(
                require_cuda=False,
                dependencies_file=dependencies_file,
                matrix_entry=matrix,
            )

            with patch(
                "rapids_build_backend.impls._get_cuda_version",
                Mock(return_value=cuda_version),
            ), patch(
                "rapids_build_backend.impls._get_cuda_suffix",
                _get_cuda_suffix.__wrapped__,
            ):
                with _edit_pyproject(config):
                    with open("pyproject.toml") as f:
                        if write_dependencies_file:
                            assert f.read() == dedent(
                                f"""\
                                [project]
                                name = "test-project{cuda_suffix}"
                                dependencies = [
                                    "{cuda_python_requirement}",
                                    "{arch_requirement}",
                                    "tomli",
                                ]

                                [build-system]
                                requires = [
                                    "scikit-build-core",
                                ]
                                """
                            )
                        else:
                            assert f.read() == dedent(
                                f"""\
                                [project]
                                name = "test-project{cuda_suffix}"
                                dependencies = []

                                [build-system]
                                requires = []
                                """
                            )
                    with open(".pyproject.toml.rapids-build-backend.bak") as f:
                        assert f.read() == original_contents

            with open("pyproject.toml") as f:
                assert f.read() == original_contents
