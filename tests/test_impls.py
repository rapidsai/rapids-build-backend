# Copyright (c) 2024, NVIDIA CORPORATION.

import os.path
from contextlib import contextmanager
from textwrap import dedent
from unittest.mock import Mock, patch

import pytest

from rapids_build_backend.impls import (
    _edit_pyproject,
    _get_cuda_suffix,
    _write_git_commits,
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
    ("project_name", "directories", "commit_files_config", "expected_commit_files"),
    [
        ("test-project", ["test_project"], None, ["test_project/GIT_COMMIT"]),
        (
            "test-project",
            ["_test_project"],
            ["_test_project/GIT_COMMIT"],
            ["_test_project/GIT_COMMIT"],
        ),
        (
            "test-project",
            ["_test_project_1", "_test_project_2"],
            ["_test_project_1/GIT_COMMIT", "_test_project_2/GIT_COMMIT"],
            ["_test_project_1/GIT_COMMIT", "_test_project_2/GIT_COMMIT"],
        ),
        (
            "test-project",
            [],
            [],
            [],
        ),
    ],
)
@patch("rapids_build_backend.impls._get_git_commit", Mock(return_value="abc123"))
def test_write_git_commits(
    tmp_path, project_name, directories, commit_files_config, expected_commit_files
):
    with set_cwd(tmp_path):
        for directory in directories:
            os.mkdir(directory)

        config = Mock(
            commit_files=commit_files_config,
        )
        with _write_git_commits(config, project_name):
            for expected_commit_file in expected_commit_files:
                with open(expected_commit_file) as f:
                    assert f.read() == "abc123\n"
            if not directories:
                assert list(os.walk(".")) == [(".", [], [])]

        for directory in directories:
            os.rmdir(directory)
        assert list(os.walk(".")) == [(".", [], [])]


@pytest.mark.parametrize(
    [
        "pyproject_dir",
        "dependencies_file",
        "write_dependencies_file",
        "disable_cuda",
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
            False,
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
            False,
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
            False,
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
            False,
            ("11", "5"),
            "-cu11",
            None,
            "",
            None,
        ),
        (
            ".",
            "dependencies.yaml",
            True,
            True,
            None,  # Ensure _get_cuda_version() isn't called and unpacked
            "",
            "cuda-python",
            "",
            "some-x86-package",
        ),
    ],
)
def test_edit_pyproject(
    tmp_path,
    pyproject_dir,
    dependencies_file,
    write_dependencies_file,
    disable_cuda,
    cuda_version,
    cuda_suffix,
    cuda_python_requirement,
    matrix,
    arch_requirement,
):
    original_contents = dedent(
        """\
        [project]
        name = "test-project"
        dependencies = []

        [build-system]
        requires = []
        """
    )
    full_pyproject_dir = os.path.join(tmp_path, pyproject_dir)
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
                                  - matrix:
                                    packages:
                                      - cuda-python
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
            disable_cuda=disable_cuda,
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
                            ] # This list was generated by """
                            """`rapids-dependency-file-generator`. To make """
                            f"""changes, edit {dependencies_file} and run """
                            """`rapids-dependency-file-generator`.

                            [build-system]
                            requires = [
                                "scikit-build-core",
                            ] # This list was generated by """
                            """`rapids-dependency-file-generator`. To make """
                            f"""changes, edit {dependencies_file} and run """
                            """`rapids-dependency-file-generator`.
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
