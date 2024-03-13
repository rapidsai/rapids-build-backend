# Copyright (c) 2024, NVIDIA CORPORATION.

import os.path
import tempfile
from unittest.mock import Mock, patch

import pytest

from rapids_build_backend.impls import _edit_git_commit


@pytest.mark.parametrize(
    ["commit_file_type", "initial_contents", "expected_contents"],
    [
        (
            "python",
            '# Begin Python file\n__git_commit__ = ""\n# End Python file\n',
            '# Begin Python file\n__git_commit__ = "abc123"\n# End Python file\n',
        ),
        ("python", None, FileNotFoundError),
        ("raw", "def456\n", "abc123\n"),
        ("raw", None, "abc123\n"),
    ],
)
@patch("rapids_build_backend.impls._get_git_commit", Mock(return_value="abc123"))
def test_edit_git_commit(commit_file_type, initial_contents, expected_contents):
    with tempfile.TemporaryDirectory() as d:
        commit_file = os.path.join(d, "commit-file")
        if initial_contents is not None:
            with open(commit_file, "w") as f:
                f.write(initial_contents)

        config = Mock(
            commit_file=commit_file,
            commit_file_type=commit_file_type,
        )
        if isinstance(expected_contents, type) and issubclass(
            expected_contents, Exception
        ):
            with pytest.raises(expected_contents):
                with _edit_git_commit(config):
                    pass
        else:
            with _edit_git_commit(config):
                with open(commit_file) as f:
                    assert f.read() == expected_contents
                assert os.path.exists(
                    os.path.join(d, ".commit-file.rapids-build-backend.bak")
                ) == (initial_contents is not None)

        if initial_contents is not None:
            with open(commit_file) as f:
                assert f.read() == initial_contents
        else:
            assert not os.path.exists(commit_file)
