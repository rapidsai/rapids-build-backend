# Copyright (c) 2024, NVIDIA CORPORATION.

import os.path
import tempfile
from unittest.mock import Mock, patch

import pytest

from rapids_build_backend.impls import _edit_git_commit


@pytest.mark.parametrize(
    ["initial_contents"],
    [
        ("def456\n",),
        ("",),
        (None,),
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
