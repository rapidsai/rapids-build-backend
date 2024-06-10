# Copyright (c) 2024, NVIDIA CORPORATION.

# Note that more-itertools is not listed anywhere in pyproject.toml.
# This import can only succeed if rapids-build-backend successfully identified it as
# a build-time dependency via dependencies.yaml.
import more_itertools
from setuptools import setup

print(more_itertools.__version__)

setup()
