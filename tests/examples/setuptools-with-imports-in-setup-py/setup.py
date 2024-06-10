# Copyright (c) 2024, NVIDIA CORPORATION.

# Note that matplotlib is not listed anywhere in pyproject.toml.
# This import can only succeed if rapids-build-backend successfully identified it as
# a build-time dependency via dependencies.yaml.
import matplotlib
from setuptools import setup

print(matplotlib.__version__)

setup()
