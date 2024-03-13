#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# TODO: In a conda environment we should probably be testing without
# isolation for any tests to be meaningfully conda-specific. If we
# think that's worthwhile it will require adding an environment
# variable to conftest.py that controls whether the default `env`
# used for installation is a virtual environment or simply the
# current Python environment.
python -m pytest -v tests/
