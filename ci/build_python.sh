#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-conda-retry build conda/recipes/rapids-build-backend
