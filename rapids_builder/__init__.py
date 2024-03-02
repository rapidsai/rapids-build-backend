# Copyright (c) 2024, NVIDIA CORPORATION.

from importlib import import_module

# All hooks defined by the PEP 517 standard.
pep517_hooks = [
    "build_wheel",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
    "build_editable",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_editable",
    "build_sdist",
    "get_requires_for_build_sdist",
]

# The backend may or may not support all of them.
__all__ = []
for hook in pep517_hooks:
    try:
        globals()[hook] = getattr(import_module("scikit_build_core"), hook)
        __all__.append(hook)
    except AttributeError:
        pass
