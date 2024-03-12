# Copyright (c) 2024, NVIDIA CORPORATION.

import os

from .utils import _get_pyproject


class Config:
    """Manage the build configuration for the project."""

    # Mapping from config option to default value (None indicates that option is
    # required) and whether it may be overridden by an environment variable or a config
    # setting.
    config_options = {
        "allow-no-cuda": (False, True),
        "build-backend": (None, False),
        "commit-file": ("", False),
        "disable-cuda-suffix": (False, True),
        "only-release-deps": (False, True),
        "requires": ([], False),
    }

    def __init__(self, dirname="."):
        pyproject_data = _get_pyproject(dirname)
        try:
            self.config = pyproject_data["tool"]["rapids_builder"]
        except KeyError as e:
            raise RuntimeError("No rapids_builder table in pyproject.toml") from e

    def __getattr__(self, name):
        config_name = name.replace("_", "-")
        if config_name in Config.config_options:
            default_value, allows_override = Config.config_options[config_name]

            # Highest priority is environment variable.
            if allows_override:
                if (env_var_name := f"RAPIDS_{name.upper()}") in os.environ:
                    return os.environ[env_var_name]

                # TODO: Support config_settings

            # Default is pyproject.toml.
            try:
                return self.config[config_name]
            except KeyError:
                # Return the default value if one is defined.
                if default_value is not None:
                    return default_value

                raise AttributeError(f"Config is missing required attribute {name}")
        else:
            raise AttributeError(f"Attempted to access unknown option {name}")
