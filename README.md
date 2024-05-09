# RAPIDS PEP517 build backend

`rapids-build-backend` is an adapter around PEP517 builders that provides support for key RAPIDS requirements.
It currently support `scikit-build-core` and `setuptools` as the wrapped builder.
The package's primary purpose is to automate the various bits of preprocessing that are typically done to RAPIDS package metadata prior to publishing packages.
This includes the following notable changes:
- Running [`rapids-dependency-file-generator`](https://github.com/rapidsai/dependency-file-generator) to get the dependencies for the CUDA version and architecture.
- Modifying the package name to include CUDA suffixes.
- Updating the git commit embedded in the importable package.

Since some of these modifications are only desirable in certain scenarios (wheel vs conda builds vs editable installs), all of these functions are customizable via the project's configuration in pyproject.toml.
In cases where more dynamic customization is sensible, suitable environment variables and `config_settings` are supported during builds of distributions.

## Supported configuration

Any option without a default is required.

| Option                | Definition                                                          | Type           | Default             | Supports dynamic modification |
|-----------------------|---------------------------------------------------------------------|----------------|---------------------|-------------------------------|
| `build-backend`       | The wrapped build backend (e.g. `setuptools.build_meta`)            | string         |                     | N                             |
| `commit-file`         | The file in which to write the git commit hash                      | string         | "" (No file)        | N                             |
| `dependencies-file`   | The path to the `dependencies.yaml` file to use                     | string         | "dependencies.yaml" | Y                             |
| `disable-cuda-suffix` | If true, don't try to write CUDA suffixes                           | bool           | false               | Y                             |
| `matrix-entry`        | A `;`-separated list of `=`-delimited key/value pairs               | string         | ""                  | Y                             |
| `require-cuda`        | If false, builds will succeed even if nvcc is not available         | bool           | true                | Y                             |
| `requires`            | List of build requirements (in addition to `build-system.requires`) | list[str]      | []                  | N                             |


## Outstanding questions

- How should we split up build requirements between `build-system` and `tool.rapids-build-backend`? In theory any dependency that doesn't need suffixing could also go into `build-system.requires`. I think it's easier to teach that all dependencies other than `rapids-build-backend` itself should to into `tool.rapids-build-backend`, but I don't know how others feel.

## Rejected ideas

- We could also include the rewrite of VERSION that we use for RAPIDS builds, but this is really more specific to our release process than the general process of building our wheels. I don't think someone building a wheel locally should see the same version as what we produce via CI. If we really wanted we could pull dunamai as a dependency and write a different version here, though.
