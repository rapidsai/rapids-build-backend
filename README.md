# RAPIDS PEP517 build backend

`rapids_build_backend` is an adapter around PEP517 builders that provides support for key RAPIDS requirements.
It currently support `scikit-build-core` and `setuptools` as the wrapped builder.
The package's primary purpose is to automate the various bits of preprocessing that are typically done to RAPIDS package metadata prior to publishing packages.
This includes the following notable changes:
- Modifying the package name to include CUDA suffixes.
- Updating the git commit embedded in the importable package.
- Modifying the package's (build, runtime, etc) dependencies to include CUDA suffixes.
- Filtering out dependencies based on the CUDA version at build time.
- Updating dependency specifiers to include an alpha specifier to allow pulling nightly dependencies in nightly builds.

Since some of these modifications are only desirable in certain scenarios (wheel vs conda builds vs editable installs), all of these functions are customizable via the project's configuration in pyproject.toml.
In cases where more dynamic customization is sensible, suitable environment variables and `config_settings` are supported during builds of distributions.

## Supported configuration

Any option without a default is required.

| Option                | Definition                                                               | Type      | Default      | Supports dynamic modification |
|-----------------------|--------------------------------------------------------------------------|-----------|--------------|-------------------------------|
| `build-backend`       | The wrapped build backend (e.g. `setuptools.build_meta`)                 | string    |              | N                             |
| `commit-file`         | The file in which to write the commit                                    | string    | "" (No file) | N                             |
| `commit-file-type`    | The type of file in which to write the commit (one of `python` or `raw`) | string    | "python"     | N                             |
| `disable-cuda-suffix` | If true, don't try to write CUDA suffixes                                | bool      | false        | Y                             |
| `only-release-deps`   | If true, do not append alpha specifiers to dependencies                  | bool      | false        | Y                             |
| `require-cuda`        | If false, builds will succeed even if nvcc is not available              | bool      | true         | Y                             |
| `requires`            | List of build requirements (in addition to `build-system.requires`)      | list[str] | []           | N                             |


## Outstanding questions

- How should we split up build requirements between `build-system` and `tool.rapids_builder`? In theory any dependency that doesn't need suffixing could also go into `build-system.requires`. I think it's easier to teach that all dependencies other than `rapids_builder` itself should to into `tool.rapids_builder`, but I don't know how others feel.

## Future improvements

- When https://github.com/rapidsai/dependency-file-generator/pull/48 is completed we can make dfg a dependency of this builder and use it to rewrite the dependency list instead of the manual logic here.

## Rejected ideas

- We could also include the rewrite of VERSION that we use for RAPIDS builds, but this is really more specific to our release process than the general process of building our wheels. I don't think someone building a wheel locally should see the same version as what we produce via CI. If we really wanted we could pull dunamai as a dependency and write a different version here, though.
