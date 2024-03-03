# rapids PEP517 build backend

`rapids_builder` is an adapter around PEP517 builders that provides support for key RAPIDS requirements.
It currently support `scikit-build-core` and `setuptools` as the wrapped builder.
The package's primary purpose is to automate the CUDA versioning strategy for wheels.
It modifies the package name to include a suffix of the form `-cu${CUDA_VERSION}`, the standard in RAPIDS.
It also modifies build requirements to ensure that isolated builds have the necessary requirements, and it modifies the install requirements as well so that they are propagated to the built wheel.
