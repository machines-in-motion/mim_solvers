# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Releases are available on the [github repository](https://github.com/machines-in-motion/mim_solvers/releases).

## [Unreleased]

### Added

- Nix flake.nix
- Nix CI on ubuntu and MacOs to run the unit-tests
- Nix flake update CI to update the version of the dependencies.
- github dependabot to update the CI yaml files
- Add support for multithread with OpenMP
- added SQP type callbacks to fddp and ddp and removed BOOST_BIND_NO_PLACEHOLDERS
- CSQP: add parallelization and update the merit function
- Add maximum resolution time as stopping criterion. Allow the solver to stop after a certain duration has elapsed.

### Fixed

- All *build folder are now ignored by git.

### Changed

- Upgrade dependencies: crocoddyl 3.0.0
- Uppdate the pre-commit checks to be much more extensive:
  - Python lint + format with ruff
  - toml (packaging) formatting
  - clang-format for C++
  - standard checks from precommit:
    - id: check-added-large-files
    - id: check-ast
    - id: check-executables-have-shebangs
    - id: check-json
    - id: check-merge-conflict
    - id: check-symlinks
    - id: check-toml
    - id: check-yaml
    - id: debug-statements
    - id: destroyed-symlinks
    - id: detect-private-key
    - id: end-of-file-fixer
    - id: fix-byte-order-marker
    - id: mixed-line-ending
    - id: trailing-whitespace
- Only depend on example-robot-data when benchmarks and/or unit-tests are compiled.
- Use pre-commit to format the different files.
- Allow the test to not be ran is the standard CMake var `BUILD_TESTING` is off.
- Changing boost::shared_ptr to std::shared_ptr
- Clean up code of CSQP

## [0.0.5] - 2024-08-16

### Added

- Added callbacks (this changes the API, see the examples)
- Added vectorization and malloc check options in CMakeLists
- Added Github CI

### Changed 

- Code optimization
- Now checking stopping criteria for QP every 25 iterations in SolverCSQP (as a result, it is now more efficient to use SolverSQP for unconstrained problems)

### Fixed

- Fixed bindings and renamed merit function parameters (`mu_dynamic` and `mu_constraint`)


## [0.0.4] - 2024-04-03

### Added

- Added callbacks in QP solvers

### Changed

- Replaces `warm_start_y` by `reset_y`.
- By default, `y` and `rho` are not reset.

### Fixed

- Fixed rho initialization in CSQP

## [0.0.3] - 2024-01-27

### Added

- Added unittest infrastructure in python and C++
- Added C++ benchmarks of the CSQP and SQP solvers

### Fixed

- Fixed bug when relative residual tolerance is set to 0 in stagewise QP solver

### Changed

- Optimized code speed for CSQP and SQP solvers
- Use systematically KKT residual stopping criteria in all solvers
- Deprecation warning of DDP and FDDP solvers
- Inform user not to use equality constraint API of Crocoddyl
- Updated python implementations of the solvers
- Updated regularization variables consistently with Crocoddyl 2.0.2
- Refactored python code (now separated from python bindings)

### Deprecation

## [0.0.2] - 2023-12-10

### Added

- Solvers C++ unittests (unconstrained) by @skleff1994 in #9

### Fixed

- Fix packaging issues with ProxSuite by @jcarpent in #12

### Changed

- CMake updates by @nim65s in #13

### New contributors

- @jcarpent made their first contribution in #12
- @nim65s made their first contribution in #13

## [0.0.1] - 2023-11-30

Implementation of numerical solvers used in the Machines in Motion Laboratory.

## Git changelogs

Full Changelog: [v0.0.4...v0.0.5](https://github.com/machines-in-motion/mim_solvers/compare/v0.0.4...v0.0.5)
Full Changelog: [v0.0.3...v0.0.4](https://github.com/machines-in-motion/mim_solvers/compare/v0.0.3...v0.0.4)
Full Changelog: [v0.0.2...v0.0.3](https://github.com/machines-in-motion/mim_solvers/compare/v0.0.2...v0.0.3)
Full Changelog: [v0.0.1...v0.0.2](https://github.com/machines-in-motion/mim_solvers/compare/v0.0.1...v0.0.2)
