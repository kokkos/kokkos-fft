<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# CI Instructions for Kokkos-FFT

These instructions apply when creating, modifying, or reviewing CI/CD workflow files (`.yaml`) in the `.github/workflows/` directory.

## License Headers

Every workflow file must start with the SPDX license header:

```yaml
# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — omitting this header causes build failure.

## Workflow Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `build_test.yaml` | PR to `main` | Main CI: formatting, linting, building, testing |
| `build_nightly.yaml` | Cron (weekdays 1am) + manual | Extended nightly tests with aggressive flags |
| `build_test_distributed.yaml` | PR to `main` | MPI distributed FFT tests |
| `python-test.yaml` | PR to `main` | Python linting and testing |
| `spack-test.yaml` | PR to `main` | Spack installation verification |
| `reuse.yml` | PR to `main` | SPDX license compliance check |
| `website-checks.yaml` | Weekly + manual | Documentation website link checking |
| `heartbeat.yaml` | Periodic | Repository health monitoring |
| `create_base.yaml` / `cleanup_base.yaml` | Manual / Scheduled | Docker image management |

## Reusable Workflows

- Reusable workflows use the `__` prefix convention: `__clang-format-check.yaml`, `__cmake-format-check.yaml`.
- They are triggered via `workflow_call` and called from the main workflow.
- Keep reusable workflows focused on a single concern.

## Main CI Pipeline (`build_test.yaml`)

The main workflow includes these stages:

1. **Format checks**: clang-format (v17), cmake-format, typos spell check.
2. **REUSE compliance**: SPDX license header verification.
3. **Build matrix**: Multiple configurations run in parallel:
   - `clang-tidy`: Clang, C++17, Serial, Debug, warnings-as-errors.
   - `openmp`: GCC, C++17, OpenMP+Serial, Debug.
   - `threads`: GCC, C++20, Threads, Release.
   - `serial`: GCC, C++17, Serial, Release.
   - `cuda`: GCC+nvcc, C++20, CUDA+OpenMP, Release.
   - `hip`: hipcc, C++17, HIP (hipFFT), Release.
   - `rocm`: hipcc, C++20, HIP (rocFFT), Release.
   - `sycl`: icpx, C++17, SYCL, Release.
4. **Unit tests**: Run on CPU backends (Azure) and NVIDIA GPUs (self-hosted, requires maintainer approval).

## Docker Usage

- CI builds run inside Docker containers from `ghcr.io/kokkos/kokkos-fft/`.
- Docker images are defined in the `docker/` directory (gcc, clang, nvcc, rocm, intel variants).
- Enable BuildKit: `DOCKER_BUILDKIT: 1`.
- Docker images are pre-built and cached via `create_base.yaml`.

## Action Version Pinning

- **Always pin actions to full Git SHAs** for security, not tags.
- Example: `actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2`.
- Include a comment with the tag version for readability.
- When updating action versions, update both the SHA and the comment.

## Build Matrix Configuration

Use `include` in the matrix strategy for explicit configuration sets:

```yaml
strategy:
  matrix:
    include:
      - name: config-name
        compiler: g++
        backend: OPENMP
        cmake_flags: "-DKokkos_ENABLE_OPENMP=ON"
```

- Each matrix entry should have a descriptive `name`.
- Include compiler, backend, C++ standard, and build type.
- Use meaningful CMake flags for each configuration.

## Format Checking

### clang-format

- Tool: `DoozyX/clang-format-lint-action` (v0.20).
- Scope: `common/`, `fft/`, `examples/`, `install_test/`, `testing/`, `distributed/`.
- Extensions: `hpp`, `cpp`.
- clang-format version: 17.

### cmake-format

- Checks CMake file formatting against `.cmake-format.py` configuration.
- Scope: All `CMakeLists.txt` and `.cmake` files.

### Spell check

- Tool: `typos` (v1.45.0+).
- Configuration: `.typos.toml`.
- Scope: `cmake/`, `CMakeLists.txt`, `docs/`, `README.md`, `common/`, `fft/`, `examples/`, `install_test/`, `testing/`, `distributed/`.

## Python CI (`python-test.yaml`)

- Python versions tested: 3.10, 3.11, 3.12.
- Uses `hatch` for dependency management.
- Jobs: `pylint` (static analysis), `pytest` (unit tests), version consistency check.

## Adding New Workflows

1. Create the workflow file in `.github/workflows/`.
2. Add the SPDX license header.
3. Pin all action references to full Git SHAs with version comments.
4. Use matrix strategies for multi-configuration testing.
5. Set appropriate timeouts for jobs and steps.
6. For reusable workflows, use the `__` prefix naming convention.

## Adding New CI Checks

1. Add the check as a new job in the appropriate workflow.
2. Keep jobs focused — one concern per job.
3. Use `needs:` to express dependencies between jobs.
4. Set `fail-fast: false` in matrix strategies to run all configurations even if one fails.

## Nightly Builds

- Run on weekdays at 1am UTC via cron schedule.
- Restricted to the original `kokkos/kokkos-fft` repository (not forks).
- Use more aggressive compiler flags: `-Wall -Wextra -Werror`.
- Test additional backend and C++ standard combinations.

## Things to Avoid

- Do not use action tags without SHA pinning (security risk).
- Do not run builds inside the source directory (always use out-of-source builds).
- Do not add workflows that run on `push` to `main` without good reason — prefer PR triggers.
- Do not hardcode Docker image tags — reference images from the project's container registry.
- Do not skip format or REUSE checks for any PR.
- Do not add secrets or credentials directly in workflow files — use GitHub Secrets.
