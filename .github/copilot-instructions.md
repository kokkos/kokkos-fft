<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Copilot Instructions for Kokkos-FFT

## Project Overview

Kokkos-FFT is a header-only C++ library providing FFT (Fast Fourier Transform) interfaces for the [Kokkos](https://github.com/kokkos/kokkos) performance portability ecosystem. It wraps vendor-specific FFT libraries (FFTW, cuFFT, hipFFT, rocFFT, oneMKL) behind a single, unified, Kokkos-native API. The library requires C++17 minimum, uses CMake 3.22+, and targets Kokkos 4.6+.

## Repository Structure

```
├── common/              # Shared utilities (types, traits, helpers, padding, layout)
│   ├── src/             # Header files (KokkosFFT_*.hpp)
│   └── unit_test/       # Google Test unit tests for common utilities
├── fft/                 # Core FFT implementation
│   ├── src/             # Plans, transforms, and backend-specific implementations
│   ├── unit_test/       # Google Test unit tests for FFT operations
│   └── perf_test/       # Google Benchmark performance tests
├── distributed/         # Distributed (MPI-based) FFT support
│   ├── src/             # MPI communication, block transpose, pack/unpack
│   └── unit_test/       # MPI-enabled Google Test unit tests
├── examples/            # 10 example programs (01_1DFFT through 10_HasegawaWakatani)
├── docs/                # Sphinx + Doxygen documentation
├── cmake/               # CMake modules (FindFFTW, FindcuFFTMp, KokkosFFT_tpls, etc.)
├── docker/              # Docker images for CI (gcc, clang, nvcc, rocm, intel)
├── tpls/                # Git submodules (kokkos, googletest, benchmark)
├── testing/             # Optional testing utility library (allclose, almost_equal helpers)
├── install_test/        # Post-installation verification tests
└── python_scripts/      # Python utility scripts
```

## Key Architecture Concepts

- **Header-only library**: `common` and `fft` are CMake INTERFACE targets. There are no `.cpp` implementation files in the library itself.
- **Namespace hierarchy**: Public API lives in `KokkosFFT::`. Internal implementation details live in `KokkosFFT::Impl::`. Distributed functionality is in `KokkosFFT::Distributed::`.
- **Backend abstraction**: Backend selection is controlled by preprocessor defines (`KOKKOSFFT_ENABLE_TPL_CUFFT`, etc.) set via CMake. Each backend has a set of files: `KokkosFFT_<Backend>_types.hpp`, `KokkosFFT_<Backend>_plans.hpp`, `KokkosFFT_<Backend>_transform.hpp`.
- **Plan-Execute pattern**: FFT plans capture configuration; `execute()` applies transforms. Plans can be reused across invocations.
- **Type safety via templates**: All public functions accept `ExecutionSpace` as the first argument and data via `Kokkos::View<>`. SFINAE (`enable_if_t`, type traits) is used extensively for compile-time validation.
- **Main header**: Users include `<KokkosFFT.hpp>` which pulls in plans, transforms, helpers, and dynamic plans.

## Building and Testing

### Prerequisites

The project uses CMake and relies on Kokkos being available either externally or via the internal submodule (`KokkosFFT_ENABLE_INTERNAL_KOKKOS=ON`). Note: `KokkosFFT_ENABLE_INTERNAL_KOKKOS=ON` is mainly intended for developers and CI — it is not optimal for end users, who should prefer linking against an externally installed Kokkos. For CPU builds, FFTW3 is needed. For GPU builds, the corresponding vendor FFT library is required.

### Build Commands for CI (CPU with OpenMP)

```bash
cmake -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ENABLE_OPENMP=ON \
  -DKokkosFFT_ENABLE_INTERNAL_KOKKOS=ON \
  -DKokkosFFT_ENABLE_TESTS=ON \
  -DKokkosFFT_ENABLE_EXAMPLES=ON

cmake --build build -j $(nproc)
```

### Build Commands for CI (CPU with Serial backend, minimal)

```bash
cmake -B build \
  -DCMAKE_CXX_COMPILER=g++ \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkosFFT_ENABLE_INTERNAL_KOKKOS=ON \
  -DKokkosFFT_ENABLE_TESTS=ON

cmake --build build -j $(nproc)
```

### Running Tests

```bash
cd build && ctest --output-on-failure
```

### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `KokkosFFT_ENABLE_INTERNAL_KOKKOS` | OFF | Build bundled Kokkos from `tpls/kokkos` |
| `KokkosFFT_ENABLE_TESTS` | OFF | Build unit tests |
| `KokkosFFT_ENABLE_EXAMPLES` | OFF | Build example programs |
| `KokkosFFT_ENABLE_BENCHMARK` | OFF | Build performance benchmarks |
| `KokkosFFT_ENABLE_FFTW` | auto | Enable FFTW backend (default ON for CPU) |
| `KokkosFFT_ENABLE_CUFFT` | auto | Enable cuFFT (NVIDIA GPU) |
| `KokkosFFT_ENABLE_HIPFFT` | auto | Enable hipFFT (AMD GPU) |
| `KokkosFFT_ENABLE_ROCFFT` | OFF | Enable rocFFT (AMD GPU, alternative to hipFFT) |
| `KokkosFFT_ENABLE_ONEMKL` | auto | Enable oneMKL (Intel GPU) |
| `KokkosFFT_ENABLE_DISTRIBUTED` | OFF | Enable distributed (MPI) FFT support |
| `KokkosFFT_ENABLE_TESTING_TOOLS` | OFF | Enable testing utility library |

### Test Executables

- `unit-tests-kokkos-fft-common` — Common utility tests
- `unit-tests-kokkos-fft-core` — Core FFT plan and transform tests
- `unit-tests-kokkos-dynfft` — Dynamic plan API tests
- `unit-tests-kokkos-fft-distributed` — Distributed FFT tests (requires MPI)

### Important Build Notes

- GPU backend builds (CUDA, HIP, SYCL) require the corresponding hardware or Docker containers. CI uses Docker images from `ghcr.io/kokkos/kokkos-fft/`.
- Only one device backend can be enabled at a time (cuFFT, hipFFT, rocFFT, or oneMKL are mutually exclusive).
- The CI runs builds inside Docker containers. For local CPU development, a standard compiler with FFTW3 installed is sufficient.

## Code Style and Formatting

### C++ Formatting (clang-format)

Based on Google style with these customizations (`.clang-format`):
- `SortIncludes: false` — Do NOT reorder includes
- `AlignConsecutiveAssignments: true`
- `AllowShortCaseLabelsOnASingleLine: true`
- `AllowShortIfStatementsOnASingleLine: true`
- `InsertNewlineAtEOF: true`

Format C++ files: `clang-format -i <file>`

### Static Analysis (clang-tidy)

Enabled checks (`.clang-tidy`):
- `modernize-type-traits`, `modernize-use-using`, `modernize-use-nullptr`
- `cppcoreguidelines-pro-type-cstyle-cast`
- `bugprone-reserved-identifier`

Header filter: `(common|fft).*\.hpp` (excludes `tpls/`)

### CMake Formatting (cmake-format)

Config in `.cmake-format.py`: line width 120, dangling parentheses enabled. Format CMake files: `cmake-format --in-place <file>`

### Spell Check

Uses `typos` with exceptions defined in `.typos.toml`. CI checks spelling in `cmake/`, `CMakeLists.txt`, `docs/`, `README.md`, `common/`, `fft/`, `examples/`, `install_test/`, `testing/`, `distributed/`.

## Coding Conventions

### File Naming

- Source headers: `KokkosFFT_<ComponentName>.hpp` (PascalCase component)
- Backend-specific: `KokkosFFT_<Backend>_<Component>.hpp` (e.g., `KokkosFFT_Cuda_types.hpp`)
- Test files: `Test_<ComponentName>.cpp` (e.g., `Test_Padding.cpp`)

### Include Guards

```cpp
#ifndef KOKKOSFFT_COMPONENTNAME_HPP
#define KOKKOSFFT_COMPONENTNAME_HPP
// ...
#endif
```

### License Headers

Every file must have an SPDX header. For C++ files:
```cpp
// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

For CMake files:
```cmake
# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — every file must have copyright and license information.

### Naming Conventions

| Entity | Convention | Examples |
|--------|-----------|---------|
| Types/Classes | PascalCase | `ViewType`, `ExecutionSpace`, `Plan` |
| Functions | snake_case | `get_modified_shape()`, `crop_or_pad()` |
| Variables | snake_case | `in_view`, `out_extents`, `fft_size` |
| Class members | `m_` prefix | `m_in_topology`, `m_plan` |
| Template params | PascalCase | `ExecutionSpace`, `InViewType`, `DIM` |
| Enums | PascalCase | `Normalization`, `Direction` |
| Enum values | snake_case | `Normalization::forward`, `Direction::backward` |
| Macros | UPPER_SNAKE_CASE | `KOKKOSFFT_THROW_IF`, `KOKKOSFFT_ENABLE_TPL_CUFFT` |
| Files | PascalCase_Component.hpp | `KokkosFFT_Padding.hpp` |

### Error Handling

- **Runtime errors**: Use `KOKKOSFFT_THROW_IF(condition, "message")` which throws `std::runtime_error` with file, line, function context. Note: the condition is true-to-throw (throws when the expression evaluates to true).
- **Compile-time checks**: Use `KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(expr, "name")` for view compatibility validation.
- **FFT call errors**: Use `KokkosFFT::Impl::check_fft_call()` for vendor FFT return code checking.

### Template Patterns

Public API functions follow this pattern:
```cpp
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void function_name(const ExecutionSpace& exec_space,
                   const InViewType& in, const OutViewType& out, ...);
```

### Documentation Style

Use Doxygen-style `/// \brief`, `\tparam`, `\param[in]`, `\param[out]`, `\return` comments for public API functions. Internal implementation functions may have shorter comments.

## CI/CD Pipeline

The main CI workflow (`.github/workflows/build_test.yaml`) runs on PRs to `main` and includes:

1. **Format checks**: clang-format, cmake-format, typos spell check
2. **REUSE compliance**: License header verification
3. **Build matrix** (all inside Docker containers):
   - `clang-tidy`: Clang, C++17, Serial, Debug, warnings-as-errors
   - `openmp`: GCC, C++17, OpenMP+Serial, Debug
   - `threads`: GCC, C++20, Threads, Release
   - `serial`: GCC, C++17, Serial, Release
   - `cuda`: GCC+nvcc, C++20, CUDA+OpenMP, Release
   - `hip`: hipcc, C++17, HIP (hipFFT), Release
   - `rocm`: hipcc, C++20, HIP (rocFFT), Release
   - `sycl`: icpx, C++17, SYCL, Release
4. **Unit tests**: Run on CPU backends (Azure) and NVIDIA GPUs (self-hosted, requires maintainer approval)

Other workflows: `build_nightly.yaml` (extended nightly tests), `build_test_distributed.yaml` (MPI tests), `spack-test.yaml` (Spack installation), `website-checks.yaml` (docs), `reuse.yml` (license compliance).

## Testing Conventions

### Test Structure

Tests use Google Test. Each test file follows this pattern:
```cpp
// SPDX license header

#include <gtest/gtest.h>
#include "KokkosFFT_ComponentUnderTest.hpp"
// other includes

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T> using View1D = Kokkos::View<T*, execution_space>;
// ... type aliases

// Test fixtures and cases
TEST(TestSuiteName, TestCaseName) { ... }
// or typed tests:
TYPED_TEST_SUITE(FixtureName, TypeList);
TYPED_TEST(FixtureName, TestCaseName) { ... }
}
```

### Test Main (Test_Main.cpp)

Each test directory has a `Test_Main.cpp` that initializes Kokkos:
```cpp
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
```

### Adding New Tests

1. Create `Test_<Name>.cpp` in the appropriate `unit_test/` directory
2. Add it to the test executable sources in `unit_test/CMakeLists.txt`
3. Link against `GTest::gtest` and the relevant library target (`common` or `KokkosFFT::fft`)
4. Use `gtest_discover_tests()` with `DISCOVERY_TIMEOUT 600` and `DISCOVERY_MODE PRE_TEST`

## Dependencies

| Dependency | Source | Required? |
|-----------|--------|-----------|
| Kokkos 4.6+ | `tpls/kokkos` submodule or external | Yes |
| Google Test | `tpls/googletest` submodule or external | For tests only |
| Google Benchmark | `tpls/benchmark` submodule | For benchmarks only |
| FFTW3 | System install | For CPU backend |
| cuFFT | CUDA Toolkit | For NVIDIA GPU backend |
| hipFFT/rocFFT | ROCm | For AMD GPU backend |
| oneMKL | Intel OneAPI | For Intel GPU backend |

## Common Pitfalls

- **Include ordering**: Do NOT sort includes — `SortIncludes: false` is intentional in clang-format.
- **Mutually exclusive backends**: Only one GPU FFT backend can be enabled per build.
- **Submodules**: Clone with `--recursive` or run `git submodule update --init --recursive` to get tpls.
- **REUSE compliance**: Every new file needs SPDX license headers or CI will fail.
- **View compatibility**: Input/output views must share the same base floating-point type, layout, and rank. This is enforced at compile time via `KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE`.
- **KOKKOSFFT_THROW_IF semantics**: The macro throws when the condition is **true** (true-to-throw), which is the opposite of a traditional assert.

## Security

- **No secrets in code**: Never commit credentials, API keys, tokens, or passwords into source code or configuration files.
- **No sensitive data exposure**: Never share sensitive repository data (code, credentials, internal configurations) with third-party systems.
- **No new vulnerabilities**: Validate that changes do not introduce security vulnerabilities (e.g., buffer overflows, unvalidated inputs, unsafe memory access).
- **Dependency vigilance**: Review new dependencies for known vulnerabilities before adding them. Do not add unnecessary dependencies.
- **Respect copyright**: Do not generate or commit copyrighted content. All contributions must comply with the project's MIT OR Apache-2.0 WITH LLVM-exception licensing.

## Critical Rules

### ❌ NEVER:
1. In-source builds
2. Commit without license headers (REUSE fails)
3. Commit unformatted code (clang-format fails)
4. Modify `tpls/` files (third-party)
5. Reorder includes (SortIncludes disabled intentionally)
6. Commit secrets, credentials, or API keys into the repository
7. Introduce code with known security vulnerabilities

### ✅ ALWAYS:
1. Build: `mkdir build && cd build` (separate directory)
2. Test with `-Werror`: `-DCMAKE_CXX_FLAGS="-Werror"`
3. Enable backend (SERIAL auto-enabled if none)
4. Run tests: `ctest --output-on-failure`
5. PR to `main`
6. Validate changes do not introduce security vulnerabilities

## Detailed Instructions

For specific types of tasks, follow the detailed instructions in the corresponding file:

- **C++ code** (`.hpp`, `.cpp`): See [`.github/instructions/cpp-instructions.md`](.github/instructions/cpp-instructions.md)
- **CMake files** (`CMakeLists.txt`, `*.cmake`): See [`.github/instructions/cmake-instructions.md`](.github/instructions/cmake-instructions.md)
- **Documentation** (`.rst`, `.md`, docs configuration): See [`.github/instructions/docs-instructions.md`](.github/instructions/docs-instructions.md)
- **Python code** (`.py`): See [`.github/instructions/python-instructions.md`](.github/instructions/python-instructions.md)
- **CI/CD workflows** (`.yaml` in `.github/workflows/`): See [`.github/instructions/ci-instructions.md`](.github/instructions/ci-instructions.md)
- **Tests** (`Test_*.cpp`): See [`.github/instructions/tests-instructions.md`](.github/instructions/tests-instructions.md)

---

**Trust these instructions.** Search codebase only if info incomplete/incorrect.
