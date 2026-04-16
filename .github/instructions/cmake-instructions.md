<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# CMake Instructions for Kokkos-FFT

These instructions apply when creating, modifying, or reviewing CMake files (`CMakeLists.txt`, `*.cmake`) in this repository.

## CMake Version

- **Minimum required**: CMake 3.22.
- Use features available in CMake 3.22+. Do not use features from newer versions without updating the minimum.
- When updating the CMake minimum version, update the descriptions in documentation and instruction files (including this file) as well

## License Headers

Every CMake file must start with the SPDX license header:

```cmake
# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
```

CI enforces REUSE compliance — omitting this header causes build failure.

## Formatting

- CMake files are formatted with `cmake-format` using the project's `.cmake-format.py` configuration.
- Key settings: line width 120, dangling parentheses enabled, prefix-aligned dangling, comment markup reflow disabled.
- Run `cmake-format --in-place <file>` before committing.
- CI will reject unformatted CMake code.

## Naming Conventions

| Entity | Convention | Examples |
|--------|-----------|---------|
| Project options | `KokkosFFT_ENABLE_<FEATURE>` | `KokkosFFT_ENABLE_TESTS`, `KokkosFFT_ENABLE_CUFFT` |
| Internal variables | `KOKKOSFFT_<NAME>` | `KOKKOSFFT_TPL_LIST`, `KOKKOSFFT_BACKEND_LIST` |
| Targets | lowercase descriptive names | `common`, `fft`, `unit-tests-kokkos-fft-common` |
| Alias targets | `KokkosFFT::<target>` | `KokkosFFT::fft`, `KokkosFFT::common` |
| Find modules | `Find<PackageName>.cmake` | `FindFFTW.cmake`, `FindcuFFTMp.cmake` |
| Utility modules | `KokkosFFT_<name>.cmake` | `KokkosFFT_tpls.cmake`, `KokkosFFT_utils.cmake` |
| CMake functions | `snake_case` | `get_tpls_list()`, `pad_string()` |

## Project Options

Options are defined using the `option()` command with the `KokkosFFT_ENABLE_` prefix:

```cmake
option(KokkosFFT_ENABLE_TESTS "Build KokkosFFT tests" OFF)
option(KokkosFFT_ENABLE_EXAMPLES "Build KokkosFFT examples" OFF)
option(KokkosFFT_ENABLE_BENCHMARK "Build benchmarks" OFF)
```

- All feature options default to `OFF` unless there is an auto-detection mechanism.
- Backend options (`KokkosFFT_ENABLE_CUFFT`, etc.) may auto-detect based on the Kokkos configuration.

## Header-Only Library Targets

The library uses CMake INTERFACE targets since it is header-only:

```cmake
add_library(component_name INTERFACE)
target_link_libraries(component_name INTERFACE Kokkos::kokkos)
target_compile_features(component_name INTERFACE cxx_std_17)
target_include_directories(
    component_name INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                             $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
add_library(KokkosFFT::component_name ALIAS component_name)
```

- Use `INTERFACE` keyword for all library properties.
- Create alias targets under the `KokkosFFT::` namespace.
- Use generator expressions for install vs. build include directories.

## Kokkos Integration

```cmake
if(NOT KokkosFFT_ENABLE_INTERNAL_KOKKOS)
    if(NOT TARGET Kokkos::kokkos)
        find_package(Kokkos ${KOKKOS_REQUIRED_VERSION} REQUIRED)
        kokkos_check(OPTIONS COMPLEX_ALIGN)
    endif()
else()
    add_subdirectory(tpls/kokkos)
endif()
```

- Support both external Kokkos (`find_package`) and internal submodule (`add_subdirectory`).
- Always check for the required Kokkos version.
- Use `kokkos_check()` to verify required Kokkos options.

## Backend (TPL) Management

- Backend selection logic lives in `cmake/KokkosFFT_tpls.cmake`.
- Device backends (cuFFT, hipFFT, rocFFT, oneMKL) are mutually exclusive — enforce this with error messages.
- Host backend (FFTW) supports multiple threading modes: OpenMP, Threads, Serial.
- Use `target_compile_definitions()` with `INTERFACE` to set backend preprocessor defines.

## Test Configuration

```cmake
add_executable(test-executable-name Test_Main.cpp Test_Component.cpp)
target_compile_features(test-executable-name PUBLIC cxx_std_17)
target_link_libraries(test-executable-name PUBLIC KokkosFFT::fft GTest::gtest)

include(GoogleTest)
gtest_discover_tests(
    test-executable-name PROPERTIES DISCOVERY_TIMEOUT 600 DISCOVERY_MODE PRE_TEST
)
```

- Use `gtest_discover_tests()` with `DISCOVERY_TIMEOUT 600` and `DISCOVERY_MODE PRE_TEST`.
- Test executables follow the naming pattern `unit-tests-kokkos-fft-<module>`.
- Link against `GTest::gtest` (not `gtest` directly) for proper imported target usage.

## MPI Test Configuration

For distributed (MPI) tests, use direct `add_test()` with `MPIEXEC_EXECUTABLE`:

```cmake
add_test(
    NAME UnitTests1MPI
    COMMAND ${MPIEXEC_EXECUTABLE} -n 1 $<TARGET_FILE:unit-tests-mpi>
)
set_property(TEST UnitTests1MPI PROPERTY TIMEOUT 600)
```

- Test with 1, 2, and 4 MPI processes.
- Set appropriate timeouts (600s for 1-2 processes, 1200s for 4 processes).

## Find Modules

Custom find modules are in `cmake/` and follow the CMake `Find<Package>.cmake` pattern:

- Set `<Package>_FOUND`, `<Package>_INCLUDE_DIRS`, `<Package>_LIBRARIES`.
- Use `find_package_handle_standard_args()` for consistent output.
- Check for both libraries and headers.

## Installation

```cmake
install(
    TARGETS target_name
    EXPORT KokkosFFTTargets
)

install(
    DIRECTORY path/to/headers/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp"
)
```

- Export targets via the `KokkosFFTTargets` export set.
- Use `CMAKE_INSTALL_INCLUDEDIR` for header installation paths.
- Install only `.hpp` files with `FILES_MATCHING PATTERN`.

## Configuration Files

- Config template: `cmake/KokkosFFTConfig.cmake.in` — uses `@PACKAGE_INIT@` from `CMakePackageConfigHelpers`.
- Version file: Generated with `write_basic_package_version_file()`.
- Generated headers: Use `configure_file()` with `@ONLY` for version and config headers.

## Build Directory

- **Always use out-of-source builds.** Never build in the source directory.
- Standard pattern: `cmake -B build [options] && cmake --build build -j $(nproc)`.

## Common Patterns

### Conditional subdirectory inclusion

```cmake
if(KokkosFFT_ENABLE_TESTS)
    add_subdirectory(unit_test)
endif()
```

### Conditional target linking

```cmake
foreach(TPL ${KOKKOSFFT_TPL_LIST})
    target_link_libraries(fft INTERFACE ${TPL})
endforeach()
```

### Status messages

Use the `pad_string()` utility from `KokkosFFT_utils.cmake` for aligned status output:

```cmake
pad_string(label "Option Name" 30)
message(STATUS "${label}: ${value}")
```

## Helper Functions

- Helper functions must be defined in `*.cmake` files under the `cmake/` directory.
- Use `snake_case` for function names.
- Add docstrings as comments above the function explaining purpose, parameters, and usage.
- Example:

```cmake
# Pad a string with spaces to a given length for aligned status output.
#
# Parameters:
#   output_string - variable name to store the padded result
#   input_string  - the string to pad
#   length        - the desired total length
function(pad_string output_string input_string length)
    string(LENGTH "${input_string}" input_length)
    math(EXPR pad_length "${length} - ${input_length}")
    if(pad_length GREATER 0)
        string(REPEAT " " ${pad_length} padding)
    else()
        set(padding "")
    endif()
    set(${output_string} "${input_string}${padding}" PARENT_SCOPE)
endfunction()
```

## Things to Avoid

- Do not hardcode paths — use CMake variables and generator expressions.
- Do not use `include_directories()` — use `target_include_directories()`.
- Do not use `link_libraries()` — use `target_link_libraries()`.
- Do not use `add_definitions()` — use `target_compile_definitions()`.
- Do not modify `tpls/` CMake files.
- Do not enable multiple device backends simultaneously.
- Do not perform in-source builds.
