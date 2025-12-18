<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Change Log

## [1.0.0](https://github.com/kokkos/kokkos-fft/tree/1.0.0) (2025-12-18)

**Implemented enhancements:**

- Documentation: Add docstring to get_map_axes[\#340](https://github.com/kokkos/kokkos-fft/pull/340)
- Documentation: Add a link to JOSS paper [\#304](https://github.com/kokkos/kokkos-fft/pull/304)
- CI: Add spack spec and fix install-test in spack CI [\#324](https://github.com/kokkos/kokkos-fft/pull/324)
- CI: Add spack installation CI for SYCL backend [\#323](https://github.com/kokkos/kokkos-fft/pull/323)
- CI: Add spack installation CI for HIP backend [\#318](https://github.com/kokkos/kokkos-fft/pull/318)
- CI: Update rocm Dockerfile to use hipcc 6.2.0 [\#316](https://github.com/kokkos/kokkos-fft/pull/316)
- CI: Update C++ version to 20 in nightly workflow [\#314](https://github.com/kokkos/kokkos-fft/pull/314)
- CI: Add spack installation CI for CUDA backend [\#312](https://github.com/kokkos/kokkos-fft/pull/312)
- General Enhancement: Update to kokkos4.6 [\#385](https://github.com/kokkos/kokkos-fft/pull/385)
- General Enhancement: use std::size_t to manipulate extents [\#384](https://github.com/kokkos/kokkos-fft/pull/384)
- General Enhancement: Refactor normalization and unit-tests [\#381](https://github.com/kokkos/kokkos-fft/pull/381)
- General Enhancement: Move get_map_axes into KokkosFFT_Mapping.hpp [\#377](https://github.com/kokkos/kokkos-fft/pull/377)
- General Enhancement: Expose DynPlans [\#376](https://github.com/kokkos/kokkos-fft/pull/376)
- General Enhancement: Add dynamic plan for SYCL backend [\#373](https://github.com/kokkos/kokkos-fft/pull/373)
- General Enhancement: Add dynamic plan for ROCM backend [\#372](https://github.com/kokkos/kokkos-fft/pull/372)
- General Enhancement: Add dynamic plan for HIPFFT backend [\#371](https://github.com/kokkos/kokkos-fft/pull/371)
- General Enhancement: Rephrase the static assertion message regarding data accessibility [\#370](https://github.com/kokkos/kokkos-fft/pull/370)
- General Enhancement: Rename ger_r2c_extent and make it public [\#369](https://github.com/kokkos/kokkos-fft/pull/369)
- General Enhancement: Add dynamic plan for CUDA backend [\#366](https://github.com/kokkos/kokkos-fft/pull/366)
- General Enhancement: Add extents test for dynamic rank case [\#364](https://github.com/kokkos/kokkos-fft/pull/364)
- General Enhancement: Improve iteration pattern in transpose [\#355](https://github.com/kokkos/kokkos-fft/pull/355)
- General Enhancement: Testing transpose with bound checks [\#352](https://github.com/kokkos/kokkos-fft/pull/352)
- General Enhancement: Rename iType to IndexType to align with mdspan convention [\#351](https://github.com/kokkos/kokkos-fft/pull/351)
- General Enhancement: Allow is_transpose_needed to work on std::size_t based array [\#350](https://github.com/kokkos/kokkos-fft/pull/350)
- General Enhancement: Introduce a helper to make a reference for transposed view [\#349](https://github.com/kokkos/kokkos-fft/pull/349)
- General Enhancement: Use execution space instance in Transpose Unit tests [\#348](https://github.com/kokkos/kokkos-fft/pull/348)
- General Enhancement: Copy if map is identical in transpose helper [\#345](https://github.com/kokkos/kokkos-fft/pull/345)
- General Enhancement: Make transpose helper to work on Views with different layout [\#344](https://github.com/kokkos/kokkos-fft/pull/344)
- General Enhancement: Introduce helper function to wrap std::reverse [\#343](https://github.com/kokkos/kokkos-fft/pull/343)
- General Enhancement: Make compute_strides function a public function [\#342](https://github.com/kokkos/kokkos-fft/pull/342)
- General Enhancement: Use std::size_t in transpose test [\#341](https://github.com/kokkos/kokkos-fft/pull/341)
- General Enhancement: Introduce safe transpose [\#339](https://github.com/kokkos/kokkos-fft/pull/339)
- General Enhancement: Extract the core logic of get_map_axes helper [\#337](https://github.com/kokkos/kokkos-fft/pull/337)
- General Enhancement: Add base type convert helper of containers [\#336](https://github.com/kokkos/kokkos-fft/pull/336)
- General Enhancement: Add base type convert helper of containers [\#336](https://github.com/kokkos/kokkos-fft/pull/336)
- General Enhancement: Improve convert negative axis to work on containers [\#334](https://github.com/kokkos/kokkos-fft/pull/334)
- General Enhancement: Fix the underflow/overflow behavior in total_size function [\#331](https://github.com/kokkos/kokkos-fft/pull/331)
- General Enhancement: Add product helper [\#328](https://github.com/kokkos/kokkos-fft/pull/328)

**Bug fix:**

- Bug Fix: Add const in order to suppress warnings by nvcc [\#365](https://github.com/kokkos/kokkos-fft/pull/365)
- Bug Fix: Interpret the array of perturbations relative to the original array [\#357](https://github.com/kokkos/kokkos-fft/pull/357)
- Bug Fix: Use the new name of a Kokkos option [\#322](https://github.com/kokkos/kokkos-fft/pull/322)

## [0.4.0](https://github.com/kokkos/kokkos-fft/tree/0.4.0) (2025-07-29)

**Implemented enhancements:**

- Documentation: Detail license in README [\#299](https://github.com/kokkos/kokkos-fft/pull/299)
- Documentation: Fixes a link in the documentation [\#296](https://github.com/kokkos/kokkos-fft/pull/296)
- Documentation: Suppress disclaimer [\#282](https://github.com/kokkos/kokkos-fft/pull/282)
- Documentation: Typo fix in README.md [\#281](https://github.com/kokkos/kokkos-fft/pull/281)
- CI: Build CUDA backend with cxx20 in CI and enable testing-tools [\#272](https://github.com/kokkos/kokkos-fft/pull/272)
- General Enhancement: Print in/out view details and axes in case of extent errors [\#301](https://github.com/kokkos/kokkos-fft/pull/301)
- General Enhancement: Improve error message of a failing find_package for FFTW [\#293](https://github.com/kokkos/kokkos-fft/pull/293)
- General Enhancement: Update to kokkos 4.5.1 [\#292](https://github.com/kokkos/kokkos-fft/pull/292)
- General Enhancement: Define KOKKOSFFT_ENABLE_TPL_\<NAME\> through KokkosFFT_config.hpp [\#291](https://github.com/kokkos/kokkos-fft/pull/291)
- General Enhancement: Make convert_negative_axis a templated function again [\#288](https://github.com/kokkos/kokkos-fft/pull/288)
- General Enhancement: Allow index_sequence to work on unsigned integers [\#287](https://github.com/kokkos/kokkos-fft/pull/287)
- General Enhancement: Refactor convert_negative_axis to unuse View [\#286](https://github.com/kokkos/kokkos-fft/pull/286)
- General Enhancement: Use value_or instead of maybe_null_to_shape [\#276](https://github.com/kokkos/kokkos-fft/pull/276)
- General Enhancement: Delegating Plan class constructor [\#275](https://github.com/kokkos/kokkos-fft/pull/275)
- General Enhancement: Merging transpose implementation details into a single functor [\#274](https://github.com/kokkos/kokkos-fft/pull/274)
- General Enhancement: Refactor 1D-8D roll functors [\#273](https://github.com/kokkos/kokkos-fft/pull/273)
- General Enhancement: Add const for container types [\#270](https://github.com/kokkos/kokkos-fft/pull/270)
- General Enhancement: Allow real input to fft and hfft [\#263](https://github.com/kokkos/kokkos-fft/pull/263)
- General Enhancement: Introduce a helper to convert a view type with different base value type [\#262](https://github.com/kokkos/kokkos-fft/pull/262)

**Bug fix:**

## [0.3.0](https://github.com/kokkos/kokkos-fft/tree/0.3.0) (2025-04-16)

**Implemented enhancements:**

- Documentation: Fix docs with CMake build [\#259](https://github.com/kokkos/kokkos-fft/pull/259)
- Documentation: Fixes a link in the documentation [\#258](https://github.com/kokkos/kokkos-fft/pull/258)
- Documentation: Add developer guide in docs [\#232](https://github.com/kokkos/kokkos-fft/pull/232)
- CI: Introduce link checker in CI [\#260](https://github.com/kokkos/kokkos-fft/pull/260)
- CI: Update dependabot.yml [\#245](https://github.com/kokkos/kokkos-fft/pull/245)
- CI: Introduce version check [\#238](https://github.com/kokkos/kokkos-fft/pull/238)
- CI: Simplify the creation of Docker/Singularity images [\#237](https://github.com/kokkos/kokkos-fft/pull/237)
- CI: Introduce pylinter in CI [\#234](https://github.com/kokkos/kokkos-fft/pull/234)
- CI: Introduce cmake-format in CI [\#229](https://github.com/kokkos/kokkos-fft/pull/229)
- CI: Add a spell check in CI [\#227](https://github.com/kokkos/kokkos-fft/pull/227)
- CI: Simplify CMake installation in Dockerfiles [\#222](https://github.com/kokkos/kokkos-fft/pull/222)
- CI: Add nightly tests using Kokkos develop branch [\#213](https://github.com/kokkos/kokkos-fft/pull/213)
- Build system: Linking library logic is moved from common/src/CMake to fft/src/CMake [\#251](https://github.com/kokkos/kokkos-fft/pull/251)
- Build system: Allow compilation while disabling device fft libraries [\#249](https://github.com/kokkos/kokkos-fft/pull/249)
- Build system: Add CMake options to specify backend FFT libs [\#239](https://github.com/kokkos/kokkos-fft/pull/239)
- Build system: Add check to ensure complex alignment [\#228](https://github.com/kokkos/kokkos-fft/pull/228)
- General Enhancement: Set commit SHA instead of version of github actions [\#257](https://github.com/kokkos/kokkos-fft/pull/257)
- General Enhancement: Unuse KokkosFFT_ENABLE_HOST_AND_DEVICE [\#256](https://github.com/kokkos/kokkos-fft/pull/256)
- General Enhancement: Improve roll function [\#254](https://github.com/kokkos/kokkos-fft/pull/254)
- General Enhancement: Refactor roll operation used in fftshift/ifftshift [\#253](https://github.com/kokkos/kokkos-fft/pull/253)
- General Enhancement: Run fill_random on a given execution space instance [\#250](https://github.com/kokkos/kokkos-fft/pull/250)
- General Enhancement: Refactor unit-tests [\#241](https://github.com/kokkos/kokkos-fft/pull/241)
- General Enhancement: Improve example docs [\#236](https://github.com/kokkos/kokkos-fft/pull/236)
- General Enhancement: Cleanup implementation details [\#235](https://github.com/kokkos/kokkos-fft/pull/235)
- General Enhancement: Make execute a free function [\#223](https://github.com/kokkos/kokkos-fft/pull/223)
- General Enhancement: Introduce global setup/cleanup in APIs [\#220](https://github.com/kokkos/kokkos-fft/pull/220)
- General Enhancement: Remove unnecessary cufftCreate and hipfftCreate to avoid creating plans twice [\#212](https://github.com/kokkos/kokkos-fft/pull/212)
- General Enhancement: Fix RAII issue by introducing wrapper classes for backend plans [\#208](https://github.com/kokkos/kokkos-fft/pull/208)
- General Enhancement: Add missing checks [\#196](https://github.com/kokkos/kokkos-fft/pull/196)

**Bug fix:**

- Fix: buffer size for rocfft execution info [\#219](https://github.com/kokkos/kokkos-fft/pull/219)
- Fix nightly [\#216](https://github.com/kokkos/kokkos-fft/pull/216)
