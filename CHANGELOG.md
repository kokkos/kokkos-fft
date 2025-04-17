<!--
SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Change Log

## [0.3.0](https://github.com/kokkos/kokkos-kernels/tree/0.3.0) (2025-04-16)

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
