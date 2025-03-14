<!--
SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Change Log

## [0.3.0](https://github.com/kokkos/kokkos-kernels/tree/0.3.0) (2025-03-14)

**Implemented enhancements:**

- Documentation: Add developer guide in docs [\#232](https://github.com/kokkos/kokkos-fft/pull/232)
- CI: Introduce cmake-format in CI [\#229](https://github.com/kokkos/kokkos-fft/pull/229)
- CI: Add a spell check in CI [\#227](https://github.com/kokkos/kokkos-fft/pull/227)
- CI: Simplify CMake installation in Dockerfiles [\#222](https://github.com/kokkos/kokkos-fft/pull/222)
- CI: Add nightly tests using Kokkos develop branch [\#213](https://github.com/kokkos/kokkos-fft/pull/213)
- Build system: Add check to ensure complex alignment [\#228](https://github.com/kokkos/kokkos-fft/pull/228)
- General Enhancement: make execute a free function [\#223](https://github.com/kokkos/kokkos-fft/pull/223)
- General Enhancement: Introduce global setup/cleanup in APIs [\#220](https://github.com/kokkos/kokkos-fft/pull/220)
- General Enhancement: Remove unnecessary cufftCreate and hipfftCreate to avoid creating plans twice [\#212](https://github.com/kokkos/kokkos-fft/pull/212)
- General Enhancement: Fix RAII issue by introducing wrapper classes for backend plans [\#208](https://github.com/kokkos/kokkos-fft/pull/208)
- General Enhancement: FAdd missing checks [\#196](https://github.com/kokkos/kokkos-fft/pull/196)

**Bug fix:**

- fix: buffer size for rocfft execution info [\#219](https://github.com/kokkos/kokkos-fft/pull/219)
- Fix nightly [\#216](https://github.com/kokkos/kokkos-fft/pull/216)
