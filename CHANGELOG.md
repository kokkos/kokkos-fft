<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Change Log

## [1.1.0](https://github.com/kokkos/kokkos-fft/releases/tag/v1.1.0) (2026-05-13)

**Documentation:**

- Add rfftn/irfftn examples [\#461](https://github.com/kokkos/kokkos-fft/pull/461)
- Add rfft2/irfft2 examples [\#460](https://github.com/kokkos/kokkos-fft/pull/460)
- Add rfft/irfft example [\#433](https://github.com/kokkos/kokkos-fft/pull/433)
- Add fftn/ifftn examples [\#430](https://github.com/kokkos/kokkos-fft/pull/430)
- Add fft2/ifft2 examples [\#428](https://github.com/kokkos/kokkos-fft/pull/428)
- Add ifft() example in API reference [\#427](https://github.com/kokkos/kokkos-fft/pull/427)
- Add fft() example in API reference [\#424](https://github.com/kokkos/kokkos-fft/pull/424)

**CI:**

- Introduce pre-commit CI [\#462](https://github.com/kokkos/kokkos-fft/pull/462)
- Add MI300A nightly via HPSF CI [\#447](https://github.com/kokkos/kokkos-fft/pull/447)
- Update CI on Ruche [\#415](https://github.com/kokkos/kokkos-fft/pull/415), [\#434](https://github.com/kokkos/kokkos-fft/pull/434)
- Fixes CI (distributed) for Intel [\#397](https://github.com/kokkos/kokkos-fft/pull/397)

**General Enhancement:**

- Update to kokkos 4.7 [\#466](https://github.com/kokkos/kokkos-fft/pull/466)
- Add md operator helper [\#446](https://github.com/kokkos/kokkos-fft/pull/446)
- Add .github/copilot-instructions.md for cloud agent onboarding [\#438](https://github.com/kokkos/kokkos-fft/pull/438)
- Allow inplace FFT even with internal transpose [\#429](https://github.com/kokkos/kokkos-fft/pull/429)
- [cuFFT] Special assertion handler for cuFFT backend [\#407](https://github.com/kokkos/kokkos-fft/pull/407)
- [HIPFFT] Special assertion handler for HIPFFT backend [\#409](https://github.com/kokkos/kokkos-fft/pull/409)
- [ROCFFT] Special assertion handler for ROCFFT backend [\#410](https://github.com/kokkos/kokkos-fft/pull/410)

**Bug fix:**

- Fixes arange and add docs [\#458](https://github.com/kokkos/kokkos-fft/pull/458)
- Fixes nvcc pointless-comparison-warning [\#456](https://github.com/kokkos/kokkos-fft/pull/456)
- Fixes transpose tests [\#448](https://github.com/kokkos/kokkos-fft/pull/448)

**Internal Changes:**

- Refactor common helper functions [\#435](https://github.com/kokkos/kokkos-fft/pull/435), [\#437](https://github.com/kokkos/kokkos-fft/pull/437), [\#441](https://github.com/kokkos/kokkos-fft/pull/441), [\#449](https://github.com/kokkos/kokkos-fft/pull/449), [\#451](https://github.com/kokkos/kokkos-fft/pull/451), [\#452](https://github.com/kokkos/kokkos-fft/pull/452), [\#453](https://github.com/kokkos/kokkos-fft/pull/453), [\#454](https://github.com/kokkos/kokkos-fft/pull/454), [\#455](https://github.com/kokkos/kokkos-fft/pull/455), [\#457](https://github.com/kokkos/kokkos-fft/pull/457), [\#459](https://github.com/kokkos/kokkos-fft/pull/459)
- Remove compute transpose extents [\#432](https://github.com/kokkos/kokkos-fft/pull/432)

## [1.0.0](https://github.com/kokkos/kokkos-fft/releases/tag/v1.0.0) (2026-01-26)

**Documentation:**

- Rename kokkos-fft to Kokkos-FFT[\#388](https://github.com/kokkos/kokkos-fft/pull/388)
- Align prerequisites with Kokkos 5.0[\#387](https://github.com/kokkos/kokkos-fft/pull/387)
- Add docstring to get_map_axes[\#340](https://github.com/kokkos/kokkos-fft/pull/340)
- Add a link to JOSS paper [\#304](https://github.com/kokkos/kokkos-fft/pull/304)

**CI:**

- Add spack spec and fix install-test in spack CI [\#324](https://github.com/kokkos/kokkos-fft/pull/324)
- Add spack installation CI for SYCL backend [\#323](https://github.com/kokkos/kokkos-fft/pull/323)
- Add spack installation CI for HIP backend [\#318](https://github.com/kokkos/kokkos-fft/pull/318)
- Update rocm Dockerfile to use hipcc 6.2.0 [\#316](https://github.com/kokkos/kokkos-fft/pull/316)
- Update C++ version to 20 in nightly workflow [\#314](https://github.com/kokkos/kokkos-fft/pull/314)
- Add spack installation CI for CUDA backend [\#312](https://github.com/kokkos/kokkos-fft/pull/312)

**General Enhancement:**

- Update to kokkos4.6 [\#385](https://github.com/kokkos/kokkos-fft/pull/385)
- Use std::size_t to manipulate extents [\#384](https://github.com/kokkos/kokkos-fft/pull/384)
- Refactor normalization and unit-tests [\#381](https://github.com/kokkos/kokkos-fft/pull/381)
- Expose DynPlans [\#376](https://github.com/kokkos/kokkos-fft/pull/376)
- Rephrase the static assertion message regarding data accessibility [\#370](https://github.com/kokkos/kokkos-fft/pull/370)

**Bug fix:**

- Add const in order to suppress warnings by nvcc [\#365](https://github.com/kokkos/kokkos-fft/pull/365)
- Interpret the array of perturbations relative to the original array [\#357](https://github.com/kokkos/kokkos-fft/pull/357)
- Use the new name of a Kokkos option [\#322](https://github.com/kokkos/kokkos-fft/pull/322)

**Internal Changes:**

- Define is_AllowedSpace in default_types.hpp [\#390](https://github.com/kokkos/kokkos-fft/pull/390)
- Introduce layout header [\#380](https://github.com/kokkos/kokkos-fft/pull/380)
- Move get_map_axes into KokkosFFT_Mapping.hpp [\#377](https://github.com/kokkos/kokkos-fft/pull/377)
- Add dynamic plan for SYCL backend [\#373](https://github.com/kokkos/kokkos-fft/pull/373)
- Add dynamic plan for ROCM backend [\#372](https://github.com/kokkos/kokkos-fft/pull/372)
- Add dynamic plan for HIPFFT backend [\#371](https://github.com/kokkos/kokkos-fft/pull/371)
- Rename ger_r2c_extent and make it public [\#369](https://github.com/kokkos/kokkos-fft/pull/369)
- Add dynamic plan for CUDA backend [\#366](https://github.com/kokkos/kokkos-fft/pull/366)
- Add extents test for dynamic rank case [\#364](https://github.com/kokkos/kokkos-fft/pull/364)
- Improve iteration pattern in transpose [\#355](https://github.com/kokkos/kokkos-fft/pull/355)
- Testing transpose with bound checks [\#352](https://github.com/kokkos/kokkos-fft/pull/352)
- Rename iType to IndexType to align with mdspan convention [\#351](https://github.com/kokkos/kokkos-fft/pull/351)
- Allow is_transpose_needed to work on std::size_t based array [\#350](https://github.com/kokkos/kokkos-fft/pull/350)
- Introduce a helper to make a reference for transposed view [\#349](https://github.com/kokkos/kokkos-fft/pull/349)
- Use execution space instance in Transpose Unit tests [\#348](https://github.com/kokkos/kokkos-fft/pull/348)
- Introduce helper function to wrap std::reverse [\#343](https://github.com/kokkos/kokkos-fft/pull/343)
- Make compute_strides function a public function [\#342](https://github.com/kokkos/kokkos-fft/pull/342)
- Use std::size_t in transpose test [\#341](https://github.com/kokkos/kokkos-fft/pull/341)
- Extract the core logic of get_map_axes helper [\#337](https://github.com/kokkos/kokkos-fft/pull/337)
- Add base type convert helper of containers [\#336](https://github.com/kokkos/kokkos-fft/pull/336)
- Add base type convert helper of containers [\#336](https://github.com/kokkos/kokkos-fft/pull/336)
- Improve convert negative axis to work on containers [\#334](https://github.com/kokkos/kokkos-fft/pull/334)
- Add product helper [\#328](https://github.com/kokkos/kokkos-fft/pull/328)

## [0.4.0](https://github.com/kokkos/kokkos-fft/releases/tag/v0.4.0) (2025-07-29)

**Documentation:**

- Detail license in README [\#299](https://github.com/kokkos/kokkos-fft/pull/299)
- Fixes a link in the documentation [\#296](https://github.com/kokkos/kokkos-fft/pull/296)
- Suppress disclaimer [\#282](https://github.com/kokkos/kokkos-fft/pull/282)
- Typo fix in README.md [\#281](https://github.com/kokkos/kokkos-fft/pull/281)

**CI:**

- CI: Build CUDA backend with cxx20 in CI and enable testing-tools [\#272](https://github.com/kokkos/kokkos-fft/pull/272)

**General Enhancement:**

- Print in/out view details and axes in case of extent errors [\#301](https://github.com/kokkos/kokkos-fft/pull/301)
- Improve error message of a failing find_package for FFTW [\#293](https://github.com/kokkos/kokkos-fft/pull/293)
- Update to kokkos 4.5.1 [\#292](https://github.com/kokkos/kokkos-fft/pull/292)
- Define KOKKOSFFT_ENABLE_TPL_\<NAME\> through KokkosFFT_config.hpp [\#291](https://github.com/kokkos/kokkos-fft/pull/291)
- Make convert_negative_axis a templated function again [\#288](https://github.com/kokkos/kokkos-fft/pull/288)
- Allow index_sequence to work on unsigned integers [\#287](https://github.com/kokkos/kokkos-fft/pull/287)
- Refactor convert_negative_axis to unuse View [\#286](https://github.com/kokkos/kokkos-fft/pull/286)
- Use value_or instead of maybe_null_to_shape [\#276](https://github.com/kokkos/kokkos-fft/pull/276)
- Delegating Plan class constructor [\#275](https://github.com/kokkos/kokkos-fft/pull/275)
- Merging transpose implementation details into a single functor [\#274](https://github.com/kokkos/kokkos-fft/pull/274)
- Refactor 1D-8D roll functors [\#273](https://github.com/kokkos/kokkos-fft/pull/273)
- Add const for container types [\#270](https://github.com/kokkos/kokkos-fft/pull/270)
- Allow real input to fft and hfft [\#263](https://github.com/kokkos/kokkos-fft/pull/263)
- Introduce a helper to convert a view type with different base value type [\#262](https://github.com/kokkos/kokkos-fft/pull/262)

**Bug fix:**

## [0.3.0](https://github.com/kokkos/kokkos-fft/releases/tag/v0.3.0) (2025-04-16)

**Documentation:**

- Fix docs with CMake build [\#259](https://github.com/kokkos/kokkos-fft/pull/259)
- Fixes a link in the documentation [\#258](https://github.com/kokkos/kokkos-fft/pull/258)
- Add developer guide in docs [\#232](https://github.com/kokkos/kokkos-fft/pull/232)

**CI:**

- Introduce link checker in CI [\#260](https://github.com/kokkos/kokkos-fft/pull/260)
- Update dependabot.yml [\#245](https://github.com/kokkos/kokkos-fft/pull/245)
- Introduce version check [\#238](https://github.com/kokkos/kokkos-fft/pull/238)
- Simplify the creation of Docker/Singularity images [\#237](https://github.com/kokkos/kokkos-fft/pull/237)
- Introduce pylinter in CI [\#234](https://github.com/kokkos/kokkos-fft/pull/234)
- Introduce cmake-format in CI [\#229](https://github.com/kokkos/kokkos-fft/pull/229)
- Add a spell check in CI [\#227](https://github.com/kokkos/kokkos-fft/pull/227)
- Simplify CMake installation in Dockerfiles [\#222](https://github.com/kokkos/kokkos-fft/pull/222)
- Add nightly tests using Kokkos develop branch [\#213](https://github.com/kokkos/kokkos-fft/pull/213)

**Build system:**

- Linking library logic is moved from common/src/CMake to fft/src/CMake [\#251](https://github.com/kokkos/kokkos-fft/pull/251)
- Allow compilation while disabling device fft libraries [\#249](https://github.com/kokkos/kokkos-fft/pull/249)
- Add CMake options to specify backend FFT libs [\#239](https://github.com/kokkos/kokkos-fft/pull/239)
- Add check to ensure complex alignment [\#228](https://github.com/kokkos/kokkos-fft/pull/228)

**General Enhancement:**

- Set commit SHA instead of version of github actions [\#257](https://github.com/kokkos/kokkos-fft/pull/257)
- Unuse KokkosFFT_ENABLE_HOST_AND_DEVICE [\#256](https://github.com/kokkos/kokkos-fft/pull/256)
- Improve roll function [\#254](https://github.com/kokkos/kokkos-fft/pull/254)
- Refactor roll operation used in fftshift/ifftshift [\#253](https://github.com/kokkos/kokkos-fft/pull/253)
- Run fill_random on a given execution space instance [\#250](https://github.com/kokkos/kokkos-fft/pull/250)
- Refactor unit-tests [\#241](https://github.com/kokkos/kokkos-fft/pull/241)
- Improve example docs [\#236](https://github.com/kokkos/kokkos-fft/pull/236)
- Cleanup implementation details [\#235](https://github.com/kokkos/kokkos-fft/pull/235)
- Make execute a free function [\#223](https://github.com/kokkos/kokkos-fft/pull/223)
- Introduce global setup/cleanup in APIs [\#220](https://github.com/kokkos/kokkos-fft/pull/220)
- Remove unnecessary cufftCreate and hipfftCreate to avoid creating plans twice [\#212](https://github.com/kokkos/kokkos-fft/pull/212)
- Fix RAII issue by introducing wrapper classes for backend plans [\#208](https://github.com/kokkos/kokkos-fft/pull/208)
- Add missing checks [\#196](https://github.com/kokkos/kokkos-fft/pull/196)

**Bug fix:**

- Fix: buffer size for rocfft execution info [\#219](https://github.com/kokkos/kokkos-fft/pull/219)
- Fix nightly [\#216](https://github.com/kokkos/kokkos-fft/pull/216)
