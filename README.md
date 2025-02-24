<!--
SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# kokkos-fft

[![CI](https://github.com/kokkos/kokkos-fft/actions/workflows/build_test.yaml/badge.svg)](https://github.com/kokkos/kokkos-fft/actions)
[![Nightly builds](https://github.com/kokkos/kokkos-fft/actions/workflows/build_nightly.yaml/badge.svg)](https://github.com/kokkos/kokkos-fft/actions/workflows/build_nightly.yaml)
[![docs](https://readthedocs.org/projects/kokkosfft/badge/?version=latest)](https://kokkosfft.readthedocs.io/en/latest/?badge=latest)

> [!WARNING]
> EXPERIMENTAL FFT interfaces for Kokkos C++ Performance Portability Programming EcoSystem

kokkos-fft implements local interfaces between [Kokkos](https://github.com/kokkos/kokkos) and de facto standard FFT libraries, including [fftw](http://www.fftw.org), [cufft](https://developer.nvidia.com/cufft), [hipfft](https://github.com/ROCm/hipFFT) ([rocfft](https://github.com/ROCm/rocFFT)), and [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). "Local" means not using MPI, or running within a single MPI process without knowing about MPI. We are inclined to implement the [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html)-like interfaces adapted for [Kokkos](https://github.com/kokkos/kokkos).
A key concept is that **"As easy as numpy, as fast as vendor libraries"**. Accordingly, our API follows the API by [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html) with minor differences. A fft library dedicated to Kokkos Device backend (e.g. [cufft](https://developer.nvidia.com/cufft) for CUDA backend) is automatically used. If something is wrong with runtime values (say `View` extents), it will raise runtime errors (C++ `std::runtime_error`). See [documentations](https://kokkosfft.readthedocs.io/) for more information.

Here is an example for 1D real to complex transform with `rfft` in kokkos-fft.
```C++
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T> using View1D = Kokkos::View<T*, execution_space>;
constexpr int n = 4;

View1D<double> x("x", n);
View1D<Kokkos::complex<double> > x_hat("x_hat", n/2+1);

Kokkos::Random_XorShift64_Pool<> random_pool(12345);
Kokkos::fill_random(x, random_pool, 1);
Kokkos::fence();

KokkosFFT::rfft(execution_space(), x, x_hat);
```

This is equivalent to the following python code.

```python3
import numpy as np
x = np.random.rand(4)
x_hat = np.fft.rfft(x)
```

There are two major differences: [`execution_space`](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html) argument and output value (`x_hat`) is an argument of API (not returned value from API). As imagined, kokkos-fft only accepts [Kokkos Views](https://kokkos.org/kokkos-core-wiki/API/core/View.html) as input data. The accessibilities of Views from `execution_space` are statically checked (compilation errors if not accessible).

Depending on a View dimension, it automatically uses the batched plans as follows
```C++
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T> using View2D = Kokkos::View<T**, execution_space>;
constexpr int n0 = 4, n1 = 8;

View2D<double> x("x", n0, n1);
View2D<Kokkos::complex<double> > x_hat("x_hat", n0, n1/2+1);

Kokkos::Random_XorShift64_Pool<> random_pool(12345);
Kokkos::fill_random(x, random_pool, 1);
Kokkos::fence();

// FFT along -1 axis and batched along 0th axis
int axis = -1;
KokkosFFT::rfft(execution_space(), x, x_hat, KokkosFFT::Normalization::backward, axis);
```

This is equivalent to

```python3
import numpy as np
x = np.random.rand(4, 8)
x_hat = np.fft.rfft(x, axis=-1)
```

In this example, the 1D batched `rfft` over 2D View along `axis -1` is executed. Some basic examples are found in [examples](examples).

## Disclaimer
**kokkos-fft is under development and subject to change without warning. The authors do not guarantee that this code runs correctly in all the environments.**

## Using kokkos-fft
For the moment, there are two ways to use kokkos-fft: including as a subdirectory in CMake project or installing as a library. First of all, you need to clone this repo.
```bash
git clone --recursive https://github.com/kokkos/kokkos-fft.git
```

### Prerequisites
To use kokkos-fft, we need the followings:
* `CMake 3.22+`
* `Kokkos 4.4+`
* `gcc 8.3.0+` (CPUs)
* `IntelLLVM 2023.0.0+` (CPUs, Intel GPUs)
* `nvcc 11.0.0+` (NVIDIA GPUs)
* `rocm 5.3.0+` (AMD GPUs)

### CMake
Since kokkos-fft is a header-only library, it is enough to simply add as a subdirectory. It is assumed that kokkos and kokkos-fft are placed under `<project_directory>/tpls`.

Here is an example to use kokkos-fft in the following CMake project.
```
---/
 |
 └──<project_directory>/
    |--tpls
    |    |--kokkos/
    |    └──kokkos-fft/
    |--CMakeLists.txt
    └──hello.cpp
```

The `CMakeLists.txt` would be
```CMake
cmake_minimum_required(VERSION 3.23)
project(kokkos-fft-as-subdirectory LANGUAGES CXX)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-fft)

add_executable(hello-kokkos-fft hello.cpp)
target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)
```

For compilation, we basically rely on the CMake options for Kokkos. For example, the compile options for A100 GPU is as follows.
```
cmake -B build \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON
cmake --build build -j 8
```
This way, all the functionalities are executed on A100 GPUs. For installation, details are provided in the [documentation](https://kokkosfft.readthedocs.io/en/latest/intro/building.html#install-kokkosfft-as-a-library).

## LICENSE

[![License](https://img.shields.io/badge/License-Apache--2.0_WITH_LLVM--exception-blue)](https://spdx.org/licenses/LLVM-exception.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

kokkos-fft is distributed under either the MIT license, or at your option, the Apache-2.0 license with LLVM exception.
