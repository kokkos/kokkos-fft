# kokkos-fft

[![CI](https://github.com/CExA-project/kokkos-fft/actions/workflows/cmake.yml/badge.svg)](https://github.com/CExA-project/kokkos-fft/actions)

UNOFFICIAL FFT interfaces for Kokkos C++ Performance Portability Programming EcoSystem

KokkosFFT implements local interfaces Kokkos and de facto standard FFT libraries, including fftw, cufft and hipfft. 
"Local" means not using MPI, or running within a
single MPI process without knowing about MPI. We are inclined to implement the [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html) interfaces based on [Kokkos](https://github.com/kokkos/kokkos). 
A key concept is that "As easy as numpy, as fast as vendor libraries". Accordingly, our APIs follow the APIs by [numpy.fft](https://numpy.org/doc/stable/reference/routines.fft.html) with minor differences. If something is wrong with runtime values, it will raise runtime errors. Here is the example for 1D real to complex transform with ```rfft``` in python and KokkosFFT.
```python3
import numpy as np
x = np.random.rand(4)
x_hat = np.fft.rfft(x)
```

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

There are two major differences: ```execution_space``` argument and updating output as argument not returned. As imagined, KokkosFFT only accepts [Kokkos Views](https://kokkos.org/kokkos-core-wiki/API/core/View.html) as input data. The accessibilities of Views from ```execution_space``` is statically checked (compilation errors if not accessible). 

Depending on the View dimension, it automatically uses the batched plans as follows
```python3
import numpy as np
x = np.random.rand(4, 8)
x_hat = np.fft.rfft(x, axis=-1)
```

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

int axis = -1;
KokkosFFT::rfft(execution_space(), x, x_hat, KokkosFFT::FFT_Normalization::BACKWARD, axis); // FFT along -1 axis and batched along 0th axis
```

In this example, the 1D batched ```rfft``` over 2D View along ```axis -1``` is executed. Some basic examples are found in [examples](https://github.com/CExA-project/kokkos-fft/tree/main/examples).

## Disclaimer
KokkosFFT is under development and subject to change without warning. The authors do not guarantee that this code runs correctly in all the environments.

## Using KokkosFFT
For the moment, there are two ways to use KokkosFFT: including as a subdirectory in CMake project or installing as a library. First of all, you need to clone this repo.
```bash
git clone --recursive https://github.com/CExA-project/kokkos-fft.git
```

### CMake
Since KokkosFFT is a header-only library, it is enough to simply add as a subdirectory. It is assumed that kokkos and kokkosFFT are placed under ```<project_directory>/tpls```.
Here is an example to use KokkosFFT in the following CMake project. 
```
---/
 |
 └──<project_directory>/
    |--tpls
    |    |--kokkos/
    |    └──kokkosFFT/
    |--CMakeLists.txt
    └──hello.cpp
```

The ```CMakeLists.txt``` would be 
```CMake
cmake_minimum_required(VERSION 3.23)
project(kokkos-fft LANGUAGES CXX)

add_subdirectory(tpls/kokkos)
add_subdirectory(tpls/kokkos-fft)

add_executable(hello-kokkos-fft hello.cpp)
target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)
```

For the compilation, we basically rely on the CMake options for Kokkos. For example, the configure options for A100 GPU is as follows.
```
cmake -DBUILD_TESTING=ON \
      -DCMAKE_CXX_COMPILER=<project_directory>/tpls/kokkos/bin/nvcc_wrapper \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkos_ENABLE_CUDA_LAMBDA=On ..
```
This way, all the functionalities are executed on A100 GPUs.

### Install as a library
Is is assumed that the Kokkos is installed under ```<lib_dir>/kokkos``` targeting Skylake with OpenMP backend. ```Kokkos_dir``` also needs to be set appropriately. Here is a recipe to install KokkosFFT under ```<lib_dir>/kokkosFFT```.

```bash
export KOKKOSFFT_INSTALL_PREFIX=<lib_dir>/kokkosFFT
export KokkosFFT_DIR=<lib_dir>/kokkosFFT/lib64/cmake/kokkos-fft

mkdir build_KokkosFFT && cd build_KokkosFFT
cmake -DBUILD_TESTING=OFF -DCMAKE_CXX_COMPILER=icpx -DKokkos_ENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX} ..
cmake --build . -j 8
cmake --install .
```

Here is an example to use KokkosFFT in the following CMake project. 
```
---/
 |
 └──<project_directory>/
    |--CMakeLists.txt
    └──hello.cpp
```

The ```CMakeLists.txt``` would be 
```CMake
cmake_minimum_required(VERSION 3.23)
project(kokkos-fft LANGUAGES CXX)

find_package(Kokkos CONFIG REQUIRED)
find_package(KokkosFFT CONFIG REQUIRED)

add_executable(hello-kokkos-fft hello.cpp)
target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)
```

The code can be built as
```bash
export KOKKOSFFT_INSTALL_PREFIX=<lib_dir>/kokkosFFT
export KokkosFFT_DIR=<lib_dir>/kokkosFFT/lib64/cmake/kokkos-fft

mkdir build && cd build
cmake -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_SKX=ON ..
cmake --build . -j 8
```
