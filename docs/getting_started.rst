.. _getting_started:

Using KokkosFFT
===============

For the moment, there are two ways to use KokkosFFT: including as a subdirectory in CMake project or installing as a library. First of all, you need to clone this repo. 
To configure KokkosFFT, we can just use CMake options for Kokkos, which automatically enables the FFT interface on Kokkos device. 
If CMake fails to find a backend FFT library, try :doc:`finding_libraries`.

.. code-block:: bash

    git clone --recursive https://github.com/CExA-project/kokkos-fft.git


CMake (subdirectory)
--------------------

Since KokkosFFT is a header-only library, it is enough to simply add as a subdirectory. It is assumed that kokkos and kokkosFFT are placed under `<project_directory>/tpls`.

Here is an example to use KokkosFFT in the following CMake project.

.. code-block:: bash

    ---/
     |
     └──<project_directory>/
        |--tpls
        |    |--kokkos/
        |    └──kokkos-fft/
        |--CMakeLists.txt
        └──hello.cpp

The `CMakeLists.txt` would be

.. code-block:: CMake

    cmake_minimum_required(VERSION 3.23)
    project(kokkos-fft-as-subdirectory LANGUAGES CXX)

    add_subdirectory(tpls/kokkos)
    add_subdirectory(tpls/kokkos-fft)

    add_executable(hello-kokkos-fft hello.cpp)
    target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)

For compilation, we basically rely on the CMake options for Kokkos. For example, the configure options for A100 GPU is as follows.

.. code-block:: bash

    cmake -DBUILD_TESTING=ON \
          -DCMAKE_CXX_COMPILER=<project_directory>/tpls/kokkos/bin/nvcc_wrapper \
          -DCMAKE_BUILD_TYPE=Release \
          -DKokkos_ENABLE_CUDA=ON \
          -DKokkos_ENABLE_CUDA_CONSTEXPR=ON \
          -DKokkos_ARCH_AMPERE80=ON \
          -DKokkos_ENABLE_CUDA_LAMBDA=On ..

This way, all the functionalities are executed on A100 GPUs.

Install as a library
--------------------

Is is assumed that the Kokkos is installed under `<install_dir>/kokkos` with OpenMP backend. Here is a recipe to install KokkosFFT under `<install_dir>/kokkos_fft`.

.. code-block:: bash

    export KOKKOSFFT_INSTALL_PREFIX=<lib_dir>/kokkosFFT
    export KokkosFFT_DIR=<lib_dir>/kokkosFFT/lib64/cmake/kokkos-fft

    mkdir build_KokkosFFT && cd build_KokkosFFT
    cmake -DBUILD_TESTING=OFF \
          -DCMAKE_CXX_COMPILER=icpx \
          -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX} ..
    cmake --build . -j 8
    cmake --install .

Here is an example to use KokkosFFT in the following CMake project.

.. code-block:: bash

    ---/
     |
     └──<project_directory>/
        |--CMakeLists.txt
        └──hello.cpp

The `CMakeLists.txt` would be

.. code-block:: CMake

    cmake_minimum_required(VERSION 3.23)
    project(kokkos-fft-as-library LANGUAGES CXX)

    ind_package(Kokkos CONFIG REQUIRED)
    find_package(KokkosFFT CONFIG REQUIRED)

    add_executable(hello-kokkos-fft hello.cpp)
    target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)

The code can be built as

.. code-block:: bash

    mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH="<install_dir>/kokkos;<install_dir>/kokkos_fft" ..
    cmake --build . -j 8

KokkosFFT CMake option listing
------------------------------

**KokkosFFT_ENABLE_HOST_AND_DEVICE**: BOOL
  Enable FFT on both host and device. Defaults to OFF.
  If enabled, it is required to use fftw.

**KokkosFFT_INTERNAL_Kokkos**: BOOL
  Build internal Kokkos instead of relying on external one. Defaults to OFF.

**KokkosFFT_ENABLE_EXAMPLES**: BOOL
  Build KokkosFFT examples. Defaults to ON.

**KokkosFFT_ENABLE_TESTS**: BOOL
  Build KokkosFFT tests. Defaults to OFF.

**KokkosFFT_ENABLE_BENCHMARK**: BOOL
  Build benchmarks for KokkosFFT. Defaults to OFF.

**KokkosFFT_ENABLE_DOCS**: BOOL
  Build KokkosFFT documentaion/website. Defaults to OFF.