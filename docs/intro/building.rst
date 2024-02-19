.. _building:

Building KokkosFFT
==================

This section describes how to build KokkosFFT with some advanced options.
In order to build KokkosFFT, we use CMake with following compilers. 
Kokkos and backend FFT libraries are also necessary.
Available CMake options for KokkosFFT are listed. 

Compiler versions
-----------------

KokkosFFT relies on quite basic functionalieis of Kokkos, and thus it is supposed to work with compilers used for `Kokkos <https://kokkos.org/kokkos-core-wiki/requirements.html>`_. 
However, we have not tested all the listed compilers there and thus recommend the following compilers which we use frequently for testing.

* ``gcc 8.3.0+`` - CPUs
* ``IntelLLVM 2023.0.0+`` - CPUs, Intel GPUs
* ``nvcc 12.0.0+`` - NVIDIA GPUs
* ``rocm 5.3.0+`` - AMD GPUs

Install KokkosFFT as a library
------------------------------

Is is assumed that the Kokkos is installed under ``<install_dir>/kokkos`` with OpenMP backend. Here is a recipe to install KokkosFFT under ``<install_dir>/kokkos_fft``.

.. code-block:: bash

    export KOKKOSFFT_INSTALL_PREFIX=<install_dir>/kokkosFFT

    mkdir build_KokkosFFT && cd build_KokkosFFT
    cmake -DCMAKE_CXX_COMPILER=icpx \
          -DCMAKE_PREFIX_PATH=<install_dir>/kokkos \
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

CMake options
-------------

We rely on CMake to build KokkosFFT, more specifically ``CMake 3.22+``. Here are the lists of CMake option. 
For FFTs on Kokkos device only, we do not need to add extra compile options but for Kokkos ones.
In order to use KokkosFFT from both host and device, it is necessary to add ``KokkosFFT_ENABLE_HOST_AND_DEVICE=ON``.
This option may be useful, for example FFT is used for initialization at host. 
However, to enable this option, we need a pre-installed ``fftw`` for FFT on host, so it is disabled in default
(see :doc:`minimum working example<../samples/05_1DFFT_HOST_DEVICE>`).

.. list-table:: CMake options
   :widths: 25 25 50
   :header-rows: 1

   * - 
     - Description
     - Default
   * - ``KokkosFFT_ENABLE_HOST_AND_DEVICE``
     - Enable FFT on both host and device.
     - OFF
   * - ``KokkosFFT_ENABLE_INTERNAL_KOKKOS``
     - Build internal Kokkos instead of relying on external one.
     - OFF
   * - ``KokkosFFT_ENABLE_EXAMPLES``
     - Build KokkosFFT examples
     - ON
   * - ``KokkosFFT_ENABLE_TESTS``
     - Build KokkosFFT tests
     - OFF
   * - ``KokkosFFT_ENABLE_BENCHMARK``
     - Build benchmarks for KokkosFFT
     - OFF

Kokkos backends
---------------

KokkosFFT requieres ``Kokkos 4.2+``. For the moment, we support following backends for CPUs and GPUs. 
A FFT library dedicated to Kokkos Device backend (e.g. cufft for CUDA backend) is automatically used. 
If CMake fails to find a backend FFT library, see :doc:`How to find fft libraries?<../finding_libraries>`.
We may support experimental backends like ``OPENMPTARGET`` in the future.
 
.. list-table:: ``Host backend``
   :widths: 25 25 25 25
   :header-rows: 1

   * - CMake option
     - Description
     - Backend FFT library
     - Default
   * - ``Kokkos_ENABLE_SERIAL``
     - Serial backend targeting CPUs 
     - ``fftw (Serial)``
     - OFF
   * - ``Kokkos_ENABLE_THREADS``
     - C++ threads backend targeting CPUs 
     - ``fftw (Threads)``
     - OFF
   * - ``Kokkos_ENABLE_OPENMP``
     - OpenMP backend targeting CPUs 
     - ``fftw (OpenMP)``
     - OFF

.. list-table:: ``Device backend``
   :widths: 25 25 25 25
   :header-rows: 1

   * - CMake option
     - Description
     - Backend FFT library
     - Default
   * - ``Kokkos_ENABLE_CUDA``
     - CUDA backend targeting NVIDIA GPUs
     - ``cufft``
     - OFF
   * - ``Kokkos_ENABLE_HIP``
     - HIP backend targeting AMD GPUs
     - ``hipfft``
     - OFF
   * - ``Kokkos_ENABLE_SYCL``
     - SYCL backend targeting Intel GPUs
     - ``oneMKL``
     - OFF