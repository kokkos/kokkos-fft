.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

.. _building:

Building kokkos-fft
===================

This section describes how to build kokkos-fft with some advanced options.
In order to build kokkos-fft, we use ``CMake`` with following compilers. 
Kokkos and backend FFT libraries are also necessary.
Available CMake options for kokkos-fft are listed. 

Compiler versions
-----------------

kokkos-fft relies on quite basic functionalities of Kokkos, and thus it is supposed to work with compilers used for `Kokkos <https://kokkos.org/kokkos-core-wiki/get-started/requirements.html>`_.
However, we have not tested all the listed compilers there and thus recommend the following compilers which we use frequently for testing.

* ``gcc 8.3.0+`` - CPUs
* ``IntelLLVM 2023.0.0+`` - CPUs, Intel GPUs
* ``nvcc 11.0.0+`` - NVIDIA GPUs
* ``rocm 5.3.0+`` - AMD GPUs

Install kokkos-fft with CMake
-----------------------------

Let's assume Kokkos is installed under ``<path/to/kokkos>`` with ``OpenMP`` backend. We build and install kokkos-fft under ``<path/to/kokkos-fft>``.

.. code-block:: bash

    export KOKKOSFFT_INSTALL_PREFIX=<path/to/kokkos-fft>

    cmake -B build_KokkosFFT \
          -DCMAKE_CXX_COMPILER=<your c++ compiler> \
          -DCMAKE_PREFIX_PATH=<path/to/kokkos> \
          -DCMAKE_INSTALL_PREFIX=${KOKKOSFFT_INSTALL_PREFIX}
    cmake --build build_KokkosFFT -j 8
    cmake --install build_KokkosFFT

Here is an example to use kokkos-fft in the following CMake project.

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

    find_package(Kokkos CONFIG REQUIRED)
    find_package(KokkosFFT CONFIG REQUIRED)

    add_executable(hello-kokkos-fft hello.cpp)
    target_link_libraries(hello-kokkos-fft PUBLIC Kokkos::kokkos KokkosFFT::fft)

The code can be built as

.. code-block:: bash

    cmake -B build \
          -DCMAKE_CXX_COMPILER=<your c++ compiler> \
          -DCMAKE_PREFIX_PATH="<path/to/kokkos>;<path/to/kokkos-fft>"
    cmake --build build -j 8

Install kokkos-fft with Spack
-----------------------------

kokkos-fft can also be installed with [spack](https://spack.io). For example, the recipe for H100 GPU with cufft is as follows:

.. code-block:: bash

    git clone --depth=2 --branch=v1.1.0 https://github.com/spack/spack.git
    source spack/share/spack/setup-env.sh # For bash

    spack install kokkos-fft device_backend=cufft ^kokkos +cuda +wrapper cuda_arch=90

We have two main parameters to configure Spack:

* ``host_backend``: Enable device backend FFT library (``fftw-serial`` or ``fftw-openmp``)
* ``device_backend``: Enable device backend FFT library (``cufft``, ``hipfft``, or ``onemkl``)

The code can be built as

.. code-block:: bash

    spack load kokkos kokkos-fft
    cmake -B build
    cmake --build build -j 8

CMake options
-------------

We rely on CMake to build kokkos-fft, more specifically ``CMake 3.22+``. Here is the list of CMake options. 
For FFTs on Kokkos device only, we do not need to add extra compile options but for Kokkos ones.
In order to use kokkos-fft from both host and device, it is necessary to add ``KokkosFFT_ENABLE_FFTW=ON``.
This option may be useful, for example FFT is used for initialization at host. 
However, to enable this option, we need a pre-installed ``FFTW`` for FFT on host, so it is disabled in default
if one of the device backend is enabled.
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
     - Build kokkos-fft examples
     - OFF
   * - ``KokkosFFT_ENABLE_TESTS``
     - Build kokkos-fft tests
     - OFF
   * - ``KokkosFFT_ENABLE_BENCHMARK``
     - Build benchmarks for kokkos-fft
     - OFF
   * - ``KokkosFFT_ENABLE_FFTW``
     - Use `FFTW <http://www.fftw.org>`_ for Host backend
     - ON (if none of Kokkos devices is enabled, otherwise OFF)
   * - ``KokkosFFT_ENABLE_CUFFT``
     - Use `cufft <https://developer.nvidia.com/cufft>`_ for CUDA backend
     - ON (if ``Kokkos_ENABLE_CUDA`` is ON, otherwise OFF)
   * - ``KokkosFFT_ENABLE_ROCFFT``
     - Use `rocfft <https://github.com/ROCm/rocFFT>`_ for HIP backend
     - OFF
   * - ``KokkosFFT_ENABLE_HIPFFT``
     - Use `hipfft <https://github.com/ROCm/hipFFT>`_ for HIP backend
     - ON (if ``Kokkos_ENABLE_HIP`` is ON, otherwise OFF)
   * - ``KokkosFFT_ENABLE_ONEMKL``
     - Use `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_ for SYCL backend
     - ON (if ``Kokkos_ENABLE_SYCL`` is ON, otherwise OFF)

.. note::

   ``KokkosFFT_ENABLE_HOST_AND_DEVICE`` has been deprecated since 0.3.0 and will be removed in the future.
   To enable kokkos-fft on both host and device, set ``KokkosFFT_ENABLE_FFTW=ON`` instead of setting ``KokkosFFT_ENABLE_HOST_AND_DEVICE=ON``.
   Multiple device tpls cannot be enabled at the same time. In addition, at least one tpl must be enabled to configure.
   For example, it is allowed to set ``KokkosFFT_ENABLE_CUFFT=OFF`` even if ``Kokkos_ENABLE_CUDA=ON`` as long as ``KokkosFFT_ENABLE_FFTW=ON``.

Kokkos backends
---------------

kokkos-fft requires ``Kokkos 4.6+``. For the moment, we support following backends for CPUs and GPUs.
A FFT library dedicated to Kokkos Device backend (e.g. cufft for CUDA backend) is automatically used. 
If CMake fails to find a backend FFT library, see :doc:`How to find fft libraries?<../finding_libraries>`.
We may support experimental backends like ``OPENMPTARGET`` in the future.
 
.. list-table:: ``Host backend``
   :widths: 25 50 25
   :header-rows: 1

   * - CMake option
     - Description
     - Backend FFT library
   * - ``Kokkos_ENABLE_SERIAL``
     - Serial backend targeting CPUs 
     - ``FFTW (Serial)``
   * - ``Kokkos_ENABLE_THREADS``
     - C++ threads backend targeting CPUs 
     - ``FFTW (Threads)``
   * - ``Kokkos_ENABLE_OPENMP``
     - OpenMP backend targeting CPUs 
     - ``FFTW (OpenMP)``

.. list-table:: ``Device backend``
   :widths: 25 50 25
   :header-rows: 1

   * - CMake option
     - Description
     - Backend FFT library
   * - ``Kokkos_ENABLE_CUDA``
     - CUDA backend targeting NVIDIA GPUs
     - ``cufft``
   * - ``Kokkos_ENABLE_HIP``
     - HIP backend targeting AMD GPUs
     - ``hipfft`` or ``rocfft``
   * - ``Kokkos_ENABLE_SYCL``
     - SYCL backend targeting Intel GPUs
     - ``oneMKL``
