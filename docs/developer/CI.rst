.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

CI (Continuous Integration)
===========================

Our CI system is designed to automate testing and ensure that every change meets our coding styles. 
If you are familiar with github actions, you may find our workflow `here <https://github.com/kokkos/kokkos-fft/blob/main/.github/workflows/build_test.yaml>`_.
The CI process includes:

- **Linting and Style Checks:** Verifying that the code follows to our style guidelines.
- **Build Verification:** Compiling and installing the project in various environments to ensure compatibility.
- **Unit tests:** Running unit tests on CPUs (on github Azure) and NVIDIA GPUs (on our local server).

Linting and Style Checks
------------------------

We have four CIs for formatting: ``reuse``, ``clang-format``, ``cmake-format`` and ``typos``. 

#. **License Check with Reuse:**  
   All the files in the repo need to include copyright and license information at the top.
   These are automatically confirmed with `REUSE compliance check <https://reuse.software>`_ in CI.
   If there is a file without the copyright, the REUSE CI will fail and notify which file misses a copyright.
   The copyright statement in ``CMakeLists.txt`` is given in the following manner.

   .. code-block:: CMake

     # SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md files
     #
     # SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
     
     cmake_minimum_required(VERSION 3.22)

#. **Code formatting with clang-format:**  
   All the source codes (``.cpp`` and ``.hpp`` files) in the repo shall be formatted with clang-format 17.
   To format a C++ file, please apply the following command

   .. code-block:: bash

      clang-format -i <cppfile_you_have_modified>

   Format are automatically confirmed with `clang-format check <https://clang.llvm.org/docs/ClangFormat.html>`_ in CI.
   If further formatting is needed, the clang-format CI will fail and notify which file needs to be modified.

#. **CMake formatting with cmake-format:**  
   All the CMake files (``CMakeLists.txt`` and ``.cmake`` files) in the repo shall be formatted with cmake-format.
   To format a CMake file, please apply the following command

   .. code-block:: bash

      cmake-format --in-place

   Format are automatically confirmed with `cmake-format check <https://github.com/cheshirekow/cmake_format>`_ in CI.
   If further formatting is needed, the cmake-format CI will fail and notify which file needs to be modified.

#. **Spell check with typos:**  
   Spell errors are checked with `typos <https://github.com/crate-ci/typos>`_ in CI.
   If potential typos are detected, they will report typos with suggestions.

Build Verification
------------------

Compilation tests are performed inside containers for each backend including NVIDIA, AMD and Intel GPUs
(see `Dockerfiles <https://github.com/kokkos/kokkos-fft/tree/main/docker>`_).
These images are useful to develop locally particularly when you are interested in modifying the
backend specific codes. In other word, if you develop and test your code inside these containers, 
your PR will likely to pass our CI. For each backend, we test to compile a simple test code by using kokkos-fft as CMake subdirectory or installed library. 

Unit tests
----------

We rely on `googletest <https://github.com/google/googletest>`_ for unit-testing. For CPU backends, we perform unit-tests on github Azure. 
We also perform unit-tests on NVIDIA GPUs in our local server, which requires a special approval from maintaniers.
If you change the backend specific source codes for HIP or SYCL backends, please make sure that all the unit-tests pass.
For Intel GPUs (Intel PVC), you can test locally in the following way.

   .. code-block:: bash

      cmake -B build \
      -DCMAKE_CXX_COMPILER=icpx \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DKokkos_ENABLE_SYCL=ON \
      -DKokkos_ARCH_INTEL_PVC=ON \
      -DKokkosFFT_ENABLE_INTERNAL_KOKKOS=ON \
      -DKokkosFFT_ENABLE_TESTS=ON \
      -DKokkosFFT_ENABLE_EXAMPLES=ON
      
      cmake --build build -j 8
      cd build && ctest --output-on-failure

Here is the summary of our compile and run tests for each backend. For GPU backends, we compile with and without ``KokkosFFT_ENABLE_HOST_AND_DEVICE`` option (see :doc:`CMake options<../intro/building>`).

.. list-table:: Test summary
   :widths: 15 15 15 15 15 15
   :header-rows: 1

   * - build name
     - Compiler, C++ standard
     - Kokkos backend
     - Description
     - Build/install test
     - Run test
   * - clang-tidy
     - clang, 17
     - ``Kokkos_ENABLE_SERIAL``
     - clang-tidy check
     - x (Aazure)
     - None
   * - serial
     - gcc, 17
     - ``Kokkos_ENABLE_SERIAL``
     -
     - x (Aazure)
     - x (Aazure)
   * - threads
     - gcc, 20
     - ``Kokkos_ENABLE_THREADS``
     -
     - x (Aazure)
     - x (Aazure)
   * - openmp
     - gcc, 17
     - ``Kokkos_ENABLE_OPENMP``
     - Debug mode
     - x (Aazure)
     - x (Aazure)
   * - cuda
     - gcc, 17
     - ``Kokkos_ENABLE_CUDA``
     -
     - x (Aazure)
     - x (self-hosted)
   * - hip
     - hipcc, 17
     - ``Kokkos_ENABLE_HIP``
     - ``hipfft`` backend
     - x (Aazure)
     - None
   * - rocm
     - hipcc, 20
     - ``Kokkos_ENABLE_HIP``
     - ``rocfft`` backend
     - x (Aazure)
     - None
   * - sycl
     - icpx, 17
     - ``Kokkos_ENABLE_SYCL``
     -
     - x (Aazure)
     - None
