.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

.. _finding_libraries:

Finding FFT libraries by CMake
==============================

Some tips to find FFT libraries for each backend. 

`FFTW <http://www.fftw.org>`_
-----------------------------

If ``FFTW`` is offered as a module, our CMake helper would likely find ``FFTW``.
Assuming ``FFTW`` is installed in ``<path/to/FFTW>``, it is expected that ``<path/to/FFTW>`` would be found under ``LIBRARY_PATH``, ``LD_LIBRARY_PATH``, and ``PATH``.
It would look like

.. code-block:: bash

    LIBRARY_PATH=...:<path/to/FFTW>/lib
    LD_LIBRARY_PATH=...:<path/to/FFTW>/lib
    PATH=...:<path/to/FFTW>/bin

If CMake fails to find ``FFTW``, please try to set ``FFTWDIR`` in the following way.

.. code-block:: bash

    export FFTWDIR=<path/to/FFTW>

`cufft <https://developer.nvidia.com/cufft>`_
---------------------------------------------

`hipfft <https://github.com/ROCm/hipFFT>`_
------------------------------------------

`rocfft <https://github.com/ROCm/rocFFT>`_
------------------------------------------

`oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_
---------------------------------------------------------------------------------------

The most likely scenario to miss ``oneMKL`` is that forgetting to initialize ``oneAPI``.
Please make sure to initialize ``oneAPI`` as

.. code-block:: bash

    <path/to/oneapi>/setvars.sh --include-intel-llvm
