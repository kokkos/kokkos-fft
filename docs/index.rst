.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

kokkos-fft documentation
=======================================

kokkos-fft implements local interfaces between `Kokkos <https://kokkos.org>`_ 
and de facto standard FFT libraries, 
including `fftw <http://www.fftw.org>`_,
`cufft <https://developer.nvidia.com/cufft>`_,
`hipfft <https://github.com/ROCm/hipFFT>`_ (`rocfft <https://github.com/ROCm/rocFFT>`_), and `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`_. 
"Local" means not using MPI, or running within a single MPI process without knowing about MPI.
We are inclined to implement the `numpy.fft <https://numpy.org/doc/stable/reference/routines.fft.html>`_-like interfaces adapted for Kokkos.
A key concept is that *"As easy as numpy, as fast as vendor libraries"*. Accordingly, our API follows the API by ``numpy.fft`` with minor differences. 
A FFT library dedicated to Kokkos Device backend (e.g. cufft for CUDA backend) is automatically used. 

kokkos-fft is open source and available on `GitHub <https://github.com/kokkos/kokkos-fft>`_.

Here is an example for 1D real to complex transform with ``rfft`` in kokkos-fft.

.. code-block:: C++

   #include <Kokkos_Core.hpp>
   #include <Kokkos_Complex.hpp>
   #include <Kokkos_Random.hpp>
   #include <KokkosFFT.hpp>
   using execution_space = Kokkos::DefaultExecutionSpace;
   template <typename T> using View1D = Kokkos::View<T*, execution_space>;
   const int n = 4;

   View1D<double> x("x", n);
   View1D<Kokkos::complex<double> > x_hat("x_hat", n/2+1);

   Kokkos::Random_XorShift64_Pool<> random_pool(12345);
   Kokkos::fill_random(x, random_pool, 1);
   Kokkos::fence();

   KokkosFFT::rfft(execution_space(), x, x_hat);

This is equivalent to the following python script.

.. code-block:: python

   import numpy as np
   x = np.random.rand(4)
   x_hat = np.fft.rfft(x)

.. note::

   It is assumed that backend FFT libraries are appropriately installed on the system.


.. toctree::
   :maxdepth: 1

   getting_started
   finding_libraries
   api_reference
   examples
   developer_guide

..
    Indices and tables
    ==================

    * :ref:`genindex`
    * :ref:`modindex`
    * :ref:`search`