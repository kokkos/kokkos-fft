.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

.. _examples:

Examples
========

There are some `examples
<https://github.com/kokkos/kokkos-fft/tree/main/examples>`_ in the
kokkos-fft repository. Most of the examples include Kokkos and numpy implementations.
For example, `01_1DFFT
<https://github.com/kokkos/kokkos-fft/tree/main/examples/01_1DFFT>`_ includes,

.. code-block:: bash

    ---/
     |
     └──01_1DFFT/
        |--CMakeLists.txt
        |--01_1DFFT.cpp (kokkos-fft version)
        └──numpy_1DFFT.py (numpy version)

Please find the examples from following links.

.. toctree::
   :maxdepth: 1

   samples/01_1DFFT.rst
   samples/02_2DFFT.rst
   samples/03_NDFFT.rst
   samples/04_batchedFFT.rst
   samples/05_1DFFT_HOST_DEVICE.rst
   samples/06_1DFFT_reuse_plans.rst
   samples/07_unmanaged_views.rst
   samples/08_inplace_FFT.rst
   samples/09_derivative.rst
   samples/10_HasegawaWakatani.rst
   