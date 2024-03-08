.. _examples:

Examples
========

There are some `examples
<https://github.com/CExA-project/kokkos-fft/tree/main/examples>`_ in the
KokkosFFT repository. Each example includes Kokkos and numpy implementations.
For example, `01_1DFFT
<https://github.com/CExA-project/kokkos-fft/tree/main/examples/01_1DFFT>`_ includes,

.. code-block:: bash

    ---/
     |
     └──01_1DFFT/
        |--CMakeLists.txt
        |--01_1DFFT.cpp (KokkosFFT version)
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