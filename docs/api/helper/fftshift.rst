.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::fftshift
-------------------

.. doxygenfunction:: KokkosFFT::fftshift(const ExecutionSpace& exec_space, const ViewType& inout, std::optional<int> axes = std::nullopt)
.. doxygenfunction:: KokkosFFT::fftshift(const ExecutionSpace& exec_space, const ViewType& inout, axis_type<DIM> axes)

Examples
========

.. literalinclude:: ../../../examples/docs/helper/docs_fftshift.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 -5 -4 -3 -2 -1 0 1 2 3 4
