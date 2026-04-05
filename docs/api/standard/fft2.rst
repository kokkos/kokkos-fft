.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::fft2
---------------

.. doxygenfunction:: KokkosFFT::fft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, axis_type<2> axes, shape_type<2> s)

.. note::

   For the real input, we internally convert it to complex and perform ``fft2`` on it.

Examples
========

.. literalinclude:: ../../../examples/docs/docs_fft2.cpp
  :language: c++
  :linenos:
  :lines: 5-
