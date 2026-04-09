.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::ifft
---------------

.. doxygenfunction:: KokkosFFT::ifft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)

Examples
========

.. literalinclude:: ../../../examples/docs/standard/docs_ifft.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (1,0) (2,0) (3,0) (4,0)
