.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::rfft
---------------

.. doxygenfunction:: KokkosFFT::rfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)

.. note::

   The input must be a real-valued view, and the output must be a complex-valued view. The output size along the transform axis is ``n/2 + 1``, where ``n`` is the input size along that axis.

Examples
========

.. literalinclude:: ../../../examples/docs/real/docs_rfft.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (10,0) (-2,2) (-2,0)