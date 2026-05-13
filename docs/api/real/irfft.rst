.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::irfft
----------------

.. doxygenfunction:: KokkosFFT::irfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)

.. note::

   The input must be a complex-valued view, and the output must be a real-valued view. The input length along the transform axis is ``n/2 + 1``, where ``n`` is the output length along that axis. If this condition is not met, the `std::runtime_error` exception will be thrown.

Examples
========

.. literalinclude:: ../../../examples/docs/real/docs_irfft.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

  1 2 3 4
