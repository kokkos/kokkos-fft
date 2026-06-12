.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::hfft
---------------

.. doxygenfunction:: KokkosFFT::hfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)

.. note::

   The input can be either a real-valued or complex-valued view, and the output must be a real-valued view. The output length along the transform axis is ``(n / 2 + 1) * 2``, where ``n`` is the input length along that axis. If this condition is not met, the `std::runtime_error` exception will be thrown.
   For the real input, we internally convert it to complex and perform ``hfft`` on it.

Examples
========

.. literalinclude:: ../../../examples/docs/hermitian/docs_hfft.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 15 -4 0 -1 0 -4
