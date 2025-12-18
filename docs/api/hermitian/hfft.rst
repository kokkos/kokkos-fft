.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::hfft
---------------

.. doxygenfunction:: KokkosFFT::hfft(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)

.. note::

   For the real input, we internally convert it to complex and perform ``hfft`` on it.
