.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::irfft
----------------

.. doxygenfunction:: KokkosFFT::irfft(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)
.. doxygenfunction:: KokkosFFT::irfft(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, const PlanType& plan, KokkosFFT::Normalization, int axis, std::optional<std::size_t> n)