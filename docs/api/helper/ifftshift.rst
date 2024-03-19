.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::ifftshift
--------------------

.. doxygenfunction:: KokkosFFT::ifftshift(const ExecutionSpace& exec_space, ViewType& inout, std::optional<int> axes = std::nullopt)
.. doxygenfunction:: KokkosFFT::ifftshift(const ExecutionSpace& exec_space, ViewType& inout, axis_type<DIM> axes)

.. note::

   For the moment, this function works on one or two dimensional Views.