.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::ifft2
----------------

.. doxygenfunction:: KokkosFFT::ifft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, axis_type<2> axes, shape_type<2> s)

Examples
========

In this example, we use the 2D View with `LayoutRight` to avoid the internal transpose. 
This allows `ifft2` to perform 2D FFT on the outermost dimension without transpose.

.. literalinclude:: ../../../examples/docs/docs_ifft2.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (1,0) (2,0) (3,0)
 (4,0) (5,0) (6,0)
 (7,0) (8,0) (9,0)
 (10,0) (11,0) (12,0)
