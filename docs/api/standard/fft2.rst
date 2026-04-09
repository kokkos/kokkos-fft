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

In this example, we use the 2D View with `LayoutRight` to avoid the internal transpose. 
This allows `fft2` to perform 2D FFT on the outermost dimension without transpose.

.. literalinclude:: ../../../examples/docs/standard/docs_fft2.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (78,0) (-6,3.4641) (-6,-3.4641)
 (-18,18) (0,0) (0,0)
 (-18,0) (0,0) (0,0)
 (-18,-18) (0,0) (0,0)
