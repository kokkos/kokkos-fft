.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::fftn
---------------

.. doxygenfunction:: KokkosFFT::fftn(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, axis_type<DIM> axes, KokkosFFT::Normalization norm, shape_type<DIM> s)

.. note::

   For the real input, we internally convert it to complex and perform ``fftn`` on it.

Examples
========

In this example, we use the 3D View with `LayoutRight` to avoid the internal transpose. 
This allows `fftn` to perform 3D FFT on the outermost dimension without transpose.

.. literalinclude:: ../../../examples/docs/docs_fftn.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (78,0) (-6,0)
 (-12,0) (0,0)

 (-24,13.8564) (0,0)
 (0,0) (0,0)

 (-24,-13.8564) (0,0)
 (0,0) (0,0)
