.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::rfftn
----------------

.. doxygenfunction:: KokkosFFT::rfftn(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, axis_type<DIM> axes, KokkosFFT::Normalization, shape_type<DIM> s)

.. note::

   The input must be a real-valued view, and the output must be a complex-valued view. The output length along the transform axis (``axes[DIM-1]``) is ``n/2 + 1``, where ``n`` is the input length along that axis. If this condition is not met, the `std::runtime_error` exception will be thrown.

Examples
========

In this example, we use the 3D View with `LayoutRight` to avoid the internal transpose. 
This allows `rfftn` to perform 3-dimensional FFT on the outermost dimension without transpose.

.. literalinclude:: ../../../examples/docs/real/docs_rfftn.cpp
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
