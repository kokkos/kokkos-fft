.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::irfft2
-----------------
.. doxygenfunction:: KokkosFFT::irfft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, axis_type<2> axes, shape_type<2> s)


.. note::

   The input must be a complex-valued view, and the output must be a real-valued view. The input length along the transform axis (``axes[1]``) is ``n/2 + 1``, where ``n`` is the output length along that axis. If this condition is not met, the `std::runtime_error` exception will be thrown.

Examples
========

In this example, we use the 2D View with `LayoutRight` to avoid the internal transpose. 
This allows `irfft2` to perform 2D FFT on the outermost dimension without transpose.

.. literalinclude:: ../../../examples/docs/real/docs_irfft2.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 1 2 3 4
 5 6 7 8
 9 10 11 12
