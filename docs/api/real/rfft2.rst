.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::rfft2
----------------

.. doxygenfunction:: KokkosFFT::rfft2(const ExecutionSpace& exec_space, const InViewType& in, const OutViewType& out, KokkosFFT::Normalization, axis_type<2> axes, shape_type<2> s)

.. note::

   The input must be a real-valued view, and the output must be a complex-valued view. The output length along the transform axis (``axes[1]``) is ``n/2 + 1``, where ``n`` is the input length along that axis. If this condition is not met, the `std::runtime_error` exception will be thrown.

Examples
========

.. literalinclude:: ../../../examples/docs/real/docs_rfft2.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (78,0) (-6,6) (-6,0)
 (-24,13.8564) (0,0) (0,0)
 (-24,-13.8564) (0,0) (0,0)
