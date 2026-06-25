.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::Plan
---------------

.. doxygenclass:: KokkosFFT::Plan
   :members:

.. note::

   We can reuse the FFT plan created once to perform FFTs multiple times on different data given that they have the same properties. In some backend, FFT plan creation leads to some overhead, wherein we need this functionality.

Examples
========

In this example, we use the 2D View with LayoutRight to avoid the internal transpose. This allows `execute` to perform 1D FFT on the outermost dimension without transpose.

.. literalinclude:: ../../../examples/docs/plan/docs_plan_execute.cpp
  :language: c++
  :linenos:
  :lines: 5-

Expected output:

.. code::

 (10,0) (-2,2) (-2,0)
 (26,0) (-2,2) (-2,0)
 (42,0) (-2,2) (-2,0)
