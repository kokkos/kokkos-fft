.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

KokkosFFT::Plan
---------------

.. doxygenclass:: KokkosFFT::Plan
   :members:

.. note::

   We can reuse the FFT plan created once to perform FFTs multiple times on different data given that they have the same properties. In some backend, FFT plan creation leads to some overhead, wherein we need this functionality.

.. warning::

   For the FFTW backend (Serial, OpenMP and Threads execution spaces), reusing a plan additionally requires the Views passed to ``execute`` to be at least as aligned as the Views used to create the plan: FFTW plans embed the SIMD alignment of the arrays used at plan creation, and executing a plan created on aligned Views on less-aligned Views is undefined behaviour (it can crash). KokkosFFT checks this at execution time and throws a ``std::runtime_error`` instead. Views freshly allocated by Kokkos are always sufficiently aligned; only Views overlapping the end of a larger allocation (e.g. a shifted 1D subview) can be misaligned. Executing a plan created on misaligned Views on aligned Views is safe and is not rejected.

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
