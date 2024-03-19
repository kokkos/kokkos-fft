.. SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

.. _using:

Using Kokkos-fft
================

This section describes how to use Kokkos-fft in practice. 
We also explain some tips to use it efficiently.

Brief introduction
------------------

Most of the numpy.fft APIs (``numpy.fft.<function_name>``) are available in Kokkos-fft (``KokkosFFT::<function_name>``) on the Kokkos device.
In fact, these are the only APIs available in Kokkos-fft (see :doc:`API reference<../api_reference>` for detail). Kokkos-fft support 1D to 3D FFT over chosen axes.
Inside FFT APIs, we first create a FFT plan for a backend FFT library based on the Views and chosen axes.
Then, we execute the FFT using the created plan on the given Views. Then, we may perform normalization based on the users' choice. 
Finally, we destroy the plan. Depending on the View Layout and chosen axes, we may need transpose operations to make data contiguous.
In that case, we perform the transpose operations internally which impose overheads in both memory and computations.

.. note::

   ``KokkosFFT::Impl`` namespace represents implementation details and should not be accessed by users.

Basic Instruction
-----------------

We have Standard and Real FFTs as APIs. Standard FFTs can be used for complex to complex transform, whereas
Real FFTs perform real to complex transform. As well as ``numpy.fft``, numbers after ``fft`` represents the dimension of FFT.
For example, ``KokkosFFT::fft2`` performs 2D (potentially batched) FFT in forward direction.
If the rank of Views is higher than the dimension of FFT, a batched FFT plan is created.
APIs start from ``i`` represent inverse transforms.  
For Real FFTs, users have to pay attention to the input and output data types as well as their extents.
Inconsistent data types are suppressed by compilation errors. If extents are inconsistent, 
it will raise runtime errors (C++ exceptions or assertions).
The following listing shows good and bad examples of Real FFTs.

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
   constexpr int n0 = 4, n1 = 8;

   View2D<double> x("x", n0, n1);
   View2D<Kokkos::complex<double> > x_hat_good("x_hat_good", n0, n1/2+1);
   View2D<Kokkos::complex<double> > x_hat_bad("x_hat_bad", n0, n1);

   // [OK] Correct types and extents
   KokkosFFT::rfft2(execution_space(), x, x_hat_good);

   // [NG, Compile time error] Inconsistent types
   // Input: double (NG) -> complex<double> (OK)
   // Output: complex<double> (NG) -> double (OK)
   //KokkosFFT::irfft2(execution_space(), x, x_hat_good);
   // [OK] Correct types and extents
   KokkosFFT::irfft2(execution_space(), x_hat_good, x);

   // [NG, Run time error] Inconsistent extents
   // Output: (n0, n1) (NG) -> (n0, n1/2+1) (OK)
   KokkosFFT::rfft2(execution_space(), x, x_hat_bad);


.. note::

   Input and ouptut views must have the same precision (either ``float`` or ``double``).

Supported data types
--------------------

Firstly, the input and output Views must have the same LayoutType and rank.
For the moment, we accept Kokkos Views with some restriction in data types and Layout.
Here are the list of available types for Views. We recommend to use dynamic allocation for Views,
since we have not tested with static shaped Views. In addition, we have not tested with non-default 
`Memory Traits <https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html#memory-traits>`_.

* DataType: ``float``, ``double``, ``Kokkos::complex<float>``, ``Kokkos::complex<double>``
* LayoutType: ``Kokkos::LayoutLeft``, ``Kokkos::LayoutRight``
* MemorySpace: ``Kokkos::DefaultExecutionSpace::memory_space``, ``Kokkos::DefaultHostExecutionSpace::memory_space`` (available if targeting CPU or ``KokkosFFT_ENABLE_HOST_AND_DEVICE`` is enabled.)

.. note::

   For the moment, ``Kokkos::LayoutStride`` is not allowed. This may be relaxed in the future.

Normalization
-------------

After the transform, normalization can be applied by setting the ``norm`` argument. We have four options:

* ``KokkosFFT::Normalization::forward``: :math:`1/n` scaling for forward transform
* ``KokkosFFT::Normalization::backward``: :math:`1/n` scaling for backward transform (default)
* ``KokkosFFT::Normalization::ortho``: :math:`1/\sqrt{n}` scaling for both forward and backward transform
* ``KokkosFFT::Normalization::none``: No scaling

For users who already have own normalization functions, please specify ``none`` option.

Memory consmpution
------------------

In order to support FFT over arbitral axes, 
Kokkos-fft performs transpose operations internally and apply FFT on contiguous data.
For size ``n`` input, this requires internal buffers of size ``2n`` in addition to the buffers used by FFT library. 
Performance overhead from transpose may be not critical but memory consumptions are problematic. 
If memory consumption matters, it is recommended to make data contiguous so that transpose is not performed. 
The following listing shows examples with and without transpose operation.

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
   constexpr int n0 = 4, n1 = 8;

   View2D<double> x("x", n0, n1);
   View2D<Kokkos::complex<double> > x_hat_good("x_hat_good", n0/2+1, n1);
   View2D<Kokkos::complex<double> > x_hat_bad("x_hat_bad", n0, n1/2+1);

   // Transpose NOT needed: equivalent to np.fft(np.rfft(axis=0), axis=1)
   KokkosFFT::rfft2(execution_space(), x, x_hat_good, /*axes=*/{-1, -2});

   // Transpose needed: equivalent to np.fft(np.rfft(axis=1), axis=0)
   KokkosFFT::rfft2(execution_space(), x, x_hat_bad, /*axes=*/{-2, -1});

Reuse FFT plan
--------------

Apart from the basic APIs, Kokkos-fft APIs include overloaded APIs which can take a FFT plan as an argument.
Using these overloaded APIs, we can reuse the FFT plan created before. 
In some backend, FFT plan creation leads to some overhead, wherein we need this functionality.
(see :doc:`minimum working example<../samples/06_1DFFT_reuse_plans>`)

.. note::

   Input and Output Views used to call FFT APIs must have the same types and extents as the ones used for plan creation.
