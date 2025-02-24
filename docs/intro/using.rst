.. SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
..
.. SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

.. _using:

Using kokkos-fft
================

This section describes how to use kokkos-fft in practice. 
We also explain some tips to use it efficiently.

Brief introduction
------------------

Most of the numpy.fft APIs (``numpy.fft.<function_name>``) are available in kokkos-fft (``KokkosFFT::<function_name>``) on the Kokkos device.
In fact, these are the only APIs available in kokkos-fft (see :doc:`API reference<../api_reference>` for detail). kokkos-fft support 1D to 3D FFT over chosen axes.
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
it will raise runtime errors (C++ ``std::runtime_error``).
The following listing shows good and bad examples of Real FFTs.

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
   const int n0 = 4, n1 = 8;

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

   Input and output views must have the same precision (either ``float`` or ``double``).

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
kokkos-fft performs transpose operations internally and apply FFT on contiguous data.
For size ``n`` input, this requires internal buffers of size ``2n`` in addition to the buffers used by FFT library. 
Performance overhead from transpose may be not critical but memory consumptions are problematic. 
If memory consumption matters, it is recommended to make data contiguous so that transpose is not performed. 
The following listing shows examples with and without transpose operation.

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
   const int n0 = 4, n1 = 8;

   View2D<double> x("x", n0, n1);
   View2D<Kokkos::complex<double> > x_hat_good("x_hat_good", n0/2+1, n1);
   View2D<Kokkos::complex<double> > x_hat_bad("x_hat_bad", n0, n1/2+1);

   // Transpose NOT needed: equivalent to np.fft(np.rfft(axis=0), axis=1)
   KokkosFFT::rfft2(execution_space(), x, x_hat_good, /*axes=*/{-1, -2});

   // Transpose needed: equivalent to np.fft(np.rfft(axis=1), axis=0)
   KokkosFFT::rfft2(execution_space(), x, x_hat_bad, /*axes=*/{-2, -1});

Reuse FFT plan
--------------

Apart from the basic APIs, kokkos-fft offers the capability to create a FFT plan wrapping the FFT plans of backend libraries.
We can reuse the FFT plan created once to perform FFTs multiple times on different data given that they have the same properties.
In some backend, FFT plan creation leads to some overhead, wherein we need this functionality (see :doc:`minimum working example<../samples/06_1DFFT_reuse_plans>`).
The following listing shows an example to reuse the FFT plan.

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
   const int n0 = 4, n1 = 8, n2 = 5, n3 = 10;

   View2D<Kokkos::complex<double> > x("x", n0, n1), x_hat("x_hat", n0, n1);
   View2D<Kokkos::complex<double> > y("y", n0, n1), y_hat("y_hat", n0, n1);
   View2D<Kokkos::complex<double> > z("z", n2, n3), z_hat("z_hat", n2, n3);

   // Create a plan for 1D FFT
   int axis = -1;
   KokkosFFT::Plan fft_plan(execution_space(), x, x_hat,
                            KokkosFFT::Direction::forward, axis);
   
   // Perform FFTs using fft_plan
   KokkosFFT::execute(fft_plan, x, x_hat);

   // [OK] Reuse the plan for different data
   KokkosFFT::execute(fft_plan, y, y_hat);

   // [NG, Run time error] Inconsistent extents
   KokkosFFT::execute(fft_plan, z, z_hat);

.. note::

   Input and Output Views used to call FFT APIs must have the same types and extents as the ones used for plan creation.

Axes parameters
---------------

As well as ``numpy.fft``, you can specify negative axes to perform FFT over chosen axes, which is not common in C++.
Actually for FFT APIs, default axes are set as ``{-DIM, -(DIM-1), ...}`` where ``DIM`` is the rank of the FFT dimensions,
corresponding to the FFTs over last ``DIM`` axes. If we consider that default View layout is C layout (row-major or ``Kokkos::LayoutRight``), 
this default axes parameter results in FFTs performed over the contiguous dimensions. For example, ``KokkosFFT::fft2(execution_space(), in, out)`` is equivalent to ``KokkosFFT::fft2(execution_space(), in, out, axis_type<2>({-2, -1}))``.
Negative axes are counted from the last axis, which is the same as ``numpy.fft``.
For example, ``-1`` means the last axis, ``-2`` means the second last axis, and so on.
Negative axes ``-1`` and ``-2`` respectively correspond to ``rank-1`` and ``rank-2``, where the ``rank`` is the rank of the Views.

The following listing shows examples of axes parameters with negative or positive values.

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutLeft, execution_space>;
   template <typename T> using View3D = Kokkos::View<T***, Kokkos::LayoutLeft, execution_space>;
   const int n0 = 4, n1 = 8, n2 = 5;

   View2D<double> x2("x2", n0, n1);
   View3D<double> x3("x3", n0, n1, n2);
   View2D<Kokkos::complex<double> > x2_hat("x2_hat", n0/2+1, n1);
   View3D<Kokkos::complex<double> > x3_hat("x3_hat", n0, n1/2+1, n2);

   // Following codes are all equivalent to np.fft(np.rfft(x2, axis=0), axis=1)
   // negative axes are converted as follows:
   // -2 -> 0 (= Rank(2) - 2), -1 -> 1 (= Rank(2) - 1)
   KokkosFFT::rfft2(execution_space(), x2, x2_hat, /*axes=*/{-1, -2});
   KokkosFFT::rfft2(execution_space(), x2, x2_hat, /*axes=*/{-1, 0});
   KokkosFFT::rfft2(execution_space(), x2, x2_hat, /*axes=*/{1, -2});
   KokkosFFT::rfft2(execution_space(), x2, x2_hat, /*axes=*/{1, 0});

   // Following codes are all equivalent to np.fft(np.rfft(x3, axis=1), axis=2)
   // negative axes are converted as follows:
   // -2 -> 1 (= Rank(3) - 2), -1 -> 2 (= Rank(3) - 1)
   KokkosFFT::rfft2(execution_space(), x3, x3_hat, /*axes=*/{-1, -2});
   KokkosFFT::rfft2(execution_space(), x3, x3_hat, /*axes=*/{-1, 1});
   KokkosFFT::rfft2(execution_space(), x3, x3_hat, /*axes=*/{2, -2});
   KokkosFFT::rfft2(execution_space(), x3, x3_hat, /*axes=*/{2, 1});

.. note::

   If you rely on negative axes, you can specify last axes no matter what the rank of Views is.
   However, the corresponding positive axes to last axes are different depending on the rank of Views.
   Thus, it is recommended to use negative axes for simplicity.

Inplace transform
-----------------

Inplace transform is supported in kokkos-fft in case transpose or reshape is not needed. 
For standard FFTs, we can just use the same input and output Views. For real FFTs, we need to use a single complex View and make 
an unmanaged View which is an alias to the complex View. In addition, we need to pay attention to the extents of a real View,
which should define the shape of the transform, not the reinterpreted shape of the complex View (see :doc:`minimum working example<../samples/08_inplace_FFT>`). 
The following listing shows examples of inplace transforms. 

.. code-block:: C++

   template <typename T> using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
   const int n0 = 4, n1 = 8;
   View2D<Kokkos::complex<double>> xc2c("xc2c", n0, n1);

   execution_space exec;

   // For standard inplace FFTs, we just reuse the same views
   KokkosFFT::fft2(exec, xc2c, xc2c);
   KokkosFFT::ifft2(exec, xc2c, xc2c);

   // Real to complex transform
   // Define a 2D complex view to handle data
   View2D<Kokkos::complex<double>> xr2c_hat("xr2c", n0, n1 / 2 + 1);

   // Create unmanaged views on the same data with the FFT shape,
   // that is (n0, n1) -> (n0, n1/2+1) R2C transform
   // The shape is incorrect from the view point of casting to real
   // For casting, the shape should be (n0, (n1/2+1) * 2)
   View2D<double> xr2c(reinterpret_cast<double *>(xr2c_hat.data()), n0, n1);

   // Perform the real to complex transform
   // [Important] You must use xr2c to define the FFT shape correctly
   KokkosFFT::rfft2(exec, xr2c, xr2c_hat);

   // Complex to real transform
   // Define a 2D complex view to handle data
   View2D<Kokkos::complex<double>> xc2r("xc2r", n0, n1 / 2 + 1);

   // Create an unmanaged view on the same data with the FFT shape
   View2D<double> xc2r_hat(reinterpret_cast<double *>(xc2r.data()), n0, n1);

   // Create a plan
   using axes_type = KokkosFFT::axis_type<2>;
   axes_type axes  = {-2, -1};
   KokkosFFT::Plan irfft2_plan(execution_space(), xc2r, xc2r_hat,
                               KokkosFFT::Direction::backward, axes);
   
   // Perform the complex to real transform
   // [Important] You must use xc2r_hat to define the FFT shape correctly
   KokkosFFT::execute(irfft2_plan, xc2r, xc2r_hat);

   View2D<double> xc2r_hat_out("xc2r_hat_out", n0, n1);

   // [NG, Runtime error] Inplace plan can only be reused for inplace transform
   KokkosFFT::execute(irfft2_plan, xc2r, xc2r_hat_out);

.. note::

   You can reuse a plan for inplace transform. However, you cannot reuse a plan
   for inplace transform for out-of-place transform and vice versa.
