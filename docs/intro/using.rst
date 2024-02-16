.. _using:

Using KokkosFFT
===============

This section describes how to use KokkosFFT in practice. 
We also explain some tips to use it efficiently.

Brief introduction
------------------

Most of the numpy.fft APIs (``numpy.fft.<function_name>``) are available in KokkosFFT (``KokkosFFT::<function_name>``) on the Kokkos device.
In fact, these are the only APIs available in KokkosFFT (see :doc:`API reference<../api_reference>` for detail). KokkosFFT support 1D to 3D FFT over choosen axes. 
Inside FFT APIs, we first create a FFT plan for backend FFT library based on the Views and choosen axes.
Then, we execute the FFT using the created plan on the given Views. Finally, we destroy the plan.
Depending on the View Layout and choosen axes, we may need transpose operations to make data contiguous.
In that case, we perform the transpose operations internally which impose overheads in both memory and computations.

.. note::

   ``KokkosFFT::Impl`` namespace is for implementation details and should not be accessed by users.

Basic Instruction
-----------------

We have Standard and Real FFTs as APIs. Standard FFTs can be used for complex to complex transform, whereas
Real FFTs perform real to complex transform. As well as ``numpy.fft``, numbers after ``fft`` represents the dimension of FFT.
For example, ``KokkosFFT::fft2`` performs 2D (potentially batched) FFT in forward direction.
If the rank of Views is higher than the dimension of FFT, a batched FFT plan is created.
APIs start from ``i`` are for inverse transform.  
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
   //KokkosFFT::irfft2(execution_space(), x_hat_good, x);

   // [NG, Run time error] Inconsistent extetns
   // Output: (n0, n1) (NG) -> (n0, n1/2+1) (OK)
   KokkosFFT::rfft2(execution_space(), x, x_hat_bad);


.. note::

   We have to use the same precision (either ``float`` or ``double``) for input and ouptut Views.

Supported Views
---------------

Firstly, the input and output Views must have the same LayoutType and rank.
For the moment, we accept Kokkos Views with some restriction in data types and Layout.
Here are the list of available types for Views. We recommend to use dynamic allocation for Views,
since we have not tested with static shaped Views. In addition, we have not tested with non-default `MemoryTraits`. 

* DataType: ``float``, ``double``, ``Kokkos::complex<float>``, ``Kokkos::complex<double>``
* LayoutType: ``Kokkos::LayoutLeft``, ``Kokkos::LayoutRight``
* MemorySpace: ``Kokkos::DefaultExecutionSpace::memory_space``, ``Kokkos::DefaultHostExecutionSpace::memory_space`` (available if targeting CPU or ``KokkosFFT_ENABLE_HOST_AND_DEVICE`` is enabled.)

.. note::

   For the moment, ``Kokkos::LayoutStride`` is not allowed. This may be relaxed in the future.

Memory consmpution
------------------

In order to support FFT over arbitral axes, 
KokkosFFT performs transpose operations internally and apply FFT on contious data.
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

Apart from the basic APIs, KokkosFFT APIs include overloaded APIs which can take a FFT plan as an argument.
Using these overloaded APIs, we can reuse the FFT plan created before. 
In some backend, FFT plan creation leads to some overhead, wherein we need this functionality.

.. note::

   Input and Output Views used to call FFT APIs must have the same types and extents as the ones used for plan creation.