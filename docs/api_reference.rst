.. _api_reference:

API Reference
=============

This section documents the public user interface of ``KokkosFFT``. 
APIs are defined in ``KokkosFFT`` namespace and implementation details are defined in ``KokkosFFT::Impl`` namespace. 
Thus, it is highly discouraged for users to access functions in ``KokkosFFT::Impl`` namespace except for ``Plan``. 
Except for ``KokkosFFT::Impl::Plan``, there are corresponding functions in ``numpy.fft`` as shown below.

FFT Plan
--------

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Description
     - ``KokkosFFT``
     - ``numpy.fft``
   * - A class that manages a FFT plan of backend FFT library
     - :doc:`api/plan`
     - 

Standard FFTs
-------------

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Description
     - ``KokkosFFT``
     - ``numpy.fft``
   * - One dimensional FFT in forward direction
     - :doc:`api/standard/fft`
     - `numpy.fft.fft <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html>`_
   * - One dimensional FFT in backward direction
     - :doc:`api/standard/ifft`
     - `numpy.fft.ifft <https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html>`_ 
   * - Two dimensional FFT in forward direction
     - :doc:`api/standard/fft2`
     - `numpy.fft.fft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html>`_ 
   * - Two dimensional FFT in backward direction
     - :doc:`api/standard/ifft2`
     - `numpy.fft.ifft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft2.html>`_ 
   * - N-dimensional FFT in forward direction
     - :doc:`api/standard/fftn`
     - `numpy.fft.fftn <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftn.html>`_ 
   * - N-dimensional FFT in backward direction
     - :doc:`api/standard/ifftn`
     - `numpy.fft.ifftn <https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html>`_ 

Real FFTs
---------

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Description
     - ``KokkosFFT``
     - ``numpy.fft``
   * - One dimensional FFT for real input
     - :doc:`api/real/rfft`
     - `numpy.fft.rfft <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html>`_
   * - Inverse of :doc:`rfft<api/real/rfft>`
     - :doc:`api/real/irfft`
     - `numpy.fft.irfft <https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft.html>`_ 
   * - Two dimensional FFT for real input
     - :doc:`api/real/rfft2`
     - `numpy.fft.rfft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html>`_ 
   * - Inverse of :doc:`rfft2<api/real/rfft2>`
     - :doc:`api/real/irfft2`
     - `numpy.fft.irfft2 <https://numpy.org/doc/stable/reference/generated/numpy.fft.irfft2.html>`_ 
   * - N-dimensional FFT for real input
     - :doc:`api/real/rfftn`
     - `numpy.fft.rfftn <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html>`_ 
   * - Inverse of :doc:`rfftn<api/real/rfftn>`
     - :doc:`api/real/irfftn`
     - `numpy.fft.irfftn <https://numpy.org/doc/stable/reference/generated/numpy.fft.irfftn.html>`_


Hermitian FFTs
--------------

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Description
     - ``KokkosFFT``
     - ``numpy.fft``
   * - One dimensional FFT of a signal that has Hermitian symmetry
     - :doc:`api/hermitian/hfft`
     - `numpy.fft.hfft <https://numpy.org/doc/stable/reference/generated/numpy.fft.hfft.html>`_
   * - Inverse of :doc:`hfft<api/hermitian/hfft>`
     - :doc:`api/hermitian/ihfft`
     - `numpy.fft.ihfft <https://numpy.org/doc/stable/reference/generated/numpy.fft.ihfft.html>`_

Helper routines
---------------

.. list-table::
   :widths: 50 25 25
   :header-rows: 1

   * - Description
     - ``KokkosFFT``
     - ``numpy.fft``
   * - Return the DFT sample frequencies
     - :doc:`api/helper/fftfreq`
     - `numpy.fft.fftfreq <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html>`_
   * - Return the DFT sample frequencies for Real FFTs
     - :doc:`api/helper/rfftfreq`
     - `numpy.fft.rfftfreq <https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html>`_
   * - Shift the zero-frequency component to the center of the spectrum
     - :doc:`api/helper/fftshift`
     - `numpy.fft.fftshift <https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html>`_
   * - The inverse of :doc:`fftshift<api/helper/fftshift>`
     - :doc:`api/helper/ifftshift`
     - `numpy.fft.ifftshift <https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html>`_