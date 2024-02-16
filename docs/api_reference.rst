.. _api_reference:

API Reference
=============

This section documents the public user interface of ``KokkosFFT``. 
APIs are defined in ``KokkosFFT`` namespace and implementation details are defined in ``KokkosFFT::Impl`` namespace. 
Thus, it is highly discouraged for users to access functions in ``KokkosFFT::Impl`` namespace except for ``Plan``.

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

.. toctree::
   :maxdepth: 1

   api/standard/fft
   api/standard/ifft
   api/standard/fft2
   api/standard/ifft2
   api/standard/fftn.rst
   api/standard/ifftn.rst

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

.. toctree::
   :maxdepth: 1

   api/real/rfft
   api/real/irfft
   api/real/rfft2.rst
   api/real/irfft2.rst
   api/real/rfftn.rst
   api/real/irfftn.rst
Hermitian FFTs
--------------

.. toctree::
   :maxdepth: 1

   api/hermitian/hfft
   api/hermitian/ihfft

Helper routines
---------------

.. toctree::
   :maxdepth: 1

   api/helper/fftfreq.rst
   api/helper/rfftfreq.rst
   api/helper/fftshift.rst
   api/helper/ifftshift.rst