.. _api_reference:

API Reference
=============

This section documents the public user interface of ``KokkosFFT``. 
APIs are defined in ``KokkosFFT`` namespace and implementation details are defined in ``KokkosFFT::Impl`` namespace. 
Thus, it is highly discouraged for users to access functions in ``KokkosFFT::Impl`` namespace except for ``Plan``.

FFT Plan
--------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - 
     - ``KokkosFFT``
     - ``numpy.fft``
   * - 
     - :doc:`KokkosFFT::Impl::Plan<api/plan>`
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