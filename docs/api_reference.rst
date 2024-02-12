
API Reference
=============

FFT Plan
--------

.. toctree::
   :maxdepth: 1

   Plan (Wrapping a FFT plan of backend FFT library) <api/plan>

Standard FFTs
-------------

.. toctree::
   :maxdepth: 1

   fft (One dimensional FFT in forward direction) <api/standard/fft.rst>
   ifft (One dimensional FFT in backward direction) <api/standard/ifft.rst>
   fft2 (Two dimensional FFT in forward direction) <api/standard/fft2.rst>
   ifft2 (Two dimensional FFT in backward direction) <api/standard/ifft2.rst>
   fftn (N-dimensional FFT in forward direction) <api/standard/fftn.rst>
   ifftn (N-dimensional FFT in backward direction) <api/standard/ifftn.rst>

Real FFTs
---------

.. toctree::
   :maxdepth: 1

   rfft (One dimensional FFT for real input) <api/real/rfft.rst>
   irfft (Inverse of rfft) <api/real/irfft.rst>
   rfft2 (Two dimensional FFT for real input) <api/real/rfft2.rst>
   irfft2 (Inverse of rfft2) <api/real/irfft2.rst>
   rfftn (N-dimensional FFT for real input) <api/real/rfftn.rst>
   irfftn (N-dimensional FFT for real input) <api/real/irfftn.rst>

Hermitian FFTs
--------------

.. toctree::
   :maxdepth: 1

   hfft (One dimensional FFT of a signal that has Hermitian symmetry) <api/hermitian/hfft.rst>
   ihfft (Inverse of hfft) <api/hermitian/ihfft.rst>

Helper routines
---------------

.. toctree::
   :maxdepth: 1

   fftfreq (Return the DFT sample frequencies) <api/helper/fftfreq.rst>
   rfftfreq (Return the DFT sample frequencies for Real FFTs) <api/helper/rfftfreq.rst>
   fftshift (Shift the zero-frequency component to the center of the spectrum) <api/helper/fftshift.rst>
   ifftshift (Inverse of fftshift) <api/helper/ifftshift.rst>