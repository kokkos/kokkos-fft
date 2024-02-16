
KokkosFFT::fftshift
-------------------
.. doxygenfunction:: KokkosFFT::fftshift(const ExecutionSpace& exec_space, ViewType& inout)
.. doxygenfunction:: KokkosFFT::fftshift(const ExecutionSpace& exec_space, ViewType& inout, int axes)
.. doxygenfunction:: KokkosFFT::fftshift(const ExecutionSpace& exec_space, ViewType& inout, axis_type<DIM> axes)

.. note::

   For the moment, this function works one or two dimensional Views.