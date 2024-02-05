
ifft2 (Two dimensional FFT in backward direction)
-------------------------------------------------

.. doxygenfunction:: KokkosFFT::ifft2(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization, axis_type<2> axes, shape_type<DIM> s)
.. doxygenfunction:: KokkosFFT::ifft2(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, const PlanType& plan, KokkosFFT::Normalization norm, axis_type<2> axes, shape_type<DIM> s)
