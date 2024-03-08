
KokkosFFT::fftn
---------------

.. doxygenfunction:: KokkosFFT::fftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization, shape_type<DIM> s)
.. doxygenfunction:: KokkosFFT::fftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, axis_type<DIM1> axes, KokkosFFT::Normalization, shape_type<DIM2> s)
.. doxygenfunction:: KokkosFFT::fftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, const PlanType& plan, axis_type<DIM1> axes, KokkosFFT::Normalization norm, shape_type<DIM2> s)