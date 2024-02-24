/// \file KokkosFFT_Plans.hpp
/// \brief Wrapping fft plans of different fft libraries
///
/// This file provides KokkosFFT::Impl::Plan.
/// This implements a local (no MPI) interface for fft plans

#ifndef KOKKOSFFT_PLANS_HPP
#define KOKKOSFFT_PLANS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include "KokkosFFT_Cuda_plans.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_HIP)
#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include "KokkosFFT_ROCM_plans.hpp"
#else
#include "KokkosFFT_HIP_plans.hpp"
#endif
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_SYCL)
#include "KokkosFFT_SYCL_plans.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "KokkosFFT_OpenMP_plans.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
#include "KokkosFFT_OpenMP_plans.hpp"
#else
#include "KokkosFFT_OpenMP_plans.hpp"
#endif

namespace KokkosFFT {
namespace Impl {
/// \brief A class that manages a FFT plan of backend FFT library.
///
/// This class is used to manage the FFT plan of backend FFT library.
/// Depending on the input and output Views and axes, appropriate FFT plans are
/// created. If there are inconsistency in input and output views, the
/// compilation would fail.
///
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class Plan {
  //! The type of Kokkos execution pace
  using execSpace = ExecutionSpace;

  //! The value type of input view
  using in_value_type = typename InViewType::non_const_value_type;

  //! The value type of output view
  using out_value_type = typename OutViewType::non_const_value_type;

  //! The real value type of input/output views
  using float_type = KokkosFFT::Impl::real_type_t<in_value_type>;

  //! The layout type of input/output views
  using layout_type = typename InViewType::array_layout;

  //! The type of fft plan
  using fft_plan_type =
      typename KokkosFFT::Impl::FFTPlanType<ExecutionSpace, in_value_type,
                                            out_value_type>::type;

  //! The type of fft info (used for rocfft only)
  using fft_info_type = 
      typename KokkosFFT::Impl::FFTInfoType<ExecutionSpace>;

  //! The type of fft size
  using fft_size_type = std::size_t;

  //! The type of map for transpose
  using map_type = axis_type<InViewType::rank()>;

  //! The non-const View type of input view
  using nonConstInViewType = std::remove_cv_t<InViewType>;

  //! The non-const View type of output view
  using nonConstOutViewType = std::remove_cv_t<OutViewType>;

  //! Naive 1D View for work buffer
  using BufferViewType = Kokkos::View<Kokkos::complex<float_type>*, layout_type, execSpace>;

  //! The type of extents of input/output views
  using extents_type = shape_type<InViewType::rank()>;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<fft_plan_type> m_plan;

  //! fft info
  fft_info_type m_info;

  //! fft size
  fft_size_type m_fft_size;

  //! maps for forward and backward transpose
  map_type m_map, m_map_inv;

  //! whether transpose is needed or not
  bool m_is_transpose_needed;

  //! axes for fft
  axis_type<DIM> m_axes;

  //! directions of fft
  KokkosFFT::Direction m_direction;

  ///@{
  //! extents of input/output views
  extents_type m_in_extents, m_out_extents;
  ///@}

  //! @{
  //! Internal buffers used for transpose
  nonConstInViewType m_in_T;
  nonConstOutViewType m_out_T;
  //! @}

  //! Internal work buffer (for rocfft)
  BufferViewType m_buffer;

 public:
  /// \brief Constructor
  ///
  /// \param exec_space [in] Kokkos execution device
  /// \param in [in] Input data
  /// \param out [in] Ouput data
  /// \param direction [in] Direction of FFT (forward/backward)
  /// \param axis [in] Axis over which FFT is performed
  //
  explicit Plan(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, KokkosFFT::Direction direction, int axis)
      : m_fft_size(1), m_is_transpose_needed(false), m_direction(direction) {
    static_assert(Kokkos::is_view<InViewType>::value,
                  "Plan::Plan: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                  "Plan::Plan: OutViewType is not a Kokkos::View.");
    static_assert(
        KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
        "Plan::Plan: InViewType must be either LayoutLeft or LayoutRight.");
    static_assert(
        KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
        "Plan::Plan: OutViewType must be either LayoutLeft or LayoutRight.");

    static_assert(InViewType::rank() == OutViewType::rank(),
                  "Plan::Plan: InViewType and OutViewType must have "
                  "the same rank.");
    static_assert(std::is_same_v<typename InViewType::array_layout,
                                 typename OutViewType::array_layout>,
                  "Plan::Plan: InViewType and OutViewType must have "
                  "the same Layout.");

    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename InViewType::memory_space>::accessible,
        "Plan::Plan: execution_space cannot access data in InViewType");
    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename OutViewType::memory_space>::accessible,
        "Plan::Plan: execution_space cannot access data in OutViewType");

    m_axes                     = {axis};
    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axis);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_fft_size = KokkosFFT::Impl::_create(exec_space, m_plan, in, out, m_buffer,
                                          m_info, direction, m_axes);
  }

  /// \brief Constructor for multidimensional FFT
  ///
  /// \param exec_space [in] Kokkos execution space for this plan
  /// \param in [in] Input data
  /// \param out [in] Ouput data
  /// \param direction [in] Direction of FFT (forward/backward)
  /// \param axes [in] Axes over which FFT is performed
  //
  explicit Plan(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, KokkosFFT::Direction direction,
                axis_type<DIM> axes)
      : m_fft_size(1),
        m_is_transpose_needed(false),
        m_direction(direction),
        m_axes(axes) {
    static_assert(Kokkos::is_view<InViewType>::value,
                  "Plan::Plan: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                  "Plan::Plan: OutViewType is not a Kokkos::View.");
    static_assert(
        KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
        "Plan::Plan: InViewType must be either LayoutLeft or LayoutRight.");
    static_assert(
        KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
        "Plan::Plan: OutViewType must be either LayoutLeft or LayoutRight.");

    static_assert(InViewType::rank() == OutViewType::rank(),
                  "Plan::Plan: InViewType and OutViewType must have "
                  "the same rank.");

    static_assert(std::is_same_v<typename InViewType::array_layout,
                                 typename OutViewType::array_layout>,
                  "Plan::Plan: InViewType and OutViewType must have "
                  "the same Layout.");

    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename InViewType::memory_space>::accessible,
        "Plan::Plan: execution_space cannot access data in InViewType");
    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace, typename OutViewType::memory_space>::accessible,
        "Plan::Plan: execution_space cannot access data in OutViewType");

    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axes);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_fft_size =
        KokkosFFT::Impl::_create(exec_space, m_plan, in, out, m_buffer,
                                 m_info, direction, axes);
  }

  ~Plan() {
    _destroy_info<ExecutionSpace, fft_info_type>(m_info);
    _destroy_plan<ExecutionSpace, fft_plan_type>(m_plan);
  }

  /// \brief Sanity check of the plan used to call FFT interface with
  ///        pre-defined FFT plan. This raises an error if there is an
  ///        incosistency between FFT function and plan
  ///
  /// \param in [in] Input data
  /// \param out [in] Ouput data
  /// \param direction [in] Direction of FFT (forward/backward)
  /// \param axes [in] Axes over which FFT is performed
  template <typename ExecutionSpace2, typename InViewType2,
            typename OutViewType2>
  void good(const InViewType2& in, const OutViewType2& out,
            KokkosFFT::Direction direction, axis_type<DIM> axes) const {
    static_assert(std::is_same_v<ExecutionSpace2, execSpace>,
                  "Plan::good: Execution spaces for plan and "
                  "execution are not identical.");
    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace2, typename InViewType2::memory_space>::accessible,
        "Plan::good: execution_space cannot access data in InViewType");
    static_assert(
        Kokkos::SpaceAccessibility<
            ExecutionSpace2, typename OutViewType2::memory_space>::accessible,
        "Plan::good: execution_space cannot access data in OutViewType");

    using nonConstInViewType2  = std::remove_cv_t<InViewType2>;
    using nonConstOutViewType2 = std::remove_cv_t<OutViewType2>;
    static_assert(std::is_same_v<nonConstInViewType2, nonConstInViewType>,
                  "Plan::good: InViewType for plan and execution "
                  "are not identical.");
    static_assert(std::is_same_v<nonConstOutViewType2, nonConstOutViewType>,
                  "Plan::good: OutViewType for plan and "
                  "execution are not identical.");

    if (direction != m_direction) {
      throw std::runtime_error(
          "Plan::good: directions for plan and execution are "
          "not identical.");
    }

    if (axes != m_axes) {
      throw std::runtime_error(
          "Plan::good: axes for plan and execution are "
          "not identical.");
    }

    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);
    if (in_extents != m_in_extents) {
      throw std::runtime_error(
          "Plan::good: extents of input View for plan and execution are "
          "not identical.");
    }

    if (out_extents != m_out_extents) {
      throw std::runtime_error(
          "Plan::good: extents of output View for plan and execution are "
          "not identical.");
    }
  }

  /// \brief Return the FFT plan
  fft_plan_type& plan() const { return *m_plan; }

  /// \brief Return the FFT info
  const fft_info_type& info() const { return m_info; }

  /// \brief Return the FFT size
  fft_size_type fft_size() const { return m_fft_size; }
  bool is_transpose_needed() const { return m_is_transpose_needed; }
  map_type map() const { return m_map; }
  map_type map_inv() const { return m_map_inv; }
  nonConstInViewType& in_T() { return m_in_T; }
  nonConstOutViewType& out_T() { return m_out_T; }
};
}  // namespace Impl
}  // namespace KokkosFFT

#endif