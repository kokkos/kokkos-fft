#ifndef KOKKOSFFT_PLANS_HPP
#define KOKKOSFFT_PLANS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
using default_device = Kokkos::Cuda;
#include "KokkosFFT_Cuda_plans.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_HIP)
using default_device = Kokkos::HIP;
#include "KokkosFFT_HIP_plans.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_OPENMP)
using default_device = Kokkos::OpenMP;
#include "KokkosFFT_OpenMP_plans.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
using default_device = Kokkos::Threads;
#include "KokkosFFT_OpenMP_plans.hpp"
#else
using default_device = Kokkos::Serial;
#include "KokkosFFT_OpenMP_plans.hpp"
#endif

namespace KokkosFFT {
namespace Impl {
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
class Plan {
  using execSpace      = ExecutionSpace;
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  using float_type     = KokkosFFT::Impl::real_type_t<in_value_type>;
  using fft_plan_type =
      typename KokkosFFT::Impl::FFTPlanType<ExecutionSpace, float_type>::type;
  using fft_size_type       = std::size_t;
  using map_type            = axis_type<InViewType::rank()>;
  using nonConstInViewType  = std::remove_cv_t<InViewType>;
  using nonConstOutViewType = std::remove_cv_t<OutViewType>;
  using extents_type = shape_type<InViewType::rank()>;

  fft_plan_type m_plan;
  fft_size_type m_fft_size;
  map_type m_map, m_map_inv;
  bool m_is_transpose_needed;
  axis_type<DIM> m_axes;
  KokkosFFT::Impl::Direction m_direction;

  // Keep extents
  extents_type m_in_extents, m_out_extents;

  // Only used when transpose needed
  nonConstInViewType m_in_T;
  nonConstOutViewType m_out_T;

 public:
  explicit Plan(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, KokkosFFT::Impl::Direction direction,
                int axis)
      : m_fft_size(1), m_is_transpose_needed(false), m_direction(direction) {
    static_assert(Kokkos::is_view<InViewType>::value,
                  "Plan::Plan: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                  "Plan::Plan: OutViewType is not a Kokkos::View.");
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "Plan::Plan: InViewType must be either LayoutLeft or LayoutRight.");
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "Plan::Plan: OutViewType must be either LayoutLeft or LayoutRight.");

    static_assert(InViewType::rank() == OutViewType::rank(),
                "Plan::Plan: InViewType and OutViewType must have "
                "the same rank.");
    static_assert(std::is_same_v<typename InViewType::array_layout, typename OutViewType::array_layout>,
                "Plan::Plan: InViewType and OutViewType must have "
                "the same Layout.");

    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "Plan::Plan: execution_space cannot access data in InViewType");
    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename OutViewType::memory_space>::accessible,
      "Plan::Plan: execution_space cannot access data in OutViewType");

    m_axes                     = {axis};
    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axis);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_fft_size = KokkosFFT::Impl::_create(exec_space, m_plan, in, out,
                                          direction, m_axes);
  }

  explicit Plan(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, KokkosFFT::Impl::Direction direction,
                axis_type<DIM> axes)
      : m_fft_size(1),
        m_is_transpose_needed(false),
        m_direction(direction),
        m_axes(axes) {
    static_assert(Kokkos::is_view<InViewType>::value,
                  "Plan::Plan: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                  "Plan::Plan: OutViewType is not a Kokkos::View.");
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "Plan::Plan: InViewType must be either LayoutLeft or LayoutRight.");
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "Plan::Plan: OutViewType must be either LayoutLeft or LayoutRight.");

    static_assert(InViewType::rank() == OutViewType::rank(),
                "Plan::Plan: InViewType and OutViewType must have "
                "the same rank.");

    static_assert(std::is_same_v<typename InViewType::array_layout, typename OutViewType::array_layout>,
                "Plan::Plan: InViewType and OutViewType must have "
                "the same Layout.");

    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "Plan::Plan: execution_space cannot access data in InViewType");
    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename OutViewType::memory_space>::accessible,
      "Plan::Plan: execution_space cannot access data in OutViewType");

    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axes);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_fft_size =
        KokkosFFT::Impl::_create(exec_space, m_plan, in, out, direction, axes);
  }

  ~Plan() { _destroy<ExecutionSpace, float_type>(m_plan); }

  template <typename ExecutionSpace2, typename InViewType2,
            typename OutViewType2>
  void good(const InViewType2& in, const OutViewType2& out, KokkosFFT::Impl::Direction direction, axis_type<DIM> axes) const {
    static_assert(std::is_same_v<ExecutionSpace2, execSpace>,
                  "Plan::good: Execution spaces for plan and "
                  "execution are not identical.");
    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace2,
                                 typename InViewType2::memory_space>::accessible,
      "Plan::good: execution_space cannot access data in InViewType");
    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace2,
                                 typename OutViewType2::memory_space>::accessible,
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

  fft_plan_type plan() const { return m_plan; }
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