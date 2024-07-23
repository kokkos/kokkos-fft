// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

/// \file KokkosFFT_Plans.hpp
/// \brief Wrapping fft plans of different fft libraries
///
/// This file provides KokkosFFT::Impl::Plan.
/// This implements a local (no MPI) interface for fft plans

#ifndef KOKKOSFFT_PLANS_HPP
#define KOKKOSFFT_PLANS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_padding.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include "KokkosFFT_Cuda_plans.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_HIP)
#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include "KokkosFFT_ROCM_plans.hpp"
#else
#include "KokkosFFT_HIP_plans.hpp"
#endif
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_SYCL)
#include "KokkosFFT_SYCL_plans.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_plans.hpp"
#endif
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "KokkosFFT_Host_plans.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
#include "KokkosFFT_Host_plans.hpp"
#else
#include "KokkosFFT_Host_plans.hpp"
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
 public:
  //! The type of Kokkos execution pace
  using execSpace = ExecutionSpace;

  //! The value type of input view
  using in_value_type = typename InViewType::non_const_value_type;

  //! The value type of output view
  using out_value_type = typename OutViewType::non_const_value_type;

  //! The real value type of input/output views
  using float_type = KokkosFFT::Impl::base_floating_point_type<in_value_type>;

  //! The layout type of input/output views
  using layout_type = typename InViewType::array_layout;

  //! The type of fft plan
  using fft_plan_type =
      typename KokkosFFT::Impl::FFTPlanType<ExecutionSpace, in_value_type,
                                            out_value_type>::type;

  //! The type of fft info (used for rocfft only)
  using fft_info_type = typename KokkosFFT::Impl::FFTInfoType<ExecutionSpace>;

  //! The type of fft size
  using fft_size_type = std::size_t;

  //! The type of map for transpose
  using map_type = axis_type<InViewType::rank()>;

  //! The non-const View type of input view
  using nonConstInViewType = std::remove_cv_t<InViewType>;

  //! The non-const View type of output view
  using nonConstOutViewType = std::remove_cv_t<OutViewType>;

  //! Naive 1D View for work buffer
  using BufferViewType =
      Kokkos::View<Kokkos::complex<float_type>*, layout_type, execSpace>;

  //! The type of extents of input/output views
  using extents_type = shape_type<InViewType::rank()>;

 private:
  //! Execution space
  execSpace m_exec_space;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<fft_plan_type> m_plan;

  //! fft info
  fft_info_type m_info;

  //! fft size
  fft_size_type m_fft_size = 1;

  //! maps for forward and backward transpose
  map_type m_map, m_map_inv;

  //! whether transpose is needed or not
  bool m_is_transpose_needed = false;

  //! whether crop or pad is needed or not
  bool m_is_crop_or_pad_needed = false;

  //! axes for fft
  axis_type<DIM> m_axes;

  //! Shape of the transformed axis of the output
  extents_type m_shape;

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
  /// \param n [in] Length of the transformed axis of the output (optional)
  //
  explicit Plan(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, KokkosFFT::Direction direction, int axis,
                std::optional<std::size_t> n = std::nullopt)
      : m_exec_space(exec_space), m_axes({axis}), m_direction(direction) {
    static_assert(
        KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                                OutViewType>,
        "Plan::Plan: InViewType and OutViewType must have the same base "
        "floating point type (float/double), the same layout "
        "(LayoutLeft/LayoutRight), "
        "and the same rank. ExecutionSpace must be accessible to the data in "
        "InViewType and OutViewType.");

    if (KokkosFFT::Impl::is_real_v<in_value_type> &&
        m_direction != KokkosFFT::Direction::forward) {
      throw std::runtime_error(
          "Plan::Plan: real to complex transform is constrcuted with backward "
          "direction.");
    }

    if (KokkosFFT::Impl::is_real_v<out_value_type> &&
        m_direction != KokkosFFT::Direction::backward) {
      throw std::runtime_error(
          "Plan::Plan: complex to real transform is constrcuted with forward "
          "direction.");
    }

    shape_type<1> s = {0};
    if (n) {
      std::size_t _n = n.value();
      s              = shape_type<1>({_n});
    }

    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axis);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_shape = KokkosFFT::Impl::get_modified_shape(in, out, s, m_axes);
    m_is_crop_or_pad_needed =
        KokkosFFT::Impl::is_crop_or_pad_needed(in, m_shape);
    m_fft_size = KokkosFFT::Impl::create_plan(
        exec_space, m_plan, in, out, m_buffer, m_info, direction, m_axes, s);
  }

  /// \brief Constructor for multidimensional FFT
  ///
  /// \param exec_space [in] Kokkos execution space for this plan
  /// \param in [in] Input data
  /// \param out [in] Ouput data
  /// \param direction [in] Direction of FFT (forward/backward)
  /// \param axes [in] Axes over which FFT is performed
  /// \param s [in] Shape of the transformed axis of the output (optional)
  //
  explicit Plan(const ExecutionSpace& exec_space, InViewType& in,
                OutViewType& out, KokkosFFT::Direction direction,
                axis_type<DIM> axes, shape_type<DIM> s = {0})
      : m_exec_space(exec_space), m_axes(axes), m_direction(direction) {
    static_assert(
        KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                                OutViewType>,
        "Plan::Plan: InViewType and OutViewType must have the same base "
        "floating point type (float/double), the same layout "
        "(LayoutLeft/LayoutRight), "
        "and the same rank. ExecutionSpace must be accessible to the data in "
        "InViewType and OutViewType.");

    if (std::is_floating_point<in_value_type>::value &&
        m_direction != KokkosFFT::Direction::forward) {
      throw std::runtime_error(
          "Plan::Plan: real to complex transform is constrcuted with backward "
          "direction.");
    }

    if (std::is_floating_point<out_value_type>::value &&
        m_direction != KokkosFFT::Direction::backward) {
      throw std::runtime_error(
          "Plan::Plan: complex to real transform is constrcuted with forward "
          "direction.");
    }

    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axes);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_shape = KokkosFFT::Impl::get_modified_shape(in, out, s, m_axes);
    m_is_crop_or_pad_needed =
        KokkosFFT::Impl::is_crop_or_pad_needed(in, m_shape);
    m_fft_size = KokkosFFT::Impl::create_plan(
        exec_space, m_plan, in, out, m_buffer, m_info, direction, axes, s);
  }

  ~Plan() {
    destroy_plan_and_info<ExecutionSpace, fft_plan_type, fft_info_type>(m_plan,
                                                                        m_info);
  }

  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  Plan& operator=(Plan&&) = delete;
  Plan(Plan&&)            = delete;

  /// \brief Sanity check of the plan used to call FFT interface with
  ///        pre-defined FFT plan. This raises an error if there is an
  ///        incosistency between FFT function and plan
  ///
  /// \param in [in] Input data
  /// \param out [in] Ouput data
  template <typename InViewType2, typename OutViewType2>
  void good(const InViewType2& in, const OutViewType2& out) const {
    using nonConstInViewType2  = std::remove_cv_t<InViewType2>;
    using nonConstOutViewType2 = std::remove_cv_t<OutViewType2>;
    static_assert(std::is_same_v<nonConstInViewType2, nonConstInViewType>,
                  "Plan::good: InViewType for plan and execution "
                  "are not identical.");
    static_assert(std::is_same_v<nonConstOutViewType2, nonConstOutViewType>,
                  "Plan::good: OutViewType for plan and "
                  "execution are not identical.");

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

  /// \brief Return the execution space
  execSpace const& exec_space() const noexcept { return m_exec_space; }

  /// \brief Return the FFT plan
  fft_plan_type& plan() const { return *m_plan; }

  /// \brief Return the FFT info
  fft_info_type const& info() const { return m_info; }

  /// \brief Return the FFT size
  fft_size_type fft_size() const { return m_fft_size; }
  KokkosFFT::Direction direction() const { return m_direction; }
  bool is_transpose_needed() const { return m_is_transpose_needed; }
  bool is_crop_or_pad_needed() const { return m_is_crop_or_pad_needed; }
  extents_type shape() const { return m_shape; }
  map_type map() const { return m_map; }
  map_type map_inv() const { return m_map_inv; }
};
}  // namespace Impl
}  // namespace KokkosFFT

#endif