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
#include "KokkosFFT_normalization.hpp"
#include "KokkosFFT_padding.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include "KokkosFFT_Cuda_plans.hpp"
#include "KokkosFFT_Cuda_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_HIP)
#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include "KokkosFFT_ROCM_plans.hpp"
#include "KokkosFFT_ROCM_transform.hpp"
#else
#include "KokkosFFT_HIP_plans.hpp"
#include "KokkosFFT_HIP_transform.hpp"
#endif
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_SYCL)
#include "KokkosFFT_SYCL_plans.hpp"
#include "KokkosFFT_SYCL_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#else
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#endif

namespace KokkosFFT {
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
 private:
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
  /// \param n [in] Length of the transformed axis of the output (default,
  /// nullopt)
  //
  explicit Plan(const ExecutionSpace& exec_space, const InViewType& in,
                const OutViewType& out, KokkosFFT::Direction direction,
                int axis, std::optional<std::size_t> n = std::nullopt)
      : m_exec_space(exec_space), m_axes({axis}), m_direction(direction) {
    static_assert(KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>,
                  "Plan::Plan: ExecutionSpace is not allowed ");
    static_assert(
        KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                                OutViewType>,
        "Plan::Plan: InViewType and OutViewType must have the same base "
        "floating point type (float/double), the same layout "
        "(LayoutLeft/LayoutRight), "
        "and the same rank. ExecutionSpace must be accessible to the data in "
        "InViewType and OutViewType.");
    static_assert(InViewType::rank() >= 1,
                  "Plan::Plan: View rank must be larger than or equal to 1");

    KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, m_axes),
                       "axes are invalid for in/out views");

    if constexpr (KokkosFFT::Impl::is_real_v<in_value_type>) {
      KOKKOSFFT_THROW_IF(
          m_direction != KokkosFFT::Direction::forward,
          "real to complex transform is constructed with backward direction.");
    }

    if constexpr (KokkosFFT::Impl::is_real_v<out_value_type>) {
      KOKKOSFFT_THROW_IF(
          m_direction != KokkosFFT::Direction::backward,
          "complex to real transform is constructed with forward direction.");
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
  /// \param s [in] Shape of the transformed axis of the output (default, {})
  //
  explicit Plan(const ExecutionSpace& exec_space, const InViewType& in,
                const OutViewType& out, KokkosFFT::Direction direction,
                axis_type<DIM> axes, shape_type<DIM> s = {})
      : m_exec_space(exec_space), m_axes(axes), m_direction(direction) {
    static_assert(KokkosFFT::Impl::is_AllowedSpace_v<ExecutionSpace>,
                  "Plan::Plan: ExecutionSpace is not allowed ");
    static_assert(
        KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                                OutViewType>,
        "Plan::Plan: InViewType and OutViewType must have the same base "
        "floating point type (float/double), the same layout "
        "(LayoutLeft/LayoutRight), "
        "and the same rank. ExecutionSpace must be accessible to the data in "
        "InViewType and OutViewType.");
    static_assert(
        DIM >= 1 && DIM <= KokkosFFT::MAX_FFT_DIM,
        "Plan::Plan: the Rank of FFT axes must be between 1 and MAX_FFT_DIM");
    static_assert(InViewType::rank() >= DIM,
                  "Plan::Plan: View rank must be larger than or equal to the "
                  "Rank of FFT axes");

    KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, m_axes),
                       "axes are invalid for in/out views");
    if constexpr (KokkosFFT::Impl::is_real_v<in_value_type>) {
      KOKKOSFFT_THROW_IF(
          m_direction != KokkosFFT::Direction::forward,
          "real to complex transform is constructed with backward direction.");
    }

    if constexpr (KokkosFFT::Impl::is_real_v<out_value_type>) {
      KOKKOSFFT_THROW_IF(
          m_direction != KokkosFFT::Direction::backward,
          "complex to real transform is constructed with forward direction.");
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
    KokkosFFT::Impl::destroy_plan_and_info<ExecutionSpace, fft_plan_type,
                                           fft_info_type>(m_plan, m_info);
  }

  Plan()            = delete;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;
  Plan& operator=(Plan&&) = delete;
  Plan(Plan&&)            = delete;

  /// \brief Execute FFT on input and output Views with normalization
  ///
  /// \param in [in] Input data
  /// \param out [out] Ouput data
  /// \param norm [in] How the normalization is applied (default, backward)
  void execute(const InViewType& in, const OutViewType& out,
               KokkosFFT::Normalization norm =
                   KokkosFFT::Normalization::backward) const {
    static_assert(
        KokkosFFT::Impl::are_operatable_views_v<execSpace, InViewType,
                                                OutViewType>,
        "Plan::execute: InViewType and OutViewType must have the same base "
        "floating point "
        "type (float/double), the same layout (LayoutLeft/LayoutRight), and "
        "the "
        "same rank. ExecutionSpace must be accessible to the data in "
        "InViewType "
        "and OutViewType.");

    // sanity check that the plan is consistent with the input/output views
    good(in, out);

    using ManagableInViewType =
        typename KokkosFFT::Impl::manageable_view_type<InViewType>::type;
    using ManagableOutViewType =
        typename KokkosFFT::Impl::manageable_view_type<OutViewType>::type;
    ManagableInViewType in_s;
    InViewType in_tmp;
    if (m_is_crop_or_pad_needed) {
      KokkosFFT::Impl::crop_or_pad(m_exec_space, in, in_s, m_shape);
      in_tmp = in_s;
    } else {
      in_tmp = in;
    }

    if (m_is_transpose_needed) {
      using LayoutType = typename ManagableInViewType::array_layout;
      ManagableInViewType const in_T(
          "in_T",
          KokkosFFT::Impl::create_layout<LayoutType>(
              KokkosFFT::Impl::compute_transpose_extents(in_tmp, m_map)));
      ManagableOutViewType const out_T(
          "out_T", KokkosFFT::Impl::create_layout<LayoutType>(
                       KokkosFFT::Impl::compute_transpose_extents(out, m_map)));

      KokkosFFT::Impl::transpose(m_exec_space, in_tmp, in_T, m_map);
      KokkosFFT::Impl::transpose(m_exec_space, out, out_T, m_map);

      execute_fft(in_T, out_T, norm);

      KokkosFFT::Impl::transpose(m_exec_space, out_T, out, m_map_inv);
    } else {
      execute_fft(in_tmp, out, norm);
    }
  }

 private:
  void execute_fft(const InViewType& in, const OutViewType& out,
                   KokkosFFT::Normalization norm) const {
    auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
        execSpace, in_value_type>::type*>(in.data());
    auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
        execSpace, out_value_type>::type*>(out.data());

    auto const direction =
        KokkosFFT::Impl::direction_type<execSpace>(m_direction);
    KokkosFFT::Impl::exec_plan(*m_plan, idata, odata, direction, m_info);
    KokkosFFT::Impl::normalize(m_exec_space, out, m_direction, norm,
                               m_fft_size);
  }

  /// \brief Sanity check of the plan used to call FFT interface with
  ///        pre-defined FFT plan. This raises an error if there is an
  ///        incosistency between FFT function and plan
  ///
  /// \param in [in] Input data
  /// \param out [in] Ouput data
  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    KOKKOSFFT_THROW_IF(
        in_extents != m_in_extents,
        "extents of input View for plan and execution are not identical.");

    KOKKOSFFT_THROW_IF(
        out_extents != m_out_extents,
        "extents of output View for plan and execution are not identical.");
  }
};
}  // namespace KokkosFFT

#endif
