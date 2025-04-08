// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

/// \file KokkosFFT_Plans.hpp
/// \brief Wrapping fft plans of different fft libraries
///
/// This file provides KokkosFFT::Plan.
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

#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
#include "KokkosFFT_Cuda_plans.hpp"
#include "KokkosFFT_Cuda_transform.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include "KokkosFFT_ROCM_plans.hpp"
#include "KokkosFFT_ROCM_transform.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
#include "KokkosFFT_HIP_plans.hpp"
#include "KokkosFFT_HIP_transform.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ONEMKL)
#include "KokkosFFT_SYCL_plans.hpp"
#include "KokkosFFT_SYCL_transform.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_Host_plans.hpp"
#include "KokkosFFT_Host_transform.hpp"
#endif

namespace KokkosFFT {
namespace Impl {
#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
// Backend libraries are available from all the execution spaces
template <typename ExecutionSpace>
struct is_AllowedSpace : std::true_type {};
#else
// Only device backend library is available
template <typename ExecutionSpace>
struct is_AllowedSpace
    : std::is_same<ExecutionSpace, Kokkos::DefaultExecutionSpace> {};
#endif
#else
// Only host backend library is available
template <typename ExecutionSpace>
struct is_AllowedSpace : is_AnyHostSpace<ExecutionSpace> {};
#endif

/// \brief Helper to check if the ExecutionSpace is allowed to construct a plan
template <typename ExecutionSpace>
inline constexpr bool is_AllowedSpace_v =
    is_AllowedSpace<ExecutionSpace>::value;

}  // namespace Impl

/// \brief A class that manages a FFT plan of backend FFT library.
///
/// This class is used to manage the FFT plan of backend FFT library.
/// Depending on the input and output Views and axes, appropriate FFT plans are
/// created. If there are inconsistency in input and output views, the
/// compilation would fail.
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
/// \tparam DIM: The dimensionality of the fft
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

  //! The type of fft size
  using fft_size_type = std::size_t;

  //! The type of map for transpose
  using map_type = axis_type<InViewType::rank()>;

  //! The type of extents of input/output views
  using extents_type = shape_type<InViewType::rank()>;

 private:
  //! Execution space
  execSpace m_exec_space;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<fft_plan_type> m_plan;

  //! fft size
  fft_size_type m_fft_size = 1;

  //! maps for forward and backward transpose
  map_type m_map, m_map_inv;

  //! whether transpose is needed or not
  bool m_is_transpose_needed = false;

  //! whether crop or pad is needed or not
  bool m_is_crop_or_pad_needed = false;

  //! in-place transform or not
  bool m_is_inplace = false;

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

 public:
  /// \brief Constructor
  ///
  /// \param exec_space [in] Kokkos execution space for this plan
  /// \param in [in] Input data
  /// \param out [in] Output data
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
      std::size_t n_tmp = n.value();
      s                 = shape_type<1>({n_tmp});
    }

    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axis);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_shape = KokkosFFT::Impl::get_modified_shape(in, out, s, m_axes);
    m_is_crop_or_pad_needed =
        KokkosFFT::Impl::is_crop_or_pad_needed(in, m_shape);
    m_is_inplace = KokkosFFT::Impl::are_aliasing(in.data(), out.data());
    KOKKOSFFT_THROW_IF(m_is_inplace && m_is_transpose_needed,
                       "In-place transform is not supported with transpose. "
                       "Please use out-of-place transform.");
    KOKKOSFFT_THROW_IF(m_is_inplace && m_is_crop_or_pad_needed,
                       "In-place transform is not supported with reshape. "
                       "Please use out-of-place transform.");

    KokkosFFT::Impl::setup<ExecutionSpace, float_type>();
    m_fft_size = KokkosFFT::Impl::create_plan(
        exec_space, m_plan, in, out, direction, m_axes, s, m_is_inplace);
  }

  /// \brief Constructor for multidimensional FFT
  ///
  /// \param exec_space [in] Kokkos execution space for this plan
  /// \param in [in] Input data
  /// \param out [in] Output data
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
      KOKKOSFFT_THROW_IF(m_direction != KokkosFFT::Direction::forward,
                         "real to complex transform is constructed "
                         "with backward direction.");
    }

    if constexpr (KokkosFFT::Impl::is_real_v<out_value_type>) {
      KOKKOSFFT_THROW_IF(m_direction != KokkosFFT::Direction::backward,
                         "complex to real transform is constructed "
                         "with forward direction.");
    }

    m_in_extents               = KokkosFFT::Impl::extract_extents(in);
    m_out_extents              = KokkosFFT::Impl::extract_extents(out);
    std::tie(m_map, m_map_inv) = KokkosFFT::Impl::get_map_axes(in, axes);
    m_is_transpose_needed      = KokkosFFT::Impl::is_transpose_needed(m_map);
    m_shape = KokkosFFT::Impl::get_modified_shape(in, out, s, m_axes);
    m_is_crop_or_pad_needed =
        KokkosFFT::Impl::is_crop_or_pad_needed(in, m_shape);
    m_is_inplace = KokkosFFT::Impl::are_aliasing(in.data(), out.data());
    KOKKOSFFT_THROW_IF(m_is_inplace && m_is_transpose_needed,
                       "In-place transform is not supported with transpose. "
                       "Please use out-of-place transform.");
    KOKKOSFFT_THROW_IF(m_is_inplace && m_is_crop_or_pad_needed,
                       "In-place transform is not supported with reshape. "
                       "Please use out-of-place transform.");

    KokkosFFT::Impl::setup<ExecutionSpace, float_type>();
    m_fft_size = KokkosFFT::Impl::create_plan(exec_space, m_plan, in, out,
                                              direction, axes, s, m_is_inplace);
  }

  ~Plan() noexcept = default;

  Plan()                       = delete;
  Plan(const Plan&)            = delete;
  Plan& operator=(const Plan&) = delete;
  Plan& operator=(Plan&&)      = delete;
  Plan(Plan&&)                 = delete;

  void execute_impl(const InViewType& in, const OutViewType& out,
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

    using ManageableInViewType =
        typename KokkosFFT::Impl::manageable_view_type<InViewType>::type;
    using ManageableOutViewType =
        typename KokkosFFT::Impl::manageable_view_type<OutViewType>::type;

    ManageableInViewType in_s;
    InViewType in_tmp;
    if (m_is_crop_or_pad_needed) {
      using LayoutType = typename ManageableInViewType::array_layout;
      in_s             = ManageableInViewType(
          "in_s", KokkosFFT::Impl::create_layout<LayoutType>(m_shape));
      KokkosFFT::Impl::crop_or_pad(m_exec_space, in, in_s);
      in_tmp = in_s;
    } else {
      in_tmp = in;
    }

    if (m_is_transpose_needed) {
      using LayoutType = typename ManageableInViewType::array_layout;
      ManageableInViewType const in_T(
          "in_T",
          KokkosFFT::Impl::create_layout<LayoutType>(
              KokkosFFT::Impl::compute_transpose_extents(in_tmp, m_map)));
      ManageableOutViewType const out_T(
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
    KokkosFFT::Impl::exec_plan(*m_plan, idata, odata, direction);

    if constexpr (KokkosFFT::Impl::is_complex_v<in_value_type> &&
                  KokkosFFT::Impl::is_real_v<out_value_type>) {
      if (m_is_inplace) {
        // For the in-place Complex to Real transform, the output must be
        // reshaped to fit the original size (in.size() * 2) for correct
        // normalization
        Kokkos::View<out_value_type*, execSpace> out_tmp(out.data(),
                                                         in.size() * 2);
        KokkosFFT::Impl::normalize(m_exec_space, out_tmp, m_direction, norm,
                                   m_fft_size);
        return;
      }
    }
    KokkosFFT::Impl::normalize(m_exec_space, out, m_direction, norm,
                               m_fft_size);
  }

  /// \brief Sanity check of the plan used to call FFT interface with
  ///        pre-defined FFT plan. This raises an error if there is an
  ///        inconsistency between FFT function and plan
  ///
  /// \param in [in] Input data
  /// \param out [in] Output data
  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    KOKKOSFFT_THROW_IF(in_extents != m_in_extents,
                       "extents of input View for plan and "
                       "execution are not identical.");

    KOKKOSFFT_THROW_IF(out_extents != m_out_extents,
                       "extents of output View for plan and "
                       "execution are not identical.");

    bool is_inplace = KokkosFFT::Impl::are_aliasing(in.data(), out.data());
    KOKKOSFFT_THROW_IF(is_inplace != m_is_inplace,
                       "If the plan is in-place, the input and output Views "
                       "must be identical.");
  }
};
}  // namespace KokkosFFT

#endif
