// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

/// \file KokkosFFT_DynPlans.hpp
/// \brief Wrapping fft plans of different fft libraries
///
/// This file provides KokkosFFT::Plan.
/// This implements a local (no MPI) interface for fft plans

#ifndef KOKKOSFFT_DYNPLANS_HPP
#define KOKKOSFFT_DYNPLANS_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_Normalization.hpp"
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

/// \brief A class that manages a FFT plan of backend FFT library
/// FFT dimension is given at runtime.
///
/// This class is used to manage the FFT plan of backend FFT library.
/// Appropriate FFT plans are created depending on the input and output Views
/// and a provided FFT dimension. If there are inconsistency in input and output
/// views, the compilation will fail.
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
class DynPlan {
  static_assert(
      KokkosFFT::Impl::is_allowed_space_v<ExecutionSpace>,
      "DynPlan: Backend fft library is not available for the ExecutionSpace");
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "DynPlan");

  //! The type of Kokkos execution pace
  using execSpace = ExecutionSpace;

  //! The value type of input view
  using in_value_type = typename InViewType::non_const_value_type;

  //! The value type of output view
  using out_value_type = typename OutViewType::non_const_value_type;

  //! The real value type of input/output views
  using float_type = KokkosFFT::Impl::base_floating_point_type<in_value_type>;

  //! The type of fft plan
  using fft_plan_type =
      typename KokkosFFT::Impl::FFTDynPlanType<ExecutionSpace, in_value_type,
                                               out_value_type>::type;

  //! The type of extents of fft
  using fft_extents_type = std::vector<std::size_t>;

  //! The real value type for normalization
  using normalization_float_type = double;

  //! The type of extents of input/output views
  using extents_type = shape_type<InViewType::rank()>;

 private:
  //! Execution space
  execSpace m_exec_space;

  //! Dynamically allocatable fft plan.
  std::unique_ptr<fft_plan_type> m_plan;

  //! fft extents
  fft_extents_type m_fft_extents;

  //! in-place transform or not
  bool m_is_inplace = false;

  //! directions of fft
  KokkosFFT::Direction m_direction;

  ///@{
  //! extents of input/output views
  extents_type m_in_extents, m_out_extents;
  ///@}

 public:
  /// \brief Constructor for multidimensional FFT
  ///
  /// \param[in] exec_space  Kokkos execution space for this plan
  /// \param[in] in Input data
  /// \param[in] out Output data
  /// \param[in] direction Direction of FFT (forward/backward)
  /// \param[in] dim The dimensionality of FFT
  explicit DynPlan(const ExecutionSpace& exec_space, const InViewType& in,
                   const OutViewType& out, KokkosFFT::Direction direction,
                   std::size_t dim)
      : m_exec_space(exec_space), m_direction(direction) {
    KOKKOSFFT_THROW_IF(dim < 1 || dim > 3,
                       "only 1D, 2D, and 3D FFTs are supported.");
    KOKKOSFFT_THROW_IF(dim > InViewType::rank(),
                       "Rank of View must be larger than Rank of FFT.");

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

    m_in_extents  = KokkosFFT::Impl::extract_extents(in);
    m_out_extents = KokkosFFT::Impl::extract_extents(out);
    m_is_inplace  = KokkosFFT::Impl::are_aliasing(in.data(), out.data());
    KokkosFFT::Impl::setup<ExecutionSpace, float_type>();
    m_fft_extents = KokkosFFT::Impl::create_dynplan(
        exec_space, m_plan, in, out, direction, dim, m_is_inplace);
  }

  ~DynPlan() noexcept = default;

  DynPlan()                          = delete;
  DynPlan(const DynPlan&)            = delete;
  DynPlan& operator=(const DynPlan&) = delete;
  DynPlan& operator=(DynPlan&&)      = delete;
  DynPlan(DynPlan&&)                 = delete;

  /// \brief Return the workspace size to be allocated in ScalarType
  /// We allocate the buffer using Kokkos::View<ScalarType*>
  /// \return the workspace size to be allocated in ScalarType
  std::size_t workspace_size(std::size_t byte) const noexcept {
    return (m_plan->workspace_size() + byte - 1) / byte;
  }

  template <typename WorkViewType>
  void set_work_area(const WorkViewType& work) {
    m_plan->set_work_area(work);
  }

  void execute_impl(const InViewType& in, const OutViewType& out,
                    KokkosFFT::Normalization norm =
                        KokkosFFT::Normalization::backward) const {
    // sanity check that the plan is consistent with the input/output views
    good(in, out);
    execute_fft(in, out, norm);
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
        KokkosFFT::Impl::normalize<normalization_float_type>(
            m_exec_space, out_tmp, m_direction, norm, m_fft_extents);
        return;
      }
    }
    KokkosFFT::Impl::normalize<normalization_float_type>(
        m_exec_space, out, m_direction, norm, m_fft_extents);
  }

  /// \brief Sanity check of the plan used to call FFT interface with
  ///        pre-defined FFT plan. This raises an error if there is an
  ///        inconsistency between FFT function and plan
  ///
  /// \param[in] in Input data
  /// \param[in] out Output data
  void good(const InViewType& in, const OutViewType& out) const {
    auto in_extents  = KokkosFFT::Impl::extract_extents(in);
    auto out_extents = KokkosFFT::Impl::extract_extents(out);

    KOKKOSFFT_THROW_IF(in_extents != m_in_extents,
                       "extents of input View for plan and "
                       "execution are not identical.");

    KOKKOSFFT_THROW_IF(out_extents != m_out_extents,
                       "extents of output View for plan and "
                       "execution are not identical.");

    if (m_is_inplace) {
      bool is_inplace = KokkosFFT::Impl::are_aliasing(in.data(), out.data());
      KOKKOSFFT_THROW_IF(!is_inplace,
                         "If the plan is in-place, the input and output Views "
                         "must be identical.");
    }
  }
};

/// \brief Returns the minimum size needed to fit any workspace from plans.
/// \tparam ScalarType The scalar type used to allocate the workspace buffer.
/// \tparam DynPlans The types of dynamic plans
///
/// \param[in] plans the plans whose workspace we are considering.
/// \returns the minimum size in bytes needed to fit any of the workspace from
/// plans.
template <typename ScalarType, class... DynPlans>
std::size_t compute_required_workspace_size(const DynPlans&... plans) {
  return std::max({plans.workspace_size(sizeof(ScalarType))...});
}

}  // namespace KokkosFFT

#endif
