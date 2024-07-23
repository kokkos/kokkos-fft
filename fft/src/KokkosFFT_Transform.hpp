// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRANSFORM_HPP
#define KOKKOSFFT_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_normalization.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_padding.hpp"
#include "KokkosFFT_Plans.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
#include "KokkosFFT_Cuda_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_HIP)
#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
#include "KokkosFFT_ROCM_transform.hpp"
#else
#include "KokkosFFT_HIP_transform.hpp"
#endif
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_SYCL)
#include "KokkosFFT_SYCL_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_Host_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_OPENMP)
#include "KokkosFFT_Host_transform.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
#include "KokkosFFT_Host_transform.hpp"
#else
#include "KokkosFFT_Host_transform.hpp"
#endif

#include <type_traits>

// General Transform Interface
namespace KokkosFFT {
namespace Impl {

template <typename PlanType, typename InViewType, typename OutViewType>
void exec_impl(
    const PlanType& plan, const InViewType& in, OutViewType& out,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, in_value_type>::type*>(in.data());
  auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, out_value_type>::type*>(out.data());

  auto const exec_space = plan.exec_space();
  auto const direction  = direction_type<ExecutionSpace>(plan.direction());
  KokkosFFT::Impl::exec_plan(plan.plan(), idata, odata, direction, plan.info());
  KokkosFFT::Impl::normalize(exec_space, out, plan.direction(), norm,
                             plan.fft_size());
}

template <typename PlanType, typename InViewType, typename OutViewType>
void fft_exec_impl(
    const PlanType& plan, const InViewType& in, OutViewType& out,
    // KokkosFFT::Direction direction,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fft_exec_impl: InViewType and OutViewType must have the same base "
      "floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  plan.template good<InViewType, OutViewType>(in, out);

  const auto exec_space = plan.exec_space();
  using ManagableInViewType =
      typename KokkosFFT::Impl::managable_view_type<InViewType>::type;
  using ManagableOutViewType =
      typename KokkosFFT::Impl::managable_view_type<OutViewType>::type;
  ManagableInViewType _in_s;
  InViewType _in;
  if (plan.is_crop_or_pad_needed()) {
    auto new_shape = plan.shape();
    KokkosFFT::Impl::crop_or_pad(exec_space, in, _in_s, new_shape);
    _in = _in_s;
  } else {
    _in = in;
  }

  if (plan.is_transpose_needed()) {
    ManagableInViewType in_T;
    ManagableOutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::exec_impl(plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

  } else {
    KokkosFFT::Impl::exec_impl(plan, _in, out, norm);
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

namespace KokkosFFT {
/// \brief One dimensional FFT in forward direction
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axis [in] Axis over which FFT is performed (optional)
/// \param n [in] Length of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void fft(const ExecutionSpace& exec_space, const InViewType& in,
         OutViewType& out,
         KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
         int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  KokkosFFT::Impl::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward,
                             axis, n);
  KokkosFFT::Impl::fft_exec_impl(plan, in, out, norm);
}

/// \brief One dimensional FFT in backward direction
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axis [in] Axis over which FFT is performed (optional)
/// \param n [in] Length of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ifft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ifft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  KokkosFFT::Impl::Plan plan(exec_space, in, out,
                             KokkosFFT::Direction::backward, axis, n);
  KokkosFFT::Impl::fft_exec_impl(plan, in, out, norm);
}

/// \brief One dimensional FFT for real input
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axis [in] Axis over which FFT is performed (optional)
/// \param n [in] Length of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void rfft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "rfft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "rfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "rfft: OutViewType must be complex");

  fft(exec_space, in, out, norm, axis, n);
}

/// \brief Inverse of rfft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (real)
/// \param norm [in] How the normalization is applied (optional)
/// \param axis [in] Axis over which FFT is performed (optional)
/// \param n [in] Length of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void irfft(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "irfft: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "irfft: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "irfft: OutViewType must be real");
  ifft(exec_space, in, out, norm, axis, n);
}

/// \brief One dimensional FFT of a signal that has Hermitian symmetry
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (real)
/// \param norm [in] How the normalization is applied (optional)
/// \param axis [in] Axis over which FFT is performed (optional)
/// \param n [in] Length of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void hfft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "hfft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  // [TO DO]
  // allow real type as input, need to obtain complex view type from in view
  // type
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "hfft: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "hfft: OutViewType must be real");
  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  // using ComplexViewType = typename
  // KokkosFFT::Impl::complex_view_type<ExecutionSpace, InViewType>::type;
  // ComplexViewType in_conj;
  InViewType in_conj;
  KokkosFFT::Impl::conjugate(exec_space, in, in_conj);
  irfft(exec_space, in_conj, out, new_norm, axis, n);
}

/// \brief Inverse of hfft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axis [in] Axis over which FFT is performed (optional)
/// \param n [in] Length of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ihfft(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ihfft: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "ihfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "ihfft: OutViewType must be complex");

  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  OutViewType out_conj;
  rfft(exec_space, in, out, new_norm, axis, n);
  KokkosFFT::Impl::conjugate(exec_space, out, out_conj);
  out = out_conj;
}

// 2D FFT

/// \brief Two dimensional FFT in forward direction
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axes [in] Axes over which FFT is performed (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void fft2(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          axis_type<2> axes = {-2, -1}, shape_type<2> s = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fft2: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  KokkosFFT::Impl::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward,
                             axes, s);
  KokkosFFT::Impl::fft_exec_impl(plan, in, out, norm);
}

/// \brief Two dimensional FFT in backward direction
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axes [in] Axes over which FFT is performed (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ifft2(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           axis_type<2> axes = {-2, -1}, shape_type<2> s = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ifft2: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  KokkosFFT::Impl::Plan plan(exec_space, in, out,
                             KokkosFFT::Direction::backward, axes, s);
  KokkosFFT::Impl::fft_exec_impl(plan, in, out, norm);
}

/// \brief Two dimensional FFT for real input
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Ouput data (complex)
/// \param norm [in] How the normalization is applied (optional)
/// \param axes [in] Axes over which FFT is performed (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void rfft2(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           axis_type<2> axes = {-2, -1}, shape_type<2> s = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "rfft2: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "rfft2: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "rfft2: OutViewType must be complex");

  fft2(exec_space, in, out, norm, axes, s);
}

/// \brief Inverse of rfft2 with a given plan
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (real)
/// \param norm [in] How the normalization is applied (optional)
/// \param axes [in] Axes over which FFT is performed (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void irfft2(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
            axis_type<2> axes = {-2, -1}, shape_type<2> s = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "irfft2: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "irfft2: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "irfft2: OutViewType must be real");

  ifft2(exec_space, in, out, norm, axes, s);
}

// ND FFT

/// \brief N-dimensional FFT in forward direction with a given plan
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (complex)
/// \param axes [in] Axes over which FFT is performed
/// \param norm [in] How the normalization is applied (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void fftn(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, axis_type<DIM> axes,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          shape_type<DIM> s             = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fftn: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  KokkosFFT::Impl::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward,
                             axes, s);
  KokkosFFT::Impl::fft_exec_impl(plan, in, out, norm);
}

/// \brief N-dimensional FFT in backward direction with a given plan
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (complex)
/// \param axes [in] Axes over which FFT is performed
/// \param norm [in] How the normalization is applied (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void ifftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, axis_type<DIM> axes,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           shape_type<DIM> s             = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ifftn: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  KokkosFFT::Impl::Plan plan(exec_space, in, out,
                             KokkosFFT::Direction::backward, axes, s);
  KokkosFFT::Impl::fft_exec_impl(plan, in, out, norm);
}

/// \brief N-dimensional FFT for real input
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Ouput data (complex)
/// \param axes [in] Axes over which FFT is performed
/// \param norm [in] How the normalization is applied (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void rfftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, axis_type<DIM> axes,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           shape_type<DIM> s             = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "rfftn: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "rfftn: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "rfftn: OutViewType must be complex");

  fftn(exec_space, in, out, axes, norm, s);
}

/// \brief Inverse of rfftn
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Ouput data (real)
/// \param axes [in] Axes over which FFT is performed
/// \param norm [in] How the normalization is applied (optional)
/// \param s [in] Shape of the transformed axis of the output (optional)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void irfftn(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out, axis_type<DIM> axes,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
            shape_type<DIM> s             = {0}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "irfftn: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "irfftn: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "irfftn: OutViewType must be real");

  ifftn(exec_space, in, out, axes, norm, s);
}

}  // namespace KokkosFFT

#endif
