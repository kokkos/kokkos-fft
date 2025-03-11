// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_TRANSFORM_HPP
#define KOKKOSFFT_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_normalization.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_Plans.hpp"

namespace KokkosFFT {

/// \brief Execute FFT given by the Plan on input and output Views with
/// normalization
///
/// \tparam PlanType: KokkosFFT Plan type
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param plan [in] KokkosFFT Plan to be executed
/// \param in [in] Input data
/// \param out [out] Output data
/// \param norm [in] How the normalization is applied (default, backward)
template <typename PlanType, typename InViewType, typename OutViewType>
void execute(
    const PlanType& plan, const InViewType& in, const OutViewType& out,
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward) {
  plan.execute_impl(in, out, norm);
}

/// \brief One dimensional FFT in forward direction
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axis [in] Axis over which FFT is performed (default, -1)
/// \param n [in] Length of the transformed axis of the output (default,
/// nullopt)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void fft(const ExecutionSpace& exec_space, const InViewType& in,
         const OutViewType& out,
         KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
         int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 1,
                "fft: View rank must be larger than or equal to 1");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::fft");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axis_type<1>({axis})),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward, axis,
                       n);
  plan.execute_impl(in, out, norm);
}

/// \brief One dimensional FFT in backward direction
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axis [in] Axis over which FFT is performed (default, -1)
/// \param n [in] Length of the transformed axis of the output (default,
/// nullopt)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ifft(const ExecutionSpace& exec_space, const InViewType& in,
          const OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ifft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 1,
                "ifft: View rank must be larger than or equal to 1");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::ifft");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axis_type<1>({axis})),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::backward,
                       axis, n);
  plan.execute_impl(in, out, norm);
}

/// \brief One dimensional FFT for real input
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axis [in] Axis over which FFT is performed (default, -1)
/// \param n [in] Length of the transformed axis of the output (default,
/// nullopt)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void rfft(const ExecutionSpace& exec_space, const InViewType& in,
          const OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "rfft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 1,
                "rfft: View rank must be larger than or equal to 1");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "rfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "rfft: OutViewType must be complex");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::rfft");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axis_type<1>({axis})),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward, axis,
                       n);
  plan.execute_impl(in, out, norm);
}

/// \brief Inverse of rfft
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (real)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axis [in] Axis over which FFT is performed (default, -1)
/// \param n [in] Length of the transformed axis of the output (default,
/// nullopt)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void irfft(const ExecutionSpace& exec_space, const InViewType& in,
           const OutViewType& out,
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
  static_assert(InViewType::rank() >= 1,
                "irfft: View rank must be larger than or equal to 1");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "irfft: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "irfft: OutViewType must be real");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::irfft");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axis_type<1>({axis})),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::backward,
                       axis, n);
  plan.execute_impl(in, out, norm);
}

/// \brief One dimensional FFT of a signal that has Hermitian symmetry
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (real)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axis [in] Axis over which FFT is performed (default, -1)
/// \param n [in] Length of the transformed axis of the output (default,
/// nullopt)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void hfft(const ExecutionSpace& exec_space, const InViewType& in,
          const OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "hfft: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 1,
                "hfft: View rank must be larger than or equal to 1");

  // [TO DO]
  // allow real type as input, need to obtain complex view type from in view
  // type
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "hfft: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "hfft: OutViewType must be real");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::hfft");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axis_type<1>({axis})),
                     "axes are invalid for in/out views");
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
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axis [in] Axis over which FFT is performed (default, -1)
/// \param n [in] Length of the transformed axis of the output (default,
/// nullopt)
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ihfft(const ExecutionSpace& exec_space, const InViewType& in,
           const OutViewType& out,
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
  static_assert(InViewType::rank() >= 1,
                "ihfft: View rank must be larger than or equal to 1");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "ihfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "ihfft: OutViewType must be complex");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::ihfft");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axis_type<1>({axis})),
                     "axes are invalid for in/out views");
  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  OutViewType out_conj;
  rfft(exec_space, in, out, new_norm, axis, n);
  KokkosFFT::Impl::conjugate(exec_space, out, out_conj);
  Kokkos::deep_copy(exec_space, out, out_conj);
}

// 2D FFT

/// \brief Two dimensional FFT in forward direction
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axes [in] Axes over which FFT is performed (default, {-2, -1})
/// \param s [in] Shape of the transformed axis of the output (default, {})
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void fft2(const ExecutionSpace& exec_space, const InViewType& in,
          const OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
          axis_type<2> axes = {-2, -1}, shape_type<2> s = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fft2: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 2,
                "fft2: View rank must be larger than or equal to 2");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::fft2");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward, axes,
                       s);
  plan.execute_impl(in, out, norm);
}

/// \brief Two dimensional FFT in backward direction
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axes [in] Axes over which FFT is performed (default, {-2, -1})
/// \param s [in] Shape of the transformed axis of the output (default, {})
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ifft2(const ExecutionSpace& exec_space, const InViewType& in,
           const OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           axis_type<2> axes = {-2, -1}, shape_type<2> s = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ifft2: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 2,
                "ifft2: View rank must be larger than or equal to 2");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::ifft2");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::backward,
                       axes, s);
  plan.execute_impl(in, out, norm);
}

/// \brief Two dimensional FFT for real input
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Output data (complex)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axes [in] Axes over which FFT is performed (default, {-2, -1})
/// \param s [in] Shape of the transformed axis of the output (default, {})
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void rfft2(const ExecutionSpace& exec_space, const InViewType& in,
           const OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
           axis_type<2> axes = {-2, -1}, shape_type<2> s = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "rfft2: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 2,
                "rfft2: View rank must be larger than or equal to 2");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "rfft2: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "rfft2: OutViewType must be complex");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::rfft2");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward, axes,
                       s);
  plan.execute_impl(in, out, norm);
}

/// \brief Inverse of rfft2
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (real)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param axes [in] Axes over which FFT is performed (default, {-2, -1})
/// \param s [in] Shape of the transformed axis of the output (default, {})
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void irfft2(const ExecutionSpace& exec_space, const InViewType& in,
            const OutViewType& out,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
            axis_type<2> axes = {-2, -1}, shape_type<2> s = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "irfft2: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(InViewType::rank() >= 2,
                "irfft2: View rank must be larger than or equal to 2");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "irfft2: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "irfft2: OutViewType must be real");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::irfft2");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::backward,
                       axes, s);
  plan.execute_impl(in, out, norm);
}

// ND FFT

/// \brief N-dimensional FFT in forward direction
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
/// \tparam DIM: The dimensionality of the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (complex)
/// \param axes [in] Axes over which FFT is performed (default, all axes)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param s [in] Shape of the transformed axis of the output (default, {})
#if defined(DOXY)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
#else
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = InViewType::rank()>
#endif
void fftn(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out,
    axis_type<DIM> axes =
        KokkosFFT::Impl::index_sequence<int, DIM, -static_cast<int>(DIM)>(),
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
    shape_type<DIM> s             = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "fftn: InViewType and OutViewType must have the same base floating point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(DIM >= 1 && DIM <= KokkosFFT::MAX_FFT_DIM,
                "fftn: the Rank of FFT axes must be between 1 and MAX_FFT_DIM");
  static_assert(
      InViewType::rank() >= DIM,
      "fftn: View rank must be larger than or equal to the Rank of FFT axes");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::fftn");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward, axes,
                       s);
  plan.execute_impl(in, out, norm);
}

/// \brief Inverse of fftn
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
/// \tparam DIM: The dimensionality of the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (complex)
/// \param axes [in] Axes over which FFT is performed (default, all axes)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param s [in] Shape of the transformed axis of the output (default, {})
#if defined(DOXY)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
#else
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = InViewType::rank()>
#endif
void ifftn(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out,
    axis_type<DIM> axes =
        KokkosFFT::Impl::index_sequence<int, DIM, -static_cast<int>(DIM)>(),
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
    shape_type<DIM> s             = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "ifftn: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(
      DIM >= 1 && DIM <= KokkosFFT::MAX_FFT_DIM,
      "ifftn: the Rank of FFT axes must be between 1 and MAX_FFT_DIM");
  static_assert(
      InViewType::rank() >= DIM,
      "ifftn: View rank must be larger than or equal to the Rank of FFT axes");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::ifftn");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::backward,
                       axes, s);
  plan.execute_impl(in, out, norm);
}

/// \brief N-dimensional FFT for real input
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
/// \tparam DIM: The dimensionality of the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (real)
/// \param out [out] Output data (complex)
/// \param axes [in] Axes over which FFT is performed (default, all axes)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param s [in] Shape of the transformed axis of the output (default, {})
#if defined(DOXY)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
#else
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = InViewType::rank()>
#endif
void rfftn(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out,
    axis_type<DIM> axes =
        KokkosFFT::Impl::index_sequence<int, DIM, -static_cast<int>(DIM)>(),
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
    shape_type<DIM> s             = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "rfftn: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(
      DIM >= 1 && DIM <= KokkosFFT::MAX_FFT_DIM,
      "rfftn: the Rank of FFT axes must be between 1 and MAX_FFT_DIM");
  static_assert(
      InViewType::rank() >= DIM,
      "rfftn: View rank must be larger than or equal to the Rank of FFT axes");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_real_v<in_value_type>,
                "rfftn: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "rfftn: OutViewType must be complex");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::rfftn");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::forward, axes,
                       s);
  plan.execute_impl(in, out, norm);
}

/// \brief Inverse of rfftn
///
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam InViewType: Input View type for the fft
/// \tparam OutViewType: Output View type for the fft
/// \tparam DIM: The dimensionality of the fft
///
/// \param exec_space [in] Kokkos execution space
/// \param in [in] Input data (complex)
/// \param out [out] Output data (real)
/// \param axes [in] Axes over which FFT is performed (default, all axes)
/// \param norm [in] How the normalization is applied (default, backward)
/// \param s [in] Shape of the transformed axis of the output (default, {})
#if defined(DOXY)
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM>
#else
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = InViewType::rank()>
#endif
void irfftn(
    const ExecutionSpace& exec_space, const InViewType& in,
    const OutViewType& out,
    axis_type<DIM> axes =
        KokkosFFT::Impl::index_sequence<int, DIM, -static_cast<int>(DIM)>(),
    KokkosFFT::Normalization norm = KokkosFFT::Normalization::backward,
    shape_type<DIM> s             = {}) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "irfftn: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");
  static_assert(
      DIM >= 1 && DIM <= KokkosFFT::MAX_FFT_DIM,
      "irfftn: the Rank of FFT axes must be between 1 and MAX_FFT_DIM");
  static_assert(
      InViewType::rank() >= DIM,
      "irfftn: View rank must be larger than or equal to the Rank of FFT axes");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex_v<in_value_type>,
                "irfftn: InViewType must be complex");
  static_assert(KokkosFFT::Impl::is_real_v<out_value_type>,
                "irfftn: OutViewType must be real");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::irfftn");
  KOKKOSFFT_THROW_IF(!KokkosFFT::Impl::are_valid_axes(in, axes),
                     "axes are invalid for in/out views");
  KokkosFFT::Plan plan(exec_space, in, out, KokkosFFT::Direction::backward,
                       axes, s);
  plan.execute_impl(in, out, norm);
}
}  // namespace KokkosFFT

#endif
