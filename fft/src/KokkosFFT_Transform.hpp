#ifndef KOKKOSFFT_TRANSFORM_HPP
#define KOKKOSFFT_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_normalization.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_padding.hpp"
#include "KokkosFFT_Plans.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
using default_device = Kokkos::Cuda;
#include "KokkosFFT_Cuda_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_HIP)
using default_device = Kokkos::HIP;
#include "KokkosFFT_HIP_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_SYCL)
#include "KokkosFFT_SYCL_transform.hpp"
#ifdef ENABLE_HOST_AND_DEVICE
#include "KokkosFFT_OpenMP_transform.hpp"
#endif
#elif defined(KOKKOS_ENABLE_OPENMP)
using default_device = Kokkos::OpenMP;
#include "KokkosFFT_OpenMP_transform.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
using default_device = Kokkos::Threads;
#include "KokkosFFT_OpenMP_transform.hpp"
#else
using default_device = Kokkos::Serial;
#include "KokkosFFT_OpenMP_transform.hpp"
#endif

// General Transform Interface
namespace KokkosFFT {
namespace Impl {
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
void _fft(const ExecutionSpace& exec_space, PlanType& plan,
          const InViewType& in, OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "_fft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "_fft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "_fft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "_fft: OutViewType must be either LayoutLeft or LayoutRight.");

  static_assert(InViewType::rank() == OutViewType::rank(),
                "_fft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "_fft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "_fft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "_fft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, in_value_type>::type*>(in.data());
  auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, out_value_type>::type*>(out.data());

  auto forward =
      direction_type<ExecutionSpace>(KokkosFFT::Impl::Direction::Forward);
  KokkosFFT::Impl::_exec(plan.plan(), idata, odata, forward);
  KokkosFFT::Impl::normalize(exec_space, out,
                             KokkosFFT::Impl::Direction::Forward, norm,
                             plan.fft_size());
}

template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType>
void _ifft(const ExecutionSpace& exec_space, PlanType& plan,
           const InViewType& in, OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "_ifft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "_ifft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "_ifft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "_ifft: OutViewType must be either LayoutLeft or LayoutRight.");

  static_assert(InViewType::rank() == OutViewType::rank(),
                "_ifft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "_ifft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "_ifft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "_ifft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, in_value_type>::type*>(in.data());
  auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, out_value_type>::type*>(out.data());

  auto backward =
      direction_type<ExecutionSpace>(KokkosFFT::Impl::Direction::Backward);
  KokkosFFT::Impl::_exec(plan.plan(), idata, odata, backward);
  KokkosFFT::Impl::normalize(exec_space, out,
                             KokkosFFT::Impl::Direction::Backward, norm,
                             plan.fft_size());
}
}  // namespace Impl
}  // namespace KokkosFFT

namespace KokkosFFT {
/// \brief One dimensional FFT in forward direction
///
/// \param in [in] ExecutionSpace
///        Kokkos execution space for this plan
/// \param in [in] Kokkos::View
///        Input data
/// \param out [out] Kokkos::View
///        Ouput data
/// \param norm [in] enum Normalization {backward, ortho, forward, none},
/// optional
///        How the normalization is applied.
/// \param axis [in] int, optional
///        Axis over which FFT is performed
/// \param n [in] size_t, optional
///        Length of the transformed axis of the output.
template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void fft(const ExecutionSpace& exec_space, const InViewType& in,
         OutViewType& out,
         KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
         int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fft: execution_space cannot access data in OutViewType");

  InViewType _in;
  if (n) {
    std::size_t _n = n.value();
    auto modified_shape =
        KokkosFFT::Impl::get_modified_shape(in, shape_type<1>({_n}));
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Forward, axis);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

/// \brief One dimensional FFT in forward direction
///
/// \param in [in] ExecutionSpace \n
///        Kokkos execution space for this plan
/// \param in [in] Kokkos::View \n
///        Input data
/// \param out [out] Kokkos::View \n
///        Ouput data
/// \param in [in] PlanType \n
///        KokkosFFT Plan for forward fft
/// \param norm [in] enum Normalization {backward, ortho, forward, none},
/// optional \n
///        How the normalization is applied.
/// \param axis [in] int, optional \n
///        Axis over which FFT is performed
/// \param n [in] size_t, optional \n
///        Length of the transformed axis of the output.
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType>
void fft(const ExecutionSpace& exec_space, const InViewType& in,
         OutViewType& out, const PlanType& plan,
         KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
         int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fft: execution_space cannot access data in OutViewType");

  InViewType _in;
  if (n) {
    std::size_t _n = n.value();
    auto modified_shape =
        KokkosFFT::Impl::get_modified_shape(in, shape_type<1>({_n}));
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  plan.template good<ExecutionSpace, InViewType, OutViewType>(
      _in, out, KokkosFFT::Impl::Direction::Forward, axis_type<1>{axis});

  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ifft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifft: execution_space cannot access data in OutViewType");

  InViewType _in;
  // [TO DO] Modify crop_or_pad to perform the following lines
  // KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, n);
  if (n) {
    std::size_t _n = n.value();
    auto modified_shape =
        KokkosFFT::Impl::get_modified_shape(in, shape_type<1>({_n}));
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Backward, axis);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType>
void ifft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, const PlanType& plan,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifft: execution_space cannot access data in OutViewType");

  InViewType _in;
  // [TO DO] Modify crop_or_pad to perform the following lines
  // KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, n);
  if (n) {
    std::size_t _n = n.value();
    auto modified_shape =
        KokkosFFT::Impl::get_modified_shape(in, shape_type<1>({_n}));
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  plan.template good<ExecutionSpace, InViewType, OutViewType>(
      _in, out, KokkosFFT::Impl::Direction::Backward, axis_type<1>{axis});

  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void rfft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfft: OutViewType must be complex");

  fft(exec_space, in, out, norm, axis, n);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType>
void rfft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, const PlanType& plan,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfft: OutViewType must be complex");

  fft(exec_space, in, out, plan, norm, axis, n);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void irfft(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "irfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rifft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "irfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "irfft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfft: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfft: OutViewType must be real");
  if (n) {
    std::size_t _n = n.value() / 2 + 1;
    ifft(exec_space, in, out, norm, axis, _n);
  } else {
    ifft(exec_space, in, out, norm, axis);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType>
void irfft(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, const PlanType& plan,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfft: OutViewType is not a Kokkos::View.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "irfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "irfft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfft: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfft: OutViewType must be real");
  if (n) {
    std::size_t _n = n.value() / 2 + 1;
    ifft(exec_space, in, out, plan, norm, axis, _n);
  } else {
    ifft(exec_space, in, out, plan, norm, axis);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void hfft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "hfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "hfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "hfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "hfft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "hfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "hfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "hfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "hfft: execution_space cannot access data in OutViewType");

  // [TO DO]
  // allow real type as input, need to obtain complex view type from in view
  // type
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "hfft: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "hfft: OutViewType must be real");
  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  // using ComplexViewType = typename
  // KokkosFFT::Impl::complex_view_type<ExecutionSpace, InViewType>::type;
  // ComplexViewType in_conj;
  InViewType in_conj;
  KokkosFFT::Impl::conjugate(exec_space, in, in_conj);
  irfft(exec_space, in_conj, out, new_norm, axis, n);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType>
void hfft(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, const PlanType& plan,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "hfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "hfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "hfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "hfft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "hfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "hfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "hfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "hfft: execution_space cannot access data in OutViewType");

  // [TO DO]
  // allow real type as input, need to obtain complex view type from in view
  // type
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "hfft: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "hfft: OutViewType must be real");
  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  // using ComplexViewType = typename
  // KokkosFFT::Impl::complex_view_type<ExecutionSpace, InViewType>::type;
  // ComplexViewType in_conj;
  InViewType in_conj;
  KokkosFFT::Impl::conjugate(exec_space, in, in_conj);
  irfft(exec_space, in_conj, out, plan, new_norm, axis, n);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void ihfft(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ihfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ihfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ihfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ihfft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ihfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ihfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ihfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ihfft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(std::is_floating_point<in_value_type>::value,
                "ihfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "ihfft: OutViewType must be complex");

  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  OutViewType out_conj;
  rfft(exec_space, in, out, new_norm, axis, n);
  KokkosFFT::Impl::conjugate(exec_space, out, out_conj);
  out = out_conj;
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType>
void ihfft(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, const PlanType& plan,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           int axis = -1, std::optional<std::size_t> n = std::nullopt) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ihfft: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ihfft: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ihfft: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ihfft: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ihfft: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ihfft: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ihfft: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ihfft: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(std::is_floating_point<in_value_type>::value,
                "ihfft: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "ihfft: OutViewType must be complex");

  auto new_norm = KokkosFFT::Impl::swap_direction(norm);
  OutViewType out_conj;
  rfft(exec_space, in, out, plan, new_norm, axis, n);
  KokkosFFT::Impl::conjugate(exec_space, out, out_conj);
  out = out_conj;
}

// 2D FFT
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void fft2(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fft2: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Forward, axes);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM = 1>
void fft2(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, const PlanType& plan,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fft2: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  plan.template good<ExecutionSpace, InViewType, OutViewType>(
      _in, out, KokkosFFT::Impl::Direction::Forward, axes);

  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void ifft2(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifft2: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Backward, axes);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM = 1>
void ifft2(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, const PlanType& plan,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifft2: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  plan.template good<ExecutionSpace, InViewType, OutViewType>(
      _in, out, KokkosFFT::Impl::Direction::Backward, axes);

  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void rfft2(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfft2: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfft2: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfft2: OutViewType must be complex");

  fft2(exec_space, in, out, norm, axes, s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM = 1>
void rfft2(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, const PlanType& plan,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfft2: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfft2: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfft2: OutViewType must be complex");

  fft2(exec_space, in, out, plan, norm, axes, s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void irfft2(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
            axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "irfft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(
      KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
      "irfft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "irfft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "irfft2: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfft2: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfft2: OutViewType must be real");

  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  shape_type<DIM> _s    = {0};
  if (s != zeros) {
    for (int i = 0; i < DIM; i++) {
      _s.at(i) = s.at(i) / 2 + 1;
    }
  }

  ifft2(exec_space, in, out, norm, axes, _s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM = 1>
void irfft2(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out, const PlanType& plan,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
            axis_type<2> axes = {-2, -1}, shape_type<DIM> s = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfft2: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfft2: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "irfft2: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(
      KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
      "irfft2: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfft2: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfft2: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "irfft2: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "irfft2: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfft2: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfft2: OutViewType must be real");

  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  shape_type<DIM> _s    = {0};
  if (s != zeros) {
    for (int i = 0; i < DIM; i++) {
      _s.at(i) = s.at(i) / 2 + 1;
    }
  }

  ifft2(exec_space, in, out, plan, norm, axes, _s);
}

// ND FFT
template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void fftn(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          shape_type<DIM> s             = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fftn: execution_space cannot access data in OutViewType");

  // Create a default sequence of axes {-rank, -(rank-1), ..., -1}
  constexpr std::size_t rank = InViewType::rank();
  constexpr int start        = -static_cast<int>(rank);
  axis_type<rank> axes       = KokkosFFT::Impl::index_sequence<rank>(start);

  InViewType _in;
  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Forward, axes);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void fftn(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, axis_type<DIM1> axes,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fftn: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM2> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Forward, axes);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void fftn(const ExecutionSpace& exec_space, const InViewType& in,
          OutViewType& out, const PlanType& plan, axis_type<DIM1> axes,
          KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
          shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "fftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "fftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "fftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "fftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "fftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "fftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "fftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "fftn: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM2> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  plan.template good<ExecutionSpace, InViewType, OutViewType>(
      _in, out, KokkosFFT::Impl::Direction::Forward, axes);

  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_fft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void ifftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           shape_type<DIM> s             = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in OutViewType");

  // Create a default sequence of axes {-rank, -(rank-1), ..., -1}
  constexpr std::size_t rank = InViewType::rank();
  constexpr int start        = -static_cast<int>(rank);
  axis_type<rank> axes       = KokkosFFT::Impl::index_sequence<rank>(start);

  InViewType _in;
  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Backward, axes);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void ifftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, axis_type<DIM1> axes,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM2> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  KokkosFFT::Impl::Plan plan(exec_space, _in, out,
                             KokkosFFT::Impl::Direction::Backward, axes);
  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void ifftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, const PlanType& plan, axis_type<DIM1> axes,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "ifftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "ifftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "ifftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "ifftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "ifftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "ifftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in OutViewType");

  InViewType _in;
  shape_type<DIM2> zeros = {0};  // default shape means no crop or pad
  if (s != zeros) {
    auto modified_shape = KokkosFFT::Impl::get_modified_shape(in, s);
    if (KokkosFFT::Impl::is_crop_or_pad_needed(in, modified_shape)) {
      KokkosFFT::Impl::crop_or_pad(exec_space, in, _in, modified_shape);
    } else {
      _in = in;
    }
  } else {
    _in = in;
  }

  plan.template good<ExecutionSpace, InViewType, OutViewType>(
      _in, out, KokkosFFT::Impl::Direction::Backward, axes);

  if (plan.is_transpose_needed()) {
    InViewType in_T;
    OutViewType out_T;

    KokkosFFT::Impl::transpose(exec_space, _in, in_T, plan.map());
    KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

    KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

    KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
  } else {
    KokkosFFT::Impl::_ifft(exec_space, plan, _in, out, norm);
  }
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void rfftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           shape_type<DIM> s             = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfftn: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfftn: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfftn: OutViewType must be complex");

  fftn(exec_space, in, out, norm, s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void rfftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, const PlanType& plan, axis_type<DIM1> axes,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfftn: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfftn: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfftn: OutViewType must be complex");

  fftn(exec_space, in, out, plan, axes, norm, s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void rfftn(const ExecutionSpace& exec_space, const InViewType& in,
           OutViewType& out, axis_type<DIM1> axes,
           KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
           shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "rfftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "rfftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "rfftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
                "rfftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "rfftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "rfftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "rfftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "rfftn: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(std::is_floating_point<in_value_type>::value,
                "rfftn: InViewType must be real");
  static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                "rfftn: OutViewType must be complex");

  fftn(exec_space, in, out, axes, norm, s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM = 1>
void irfftn(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
            shape_type<DIM> s             = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "irfftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(
      KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
      "irfftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "irfftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "irfftn: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfftn: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfftn: OutViewType must be real");

  shape_type<DIM> zeros = {0};  // default shape means no crop or pad
  shape_type<DIM> _s    = {0};
  if (s != zeros) {
    for (int i = 0; i < DIM; i++) {
      _s.at(i) = s.at(i) / 2 + 1;
    }
  }

  ifftn(exec_space, in, out, norm, _s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void irfftn(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out, axis_type<DIM1> axes,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
            shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "irfftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(
      KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
      "irfftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "irfftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "irfftn: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfftn: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfftn: OutViewType must be real");

  shape_type<DIM2> zeros = {0};  // default shape means no crop or pad
  shape_type<DIM2> _s    = {0};
  if (s != zeros) {
    for (int i = 0; i < DIM2; i++) {
      _s.at(i) = s.at(i) / 2 + 1;
    }
  }

  ifftn(exec_space, in, out, axes, norm, _s);
}

template <typename ExecutionSpace, typename InViewType, typename OutViewType,
          typename PlanType, std::size_t DIM1 = 1, std::size_t DIM2 = 1>
void irfftn(const ExecutionSpace& exec_space, const InViewType& in,
            OutViewType& out, const PlanType& plan, axis_type<DIM1> axes,
            KokkosFFT::Normalization norm = KokkosFFT::Normalization::BACKWARD,
            shape_type<DIM2> s            = {0}) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "irfftn: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<OutViewType>::value,
                "irfftn: OutViewType is not a Kokkos::View.");
  static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<InViewType>,
                "irfftn: InViewType must be either LayoutLeft or LayoutRight.");
  static_assert(
      KokkosFFT::Impl::is_layout_left_or_right_v<OutViewType>,
      "irfftn: OutViewType must be either LayoutLeft or LayoutRight.");
  static_assert(InViewType::rank() == OutViewType::rank(),
                "irfftn: InViewType and OutViewType must have "
                "the same rank.");
  static_assert(std::is_same_v<typename InViewType::array_layout,
                               typename OutViewType::array_layout>,
                "irfftn: InViewType and OutViewType must have "
                "the same Layout.");

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in InViewType");
  static_assert(
      Kokkos::SpaceAccessibility<
          ExecutionSpace, typename OutViewType::memory_space>::accessible,
      "ifftn: execution_space cannot access data in OutViewType");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                "irfftn: InViewType must be complex");
  static_assert(std::is_floating_point<out_value_type>::value,
                "irfftn: OutViewType must be real");

  shape_type<DIM2> zeros = {0};  // default shape means no crop or pad
  shape_type<DIM2> _s    = {0};
  if (s != zeros) {
    for (int i = 0; i < DIM2; i++) {
      _s.at(i) = s.at(i) / 2 + 1;
    }
  }

  ifftn(exec_space, in, out, plan, axes, norm, _s);
}
}  // namespace KokkosFFT

#endif