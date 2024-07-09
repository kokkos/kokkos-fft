// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_PLANS_HPP
#define KOKKOSFFT_HIP_PLANS_HPP

#include <numeric>
#include "KokkosFFT_HIP_types.hpp"
#include "KokkosFFT_layouts.hpp"

namespace KokkosFFT {
namespace Impl {
// 1D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename BufferViewType, typename InfoType,
          std::enable_if_t<InViewType::rank() == 1 &&
                               std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, BufferViewType&, InfoType&,
                 Direction /*direction*/, axis_type<1> axes, shape_type<1> s) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: OutViewType is not a Kokkos::View.");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  plan                   = std::make_unique<PlanType>();
  hipfftResult hipfft_rt = hipfftCreate(&(*plan));
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftCreate failed");

  hipStream_t stream = exec_space.hip_stream();
  hipfftSetStream((*plan), stream);

  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s);
  const int nx = fft_extents.at(0);
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());

  hipfft_rt = hipfftPlan1d(&(*plan), nx, type, howmany);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftPlan1d failed");
  return fft_size;
}

// 2D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename BufferViewType, typename InfoType,
          std::enable_if_t<InViewType::rank() == 2 &&
                               std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, BufferViewType&, InfoType&,
                 Direction /*direction*/, axis_type<2> axes, shape_type<2> s) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: OutViewType is not a Kokkos::View.");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  plan                   = std::make_unique<PlanType>();
  hipfftResult hipfft_rt = hipfftCreate(&(*plan));
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftCreate failed");

  hipStream_t stream = exec_space.hip_stream();
  hipfftSetStream((*plan), stream);

  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();
  [[maybe_unused]] auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s);
  const int nx = fft_extents.at(0), ny = fft_extents.at(1);
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());

  hipfft_rt = hipfftPlan2d(&(*plan), nx, ny, type);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftPlan2d failed");
  return fft_size;
}

// 3D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename BufferViewType, typename InfoType,
          std::enable_if_t<InViewType::rank() == 3 &&
                               std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, BufferViewType&, InfoType&,
                 Direction /*direction*/, axis_type<3> axes, shape_type<3> s) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: OutViewType is not a Kokkos::View.");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  plan                   = std::make_unique<PlanType>();
  hipfftResult hipfft_rt = hipfftCreate(&(*plan));
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftCreate failed");

  hipStream_t stream = exec_space.hip_stream();
  hipfftSetStream((*plan), stream);

  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();
  [[maybe_unused]] auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s);

  const int nx = fft_extents.at(0), ny = fft_extents.at(1),
            nz = fft_extents.at(2);
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());

  hipfft_rt = hipfftPlan3d(&(*plan), nx, ny, nz, type);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftPlan3d failed");
  return fft_size;
}

// batched transform, over ND Views
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, typename BufferViewType, typename InfoType,
          std::size_t fft_rank             = 1,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, BufferViewType&, InfoType&,
                 Direction /*direction*/, axis_type<fft_rank> axes,
                 shape_type<fft_rank> s) {
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: InViewType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::create_plan: OutViewType is not a Kokkos::View.");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  static_assert(
      InViewType::rank() >= fft_rank,
      "KokkosFFT::create_plan: Rank of View must be larger than Rank of FFT.");
  const int rank = fft_rank;
  constexpr auto type =
      KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                      out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s);
  int idist    = std::accumulate(in_extents.begin(), in_extents.end(), 1,
                              std::multiplies<>());
  int odist    = std::accumulate(out_extents.begin(), out_extents.end(), 1,
                              std::multiplies<>());
  int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1,
                                 std::multiplies<>());

  // For the moment, considering the contiguous layout only
  int istride = 1, ostride = 1;

  plan                   = std::make_unique<PlanType>();
  hipfftResult hipfft_rt = hipfftCreate(&(*plan));
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftCreate failed");

  hipStream_t stream = exec_space.hip_stream();
  hipfftSetStream((*plan), stream);

  hipfft_rt = hipfftPlanMany(&(*plan), rank, fft_extents.data(),
                             in_extents.data(), istride, idist,
                             out_extents.data(), ostride, odist, type, howmany);

  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftPlan failed");
  return fft_size;
}

template <typename ExecutionSpace, typename PlanType,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
void destroy_plan(std::unique_ptr<PlanType>& plan) {
  hipfftDestroy(*plan);
}

template <typename ExecutionSpace, typename InfoType,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
void destroy_info(InfoType&) {
  // not used, no finalization is required
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif