// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_PLANS_HPP
#define KOKKOSFFT_HIP_PLANS_HPP

#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_HIP_types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_asserts.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ExecutionSpace, typename T,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
void setup() {
  [[maybe_unused]] static bool once = [] {
    if (!(Kokkos::is_initialized() || Kokkos::is_finalized())) {
      Kokkos::abort(
          "Error: KokkosFFT APIs must not be called before initializing "
          "Kokkos.\n");
    }
    if (Kokkos::is_finalized()) {
      Kokkos::abort(
          "Error: KokkosFFT APIs must not be called after finalizing "
          "Kokkos.\n");
    }
    return true;
  }();
}

// 1D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType,
          std::enable_if_t<InViewType::rank() == 1 &&
                               std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction /*direction*/,
                 axis_type<1> axes, shape_type<1> s, bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_plan");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_hipfft]");
  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);
  const int nx = fft_extents.at(0);
  plan         = std::make_unique<PlanType>(nx, type, howmany);
  plan->commit(exec_space);

  return fft_extents;
}

// 2D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType,
          std::enable_if_t<InViewType::rank() == 2 &&
                               std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction /*direction*/,
                 axis_type<2> axes, shape_type<2> s, bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_plan");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_hipfft]");
  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();
  [[maybe_unused]] auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);
  const int nx = fft_extents.at(0), ny = fft_extents.at(1);
  plan = std::make_unique<PlanType>(nx, ny, type);
  plan->commit(exec_space);

  return fft_extents;
}

// 3D transform
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType,
          std::enable_if_t<InViewType::rank() == 3 &&
                               std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction /*direction*/,
                 axis_type<3> axes, shape_type<3> s, bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_plan");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_hipfft]");
  auto type = KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                              out_value_type>::type();
  [[maybe_unused]] auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);

  const int nx = fft_extents.at(0), ny = fft_extents.at(1),
            nz = fft_extents.at(2);
  plan         = std::make_unique<PlanType>(nx, ny, nz, type);
  plan->commit(exec_space);

  return fft_extents;
}

// batched transform, over ND Views
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t fft_rank = 1,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction /*direction*/,
                 axis_type<fft_rank> axes, shape_type<fft_rank> s,
                 bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_plan");
  static_assert(InViewType::rank() >= fft_rank,
                "create_plan: Rank of View must be larger than Rank of FFT.");

  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_hipfft]");
  constexpr auto type =
      KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                      out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);
  int idist = total_size(in_extents);
  int odist = total_size(out_extents);

  auto int_in_extents  = convert_base_int_type<int>(in_extents);
  auto int_out_extents = convert_base_int_type<int>(out_extents);
  auto int_fft_extents = convert_base_int_type<int>(fft_extents);

  // For the moment, considering the contiguous layout only
  int istride = 1, ostride = 1;
  plan = std::make_unique<PlanType>(
      int_fft_extents.size(), int_fft_extents.data(), int_in_extents.data(),
      istride, idist, int_out_extents.data(), ostride, odist, type, howmany);
  plan->commit(exec_space);

  return fft_extents;
}

// Interface for dynamic plans
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType,
          std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                           std::nullptr_t> = nullptr>
auto create_dynplan(const ExecutionSpace& exec_space,
                    std::unique_ptr<PlanType>& plan, const InViewType& in,
                    const OutViewType& out, Direction /*direction*/,
                    std::size_t dim, bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_dynplan");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::create_dynplan[TPL_hipfft]");
  const auto type =
      KokkosFFT::Impl::transform_type<ExecutionSpace, in_value_type,
                                      out_value_type>::type();
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, dim, is_inplace);
  int idist = total_size(in_extents);
  int odist = total_size(out_extents);

  if (dim == 1) {
    const int nx = fft_extents.at(0);
    plan         = std::make_unique<PlanType>(nx, type, howmany);
  } else if (dim == 2 || dim == 3) {
    auto int_fft_extents = convert_base_int_type<int>(fft_extents);
    if (InViewType::rank() == dim) {
      plan = std::make_unique<PlanType>(int_fft_extents, type);
    } else {
      auto int_in_extents  = convert_base_int_type<int>(in_extents);
      auto int_out_extents = convert_base_int_type<int>(out_extents);
      int istride = 1, ostride = 1;
      plan = std::make_unique<PlanType>(
          int_fft_extents.size(), int_fft_extents.data(), int_in_extents.data(),
          istride, idist, int_out_extents.data(), ostride, odist, type,
          howmany);
    }
  } else {
    KOKKOSFFT_THROW_IF(true,
                       "create_dynplan: Only 1D, 2D, and 3D transforms are "
                       "supported for dynamic plans.");
  }

  plan->commit(exec_space);

  return fft_extents;
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
