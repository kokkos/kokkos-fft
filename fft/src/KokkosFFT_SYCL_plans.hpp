// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_PLANS_HPP
#define KOKKOSFFT_SYCL_PLANS_HPP

#include <numeric>
#include <algorithm>
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250100
#include <oneapi/mkl/dft.hpp>
#else
#include <oneapi/mkl/dfti.hpp>
#endif
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_SYCL_types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

template <
    typename ExecutionSpace, typename T,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
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

// Helper to compute strides from extents
// (n0, n1) -> (0, n1, 1)
// (n0) -> (0, 1)
template <typename InType, typename OutType>
auto compute_strides(std::vector<InType>& extents) -> std::vector<OutType> {
  std::vector<OutType> out;

  OutType stride = 1;
  for (auto it = extents.rbegin(); it != extents.rend(); ++it) {
    out.push_back(stride);
    stride *= static_cast<OutType>(*it);
  }
  out.push_back(0);
  std::reverse(out.begin(), out.end());

  return out;
}

// batched transform, over ND Views
template <
    typename ExecutionSpace, typename PlanType, typename InViewType,
    typename OutViewType, std::size_t fft_rank = 1,
    std::enable_if_t<std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
                     std::nullptr_t> = nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction direction,
                 axis_type<fft_rank> axes, shape_type<fft_rank> s,
                 bool is_inplace) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "create_plan: InViewType and OutViewType must have the same base "
      "floating point type (float/double), the same layout "
      "(LayoutLeft/LayoutRight), "
      "and the same rank. ExecutionSpace must be accessible to the data in "
      "InViewType and OutViewType.");

  static_assert(
      InViewType::rank() >= fft_rank,
      "KokkosFFT::create_plan: Rank of View must be larger than Rank of FFT.");

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_oneMKL]");
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);
  int idist    = total_size(in_extents);
  int odist    = total_size(out_extents);
  int fft_size = total_size(fft_extents);

  // Create plan
  auto in_strides        = compute_strides<int, std::int64_t>(in_extents);
  auto out_strides       = compute_strides<int, std::int64_t>(out_extents);
  auto int64_fft_extents = convert_base_int_type<std::int64_t>(fft_extents);

  auto fwd_strides = direction == Direction::forward ? in_strides : out_strides;
  auto bwd_strides = direction == Direction::forward ? out_strides : in_strides;

  // In oneMKL, the distance is always defined based on R2C transform
  std::int64_t max_idist = static_cast<std::int64_t>(std::max(idist, odist));
  std::int64_t max_odist = static_cast<std::int64_t>(std::min(idist, odist));

  plan = std::make_unique<PlanType>(int64_fft_extents);
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250100
  const oneapi::mkl::dft::config_value placement =
      is_inplace ? oneapi::mkl::dft::config_value::INPLACE
                 : oneapi::mkl::dft::config_value::NOT_INPLACE;
  const oneapi::mkl::dft::config_value storage =
      oneapi::mkl::dft::config_value::COMPLEX_COMPLEX;
  plan->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
  plan->set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
  plan->set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE, storage);
#else
  const DFTI_CONFIG_VALUE placement =
      is_inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE;
  const DFTI_CONFIG_VALUE storage = DFTI_COMPLEX_COMPLEX;
  plan->set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                  in_strides.data());
  plan->set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                  out_strides.data());
  plan->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                  storage);
#endif

  // Configuration for batched plan
  plan->set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, max_idist);
  plan->set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, max_odist);
  plan->set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                  static_cast<std::int64_t>(howmany));

  // Data layout in conjugate-even domain
  plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT, placement);
  sycl::queue q = exec_space.sycl_queue();
  plan->commit(q);

  return fft_size;
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
