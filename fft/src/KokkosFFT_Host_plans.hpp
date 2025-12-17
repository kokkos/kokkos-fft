// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HOST_PLANS_HPP
#define KOKKOSFFT_HOST_PLANS_HPP

#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ExecutionSpace, typename T,
          std::enable_if_t<is_AnyHostSpace_v<ExecutionSpace>, std::nullptr_t> =
              nullptr>
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
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::is_same_v<ExecutionSpace,
                                 Kokkos::DefaultHostExecutionSpace>) {
      if constexpr (std::is_same_v<T, float>) {
        fftwf_init_threads();
      } else {
        fftw_init_threads();
      }

      // Register cleanup function as a hook in Kokkos::finalize
      Kokkos::push_finalize_hook([]() {
        if constexpr (std::is_same_v<T, float>) {
          fftwf_cleanup_threads();
        } else {
          fftw_cleanup_threads();
        }
      });
    }
#endif
    return true;
  }();
}

// batched transform, over ND Views
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType, std::size_t fft_rank = 1,
          std::enable_if_t<is_AnyHostSpace_v<ExecutionSpace>, std::nullptr_t> =
              nullptr>
auto create_plan(const ExecutionSpace& exec_space,
                 std::unique_ptr<PlanType>& plan, const InViewType& in,
                 const OutViewType& out, Direction direction,
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

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_plan[TPL_fftw]");
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, axes, s, is_inplace);

  using index_type = FFTIndexType<ExecutionSpace>;
  index_type idist = total_size(in_extents);
  index_type odist = total_size(out_extents);

  auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, in_value_type>::type*>(in.data());
  auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, out_value_type>::type*>(out.data());

  // For the moment, considering the contiguous layout only
  index_type istride = 1, ostride = 1;
  auto int_in_extents  = convert_base_int_type<index_type>(in_extents);
  auto int_out_extents = convert_base_int_type<index_type>(out_extents);
  auto int_fft_extents = convert_base_int_type<index_type>(fft_extents);
  [[maybe_unused]] auto sign =
      KokkosFFT::Impl::direction_type<ExecutionSpace>(direction);

  plan = std::make_unique<PlanType>(
      exec_space, int_fft_extents.size(), int_fft_extents.data(), howmany,
      idata, int_in_extents.data(), istride, idist, odata,
      int_out_extents.data(), ostride, odist, sign, FFTW_ESTIMATE);

  return fft_extents;
}

// batched transform, over ND Views
template <typename ExecutionSpace, typename PlanType, typename InViewType,
          typename OutViewType,
          std::enable_if_t<is_AnyHostSpace_v<ExecutionSpace>, std::nullptr_t> =
              nullptr>
auto create_dynplan(const ExecutionSpace& exec_space,
                    std::unique_ptr<PlanType>& plan, const InViewType& in,
                    const OutViewType& out, Direction direction,
                    std::size_t dim, bool is_inplace) {
  KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(
      (KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                               OutViewType>),
      "create_dynplan");
  using in_value_type  = typename InViewType::non_const_value_type;
  using out_value_type = typename OutViewType::non_const_value_type;

  Kokkos::Profiling::ScopedRegion region("KokkosFFT::create_dynplan[TPL_fftw]");
  auto [in_extents, out_extents, fft_extents, howmany] =
      KokkosFFT::Impl::get_extents(in, out, dim, is_inplace);

  using index_type = FFTIndexType<ExecutionSpace>;
  index_type idist = total_size(in_extents);
  index_type odist = total_size(out_extents);

  auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, in_value_type>::type*>(in.data());
  auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<
      ExecutionSpace, out_value_type>::type*>(out.data());

  // For the moment, considering the contiguous layout only
  index_type istride = 1, ostride = 1;
  auto int_in_extents  = convert_base_int_type<index_type>(in_extents);
  auto int_out_extents = convert_base_int_type<index_type>(out_extents);
  auto int_fft_extents = convert_base_int_type<index_type>(fft_extents);
  auto sign = KokkosFFT::Impl::direction_type<ExecutionSpace>(direction);

  plan = std::make_unique<PlanType>(
      exec_space, int_fft_extents.size(), int_fft_extents.data(), howmany,
      idata, int_in_extents.data(), istride, idist, odata,
      int_out_extents.data(), ostride, odist, sign, FFTW_ESTIMATE);

  return fft_extents;
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
