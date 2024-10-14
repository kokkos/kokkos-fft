// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_NORMALIZATION_HPP
#define KOKKOSFFT_NORMALIZATION_HPP

#include <tuple>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ExecutionSpace, typename ViewType, typename T>
void normalize_impl(const ExecutionSpace& exec_space, ViewType& inout,
                    const T coef) {
  std::size_t size = inout.size();
  auto* data       = inout.data();

  Kokkos::parallel_for(
      "KokkosFFT::normalize",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, size),
      KOKKOS_LAMBDA(std::size_t i) { data[i] *= coef; });
}

template <typename ViewType>
auto get_coefficients(ViewType, Direction direction,
                      Normalization normalization, std::size_t fft_size) {
  using value_type = KokkosFFT::Impl::base_floating_point_type<
      typename ViewType::non_const_value_type>;
  value_type coef   = 1;
  bool to_normalize = false;

  switch (normalization) {
    case Normalization::forward:
      if (direction == Direction::forward) {
        coef = static_cast<value_type>(1) / static_cast<value_type>(fft_size);
        to_normalize = true;
      }

      break;
    case Normalization::backward:
      if (direction == Direction::backward) {
        coef = static_cast<value_type>(1) / static_cast<value_type>(fft_size);
        to_normalize = true;
      }

      break;
    case Normalization::ortho:
      coef = static_cast<value_type>(1) /
             Kokkos::sqrt(static_cast<value_type>(fft_size));
      to_normalize = true;

      break;
    default:  // No normalization
      break;
  };
  return std::tuple<value_type, bool>(coef, to_normalize);
}

template <typename ExecutionSpace, typename ViewType>
void normalize(const ExecutionSpace& exec_space, ViewType& inout,
               Direction direction, Normalization normalization,
               std::size_t fft_size) {
  static_assert(KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, ViewType>,
                "normalize: View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "ExecutionSpace must be able to access data in ViewType");
  auto [coef, to_normalize] =
      get_coefficients(inout, direction, normalization, fft_size);
  if (to_normalize) normalize_impl(exec_space, inout, coef);
}

inline auto swap_direction(Normalization normalization) {
  Normalization new_direction = Normalization::none;
  switch (normalization) {
    case Normalization::forward: new_direction = Normalization::backward; break;
    case Normalization::backward: new_direction = Normalization::forward; break;
    case Normalization::ortho: new_direction = Normalization::ortho; break;
    default: break;
  };
  return new_direction;
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif