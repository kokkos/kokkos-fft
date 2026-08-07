// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef MATH_UTILS_HPP
#define MATH_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace Math {
namespace Impl {}  // namespace Impl

template <typename ExecutionSpace, typename RealType>
auto linspace(const ExecutionSpace&, const RealType start, const RealType stop,
              std::size_t num = 50, bool endpoint = true) {
  static_assert(KokkosFFT::Impl::is_real_v<RealType>,
                "linspace: start and stop must be float or double");
  KOKKOSFFT_THROW_IF(num == 0, "Number of elements must be larger than 0");
  using ViewType = Kokkos::View<RealType*, ExecutionSpace>;

  std::size_t length = endpoint ? (num - 1) : num;
  RealType delta     = (stop - start) / static_cast<RealType>(length);
  ViewType result("linspace", num);

  auto h_result = Kokkos::create_mirror_view(result);
  for (std::size_t i = 0; i < length; i++) {
    h_result(i) = start + delta * static_cast<RealType>(i);
  }
  if (endpoint) h_result(length) = stop;
  Kokkos::deep_copy(result, h_result);

  return result;
}
}  // namespace Math

#endif
