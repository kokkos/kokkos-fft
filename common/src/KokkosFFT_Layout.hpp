// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_LAYOUT_HPP
#define KOKKOSFFT_LAYOUT_HPP

#include <type_traits>
#include <numeric>
#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Impl {

// Copied from Kokkos_Layout.hpp
// Since this is not publicly exposed, we re-implement it here
// to avoid dependency on Kokkos implementation details
template <typename... Layout>
struct layout_iterate_type_selector {
  static_assert(true,
                "layout_iterate_type_selector: Layout must be one of "
                "LayoutLeft, LayoutRight, LayoutStride");
};

template <>
struct layout_iterate_type_selector<Kokkos::LayoutRight> {
  static constexpr Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Iterate::Right;
  static constexpr Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Iterate::Right;
};

template <>
struct layout_iterate_type_selector<Kokkos::LayoutLeft> {
  static constexpr Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Iterate::Left;
  static constexpr Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Iterate::Left;
};

template <>
struct layout_iterate_type_selector<Kokkos::LayoutStride> {
  static constexpr Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Iterate::Default;
  static constexpr Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Iterate::Default;
};

template <typename Layout, typename IndexType, std::size_t N>
Layout create_layout(const std::array<IndexType, N>& extents) {
  static_assert(std::is_integral_v<IndexType>,
                "create_layout: IndexType must be an integral type");
  static_assert(std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                    std::is_same_v<Layout, Kokkos::LayoutRight>,
                "create_layout: Layout must be either Kokkos::LayoutLeft or "
                "Kokkos::LayoutRight.");
  Layout layout;
  std::copy_n(extents.begin(), N, layout.dimension);
  return layout;
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
