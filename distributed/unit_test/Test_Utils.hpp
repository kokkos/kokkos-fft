// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <utility>
#include <algorithm>
#include <Kokkos_Core.hpp>

template <typename ExecutionSpace, typename AViewType, typename BViewType>
bool allclose(const ExecutionSpace& exec, const AViewType& a,
              const BViewType& b, double rtol = 1.e-5, double atol = 1.e-8) {
  constexpr std::size_t rank = AViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      "KokkosFFT::Test::allclose",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const std::size_t& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  return error == 0;
}

template <typename LayoutType, typename iType, std::size_t Rank>
inline auto get_buffer_shape(const std::array<iType, Rank>& shape,
                             iType nprocs) {
  std::array<iType, Rank + 1> buffer_shape;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    for (std::size_t i = 0; i < Rank; i++) {
      buffer_shape[i] = shape[i];
    }
    buffer_shape[Rank] = nprocs;
  } else {
    buffer_shape[0] = nprocs;
    for (std::size_t i = 0; i < Rank; i++) {
      buffer_shape[i + 1] = shape[i];
    }
  }

  return buffer_shape;
}

template <typename iType>
inline auto distribute_extents(iType n, iType r, iType p) {
  iType base_size = n / p, remainder = n % p;
  iType length = base_size + (r < remainder ? 1 : 0);
  iType start  = r * base_size + std::min(r, remainder);
  return std::pair<iType, iType>{start, length};
};

#endif
