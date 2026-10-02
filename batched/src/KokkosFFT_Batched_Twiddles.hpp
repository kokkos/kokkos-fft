#ifndef KOKKOSFFT_BATCHED_TWIDDLES_HPP
#define KOKKOSFFT_BATCHED_TWIDDLES_HPP

#include "KokkosFFT_Batched_Factorize.hpp"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief Host-side tables of a 1D mixed-radix Stockham FFT. Only forward
/// twiddles are stored; the backward transform uses their conjugates.
///
/// Stage s has radix r and current length len (len = n / r_0 / ... / r_{s-1}),
/// and m = len / r. Its twiddles w_len^{j p} (j = 1..r-1, p = 0..m-1) are
/// stored p-major at twiddles[tw_offset[s] + p (r-1) + (j-1)], so the total
/// size is sum_s (len_s - len_{s+1}) = n - 1.
/// A generic-radix stage also needs the r-th roots of unity, stored at
/// roots[root_offset[s] + t], t = 0..r-1 (root_offset is 0 for other stages).
template <typename ComplexType> struct StageTables {
  std::vector<std::size_t> tw_offset;
  std::vector<std::size_t> root_offset;
  std::vector<ComplexType> twiddles;
  std::vector<ComplexType> roots;
};

/// \brief exp(-2 pi i num / den), evaluated in double and cast
template <typename ComplexType>
ComplexType forward_root(std::size_t num, std::size_t den) {
  using float_type = typename ComplexType::value_type;
  const double theta = -2.0 * Kokkos::numbers::pi_v<double> *
                       static_cast<double>(num) / static_cast<double>(den);
  return ComplexType(static_cast<float_type>(std::cos(theta)),
                     static_cast<float_type>(std::sin(theta)));
}

template <typename ComplexType>
StageTables<ComplexType>
make_stage_tables(std::size_t n, const std::vector<std::size_t> &radices) {
  StageTables<ComplexType> tables;
  std::size_t len = n;
  for (auto radix : radices) {
    const std::size_t m = len / radix;
    tables.tw_offset.push_back(tables.twiddles.size());
    for (std::size_t p = 0; p < m; ++p) {
      for (std::size_t j = 1; j < radix; ++j) {
        // j * p < len: no reduction needed
        tables.twiddles.push_back(forward_root<ComplexType>(j * p, len));
      }
    }

    if (is_specialized_radix(radix)) {
      tables.root_offset.push_back(0);
    } else {
      tables.root_offset.push_back(tables.roots.size());
      for (std::size_t t = 0; t < radix; ++t) {
        tables.roots.push_back(forward_root<ComplexType>(t, radix));
      }
    }
    len = m;
  }
  return tables;
}

/// \brief Twiddles W^k = exp(-2 pi i k / n), k = 0..n/4, for the pre/post
/// processing of even-n real transforms (see r2c_postprocess). Empty for
/// n < 2.
template <typename ComplexType>
std::vector<ComplexType> make_real_twiddles(std::size_t n) {
  std::vector<ComplexType> twiddles;
  if (n < 2)
    return twiddles;
  const std::size_t h = n / 2;
  for (std::size_t k = 0; k <= h / 2; ++k) {
    twiddles.push_back(forward_root<ComplexType>(k, n));
  }
  return twiddles;
}

/// \brief Twiddles W_n^t = exp(-2 pi i t / n), t = 0..n-1, for odd-n real
/// transforms (see KokkosFFT_Batched_Kernels_Odd.hpp)
template <typename ComplexType>
std::vector<ComplexType> make_odd_real_twiddles(std::size_t n) {
  std::vector<ComplexType> twiddles;
  for (std::size_t t = 0; t < n; ++t) {
    twiddles.push_back(forward_root<ComplexType>(t, n));
  }
  return twiddles;
}

/// \brief Copy a host vector into a new labelled View in the memory space of
/// `exec`. The copy is fenced so the host vector may be freed afterwards.
/// No host mirror is allocated.
template <typename ExecutionSpace, typename ValueType>
Kokkos::View<ValueType *, typename ExecutionSpace::memory_space>
to_view(const ExecutionSpace &exec, const std::string &label,
        const std::vector<ValueType> &host) {
  using view_type =
      Kokkos::View<ValueType *, typename ExecutionSpace::memory_space>;
  view_type view(Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, label),
                 host.size());
  Kokkos::View<const ValueType *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      host_view(host.data(), host.size());
  Kokkos::deep_copy(exec, view, host_view);
  exec.fence("KokkosFFT::Batched::Plan: twiddle initialization");
  return view;
}

} // namespace Impl
} // namespace Batched
} // namespace KokkosFFT

#endif
