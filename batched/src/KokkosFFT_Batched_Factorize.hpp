#ifndef KOKKOSFFT_BATCHED_FACTORIZE_HPP
#define KOKKOSFFT_BATCHED_FACTORIZE_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief Radices with a specialised butterfly. Any other prime factor is
/// handled by the generic radix-p butterfly.
inline constexpr bool is_specialized_radix(std::size_t radix) {
  return radix == 2 || radix == 3 || radix == 4 || radix == 5;
}

/// \brief Factorise n into the radices of a mixed-radix Stockham FFT (host).
///
/// Radices 4, 2, 3 and 5 are extracted first, then the remaining prime
/// factors (generic radix). The stages ping-pong between `in` and `out`,
/// starting from `in`, so the result lands in `out` iff the number of stages
/// is odd. If the count is even, one radix-4 stage is split into 2 x 2 to
/// avoid the final copy. n = 1 gives no stage.
///
/// \param n [in] Transform length (n >= 1)
/// \return Radices in the order the stages are executed
inline std::vector<std::size_t> factorize(std::size_t n) {
  std::vector<std::size_t> radices;
  while (n % 4 == 0) {
    radices.push_back(4);
    n /= 4;
  }
  if (n % 2 == 0) {
    radices.push_back(2);
    n /= 2;
  }
  for (std::size_t p : {std::size_t(3), std::size_t(5)}) {
    while (n % p == 0) {
      radices.push_back(p);
      n /= p;
    }
  }
  for (std::size_t p = 7; p * p <= n; p += 2) {
    while (n % p == 0) {
      radices.push_back(p);
      n /= p;
    }
  }
  if (n > 1)
    radices.push_back(n);

  // Prefer an odd number of stages so that the result lands in `out`
  if (radices.size() % 2 == 0) {
    auto it = std::find(radices.begin(), radices.end(), std::size_t(4));
    if (it != radices.end()) {
      *it = 2;
      radices.insert(it, std::size_t(2));
    }
  }
  return radices;
}

} // namespace Impl
} // namespace Batched
} // namespace KokkosFFT

#endif
