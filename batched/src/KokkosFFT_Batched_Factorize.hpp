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
  return radix == 2 || radix == 3 || radix == 4 || radix == 5 || radix == 8;
}

/// \brief Whether the merged C2C passes of a team plan include the radix-8
/// butterfly (G2d): not for slices of rank 3 or more. There, radix 8 raised
/// the registers of the 3-D kernel from 125 to 134, so a 512-thread team no
/// longer fit in a block, and 3-D 32^3 lost its best team size
/// (optimization/measurements/2026-10-04_64d2aab_A100). The kernels compile
/// radix 8 out, and the plans of those slices factorise without it.
/// Radix 8 under launch bounds of 512 threads (experiment M18) fits in 128
/// registers but is not faster in every case
/// (optimization/measurements/2026-10-04_2aca5c2_A100_M18).
///
/// A variable template, not a constexpr function: the device code uses it as
/// a template argument, and nvcc rejects a call to a constexpr host function
/// there without --expt-relaxed-constexpr.
template <std::size_t SliceRank>
inline constexpr bool merged_c2c_radix8_v = SliceRank < 3;

/// \brief Radices for the power of two 2^k in `nstages` stages, as balanced
/// as possible (each stage 2, 4 or 8), in ascending order. Requires
/// ceil(k / 3) <= nstages <= k.
inline std::vector<std::size_t> pow2_radices(std::size_t k,
                                             std::size_t nstages) {
  std::vector<std::size_t> radices;
  for (std::size_t s = 0; s < nstages; ++s) {
    const std::size_t e = k / nstages + (s < k % nstages ? 1 : 0);
    radices.push_back(std::size_t(1) << e);
  }
  std::sort(radices.begin(), radices.end());
  return radices;
}

/// \brief Factorise n into the radices of a mixed-radix Stockham FFT (host).
///
/// Every stage is a full pass over the line, so the factorisation minimises
/// the number of stages (G2a): the power of two 2^k takes ceil(k / 3)
/// stages (radix 8, with 4 x 4, 4 or 2 for the rest: 4 x 4 x 8, not
/// 2 x 8 x 8). Then radices 3 and 5, then the remaining prime factors
/// (generic radix). n = 1 gives no stage.
///
/// The stages ping-pong between `in` and `out`, so the result lands in `in`
/// when their number is even. Where that costs a copy of the result (a 1-D
/// C2C plan, whose finalize then copies; a C2R leaf, which copies each line),
/// `odd_stages` asks for an odd number of stages (G2c): if the minimal count
/// is even, the power of two is spread over one more stage, as evenly as
/// possible (e.g. 64 = 4 x 4 x 4 instead of 8 x 8, 1024 = 4^5). On the A100
/// that copy cost about as much as the stage it replaced
/// (optimization/measurements/2026-10-04_64d2aab_A100).
/// Only if every stage of that split is radix 4 or 8 (G2e, k >= 2 x the new
/// stage count): a radix-2 stage costs more than the copy (16 = 2 x 2 x 4
/// was 11-15% slower than 4 x 4 and a copy,
/// optimization/measurements/2026-10-04_f6689c0_A100). Otherwise the count
/// stays even: 16 = 4 x 4, 32 = 4 x 8.
///
/// Without `radix8` (G2d, see merged_c2c_radix8_v), the power of two takes
/// ceil(k / 2) stages of radix 4, with one radix 2 if k is odd.
///
/// \param n [in] Transform length (n >= 1)
/// \param odd_stages [in] Prefer an odd number of stages
/// \param radix8 [in] Radix 8 is allowed
/// \return Radices in the order the stages are executed
inline std::vector<std::size_t> factorize(std::size_t n,
                                          bool odd_stages = false,
                                          bool radix8     = true) {
  std::size_t k = 0;  // power of two in n
  while (n % 2 == 0) {
    n /= 2;
    ++k;
  }
  std::vector<std::size_t> others;
  for (std::size_t p : {std::size_t(3), std::size_t(5)}) {
    while (n % p == 0) {
      others.push_back(p);
      n /= p;
    }
  }
  for (std::size_t p = 7; p * p <= n; p += 2) {
    while (n % p == 0) {
      others.push_back(p);
      n /= p;
    }
  }
  if (n > 1) others.push_back(n);

  std::size_t pow2_stages = radix8 ? (k + 2) / 3 : (k + 1) / 2;
  if (odd_stages && (pow2_stages + others.size()) % 2 == 0 &&
      2 * (pow2_stages + 1) <= k) {
    ++pow2_stages;
  }
  std::vector<std::size_t> radices = pow2_radices(k, pow2_stages);
  radices.insert(radices.end(), others.begin(), others.end());
  return radices;
}

}  // namespace Impl
}  // namespace Batched
}  // namespace KokkosFFT

#endif
