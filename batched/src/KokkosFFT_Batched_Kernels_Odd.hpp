#ifndef KOKKOSFFT_BATCHED_KERNELS_ODD_HPP
#define KOKKOSFFT_BATCHED_KERNELS_ODD_HPP

#include "KokkosFFT_Batched_Accessors.hpp"
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

// Odd-n R2C/C2R: a real decimation-in-time Stockham FFT on half-complex data.
//
// n = r_0 r_1 ... r_{S-1} (all radices odd). Before stage q, the data holds
// G = n / L groups; group g is the DFT Y_g (length L) of the real sequence
// x[g + G t], t = 0..L-1. Stage q (radix r) combines r groups into one of
// length rL (W_N = exp(-2 pi i / N)):
//
//   forward : V_g[F]      = sum_k W_{rL}^{k F} Y_{g+Gk}[F mod L]
//   backward: Y_{g+Gk}[f] = sum_j V_g[f + L j] W_{rL}^{-k (f + L j)}
//
// The backward stage gives r Y, so the full backward transform gives n x,
// the unnormalized C2R result. Initially (L = 1) group g is x[g]; finally
// (L = n) the single group is X.
//
// Every group is the DFT of a real sequence of odd length L, so it is stored
// half-complex (HC) in L reals at base L g:
//   hc[0] = Re Y[0], hc[2f-1] = Re Y[f], hc[2f] = Im Y[f], f = 1..(L-1)/2
// and Y[L-f] = conj(Y[f]). The stages ping-pong between two real buffers of
// n reals: the `in` line and the `out` line read as reals (ComplexAsReal).
// Each output element is an independent O(r) sum, so the work items of a
// stage are independent. The twiddles come from one table W_n^t,
// t = 0..n-1, since W_{rL}^t = W_n^{t G'} with G' = n / (rL).

/// \brief Y[F] of an HC group of odd length L stored at line[base..base+L)
template <typename LineType>
KOKKOS_INLINE_FUNCTION Kokkos::complex<typename LineType::value_type>
hc_load(const LineType &line, std::size_t base, std::size_t L, std::size_t F) {
  using T = typename LineType::value_type;
  using complex_type = Kokkos::complex<T>;
  if (F == 0)
    return complex_type(line.load(base), T(0));
  if (2 * F < L) {
    return complex_type(line.load(base + 2 * F - 1), line.load(base + 2 * F));
  }
  const std::size_t f = L - F;
  return complex_type(line.load(base + 2 * f - 1), -line.load(base + 2 * f));
}

/// \brief Store Y[F] (F <= (L-1)/2) of an HC group at line[base..)
template <typename LineType, typename ComplexType>
KOKKOS_INLINE_FUNCTION void hc_store(const LineType &line, std::size_t base,
                                     std::size_t F, const ComplexType &y) {
  if (F == 0) {
    line.store(base, y.real());
  } else {
    line.store(base + 2 * F - 1, y.real());
    line.store(base + 2 * F, y.imag());
  }
}

/// \brief X[F] (F = 0..n-1) of a Hermitian spectrum given by X[0..(n-1)/2]
/// (odd n). Im X[0] is ignored, as in numpy.
template <typename LineType>
KOKKOS_INLINE_FUNCTION typename LineType::value_type
hermitian_load(const LineType &line, std::size_t n, std::size_t F) {
  using complex_type = typename LineType::value_type;
  using T = typename complex_type::value_type;
  if (F == 0)
    return complex_type(line.load(0).real(), T(0));
  if (2 * F < n)
    return line.load(F);
  return Kokkos::conj(line.load(n - F));
}

/// \brief Forward stage: groups of length L (HC in `src`) -> groups of
/// length rL. ToComplex (last stage, one group): X[F] is written to the
/// complex line `dst`; otherwise HC groups are written to the real line.
template <bool ToComplex, typename LevelType, typename AxisDataType,
          typename SrcLineType, typename DstLineType>
KOKKOS_FUNCTION void odd_r2c_stage(const LevelType &level,
                                   const AxisDataType &axis, std::size_t radix,
                                   std::size_t L, const SrcLineType &src,
                                   const DstLineType &dst) {
  using complex_type = typename AxisDataType::complex_type;
  const std::size_t n = axis.length(0);
  const std::size_t Ls = radix * L;
  // radix >= 3 and L >= 1 by construction; this makes it explicit for the
  // divisions below (and for static analysis)
  if (Ls == 0)
    return;
  const std::size_t G = n / Ls; // output groups
  const std::size_t H = (Ls + 1) / 2;
  const auto &tw = axis.real_twiddles(); // W_n^t

  level.for_each(G * H, [&](std::size_t idx) {
    const std::size_t g = idx / H;
    const std::size_t F = idx - g * H;
    const std::size_t f = F % L;
    complex_type acc(0);
    std::size_t t = 0; // (k F) mod Ls
    for (std::size_t k = 0; k < radix; ++k) {
      acc += tw(t * G) * hc_load(src, L * (g + G * k), L, f);
      t += F;
      if (t >= Ls)
        t -= Ls;
    }
    if constexpr (ToComplex) {
      dst.store(F, acc);
    } else {
      hc_store(dst, Ls * g, F, acc);
    }
  });
}

/// \brief Backward stage: groups of length rL -> groups of length L (HC in
/// the real line `dst`). FromComplex (first stage): the single input group is
/// the complex Hermitian spectrum X[0..(n-1)/2] in `src`; otherwise `src`
/// holds HC groups.
template <bool FromComplex, typename LevelType, typename AxisDataType,
          typename SrcLineType, typename DstLineType>
KOKKOS_FUNCTION void odd_c2r_stage(const LevelType &level,
                                   const AxisDataType &axis, std::size_t radix,
                                   std::size_t L, const SrcLineType &src,
                                   const DstLineType &dst) {
  using complex_type = typename AxisDataType::complex_type;
  const std::size_t n = axis.length(0);
  const std::size_t Ls = radix * L;
  // radix >= 3 and L >= 1 by construction; this makes it explicit for the
  // divisions below (and for static analysis)
  if (Ls == 0)
    return;
  const std::size_t G = n / Ls; // input groups
  const std::size_t H = (L + 1) / 2;
  const auto &tw = axis.real_twiddles(); // W_n^t

  level.for_each(G * radix * H, [&](std::size_t idx) {
    const std::size_t gk = idx / H; // output group g + G k
    const std::size_t f = idx - gk * H;
    const std::size_t g = gk % G;
    const std::size_t k = gk / G;
    const std::size_t step = (k * L) % Ls;
    complex_type acc(0);
    std::size_t t = (k * f) % Ls; // (k (f + L j)) mod Ls
    for (std::size_t j = 0; j < radix; ++j) {
      const std::size_t F = f + L * j;
      complex_type v;
      if constexpr (FromComplex) {
        v = hermitian_load(src, n, F);
      } else {
        v = hc_load(src, Ls * g, Ls, F);
      }
      acc += v * Kokkos::conj(tw(t * G));
      t += step;
      if (t >= Ls)
        t -= Ls;
    }
    hc_store(dst, L * gk, f, acc);
  });
}

/// \brief Odd-n R2C of one line: real `x` (n) -> complex `y` ((n+1)/2).
/// Stages alternate x -> y (as reals) -> x -> ...; the last stage must read
/// `x`, so if the stage count is even the data is first copied back to `x`.
/// `x` is overwritten.
template <typename LevelType, typename AxisDataType, typename RealLineType,
          typename ComplexLineType>
KOKKOS_FUNCTION void
odd_r2c_line(const LevelType &level, const AxisDataType &axis,
             const RealLineType &x, const ComplexLineType &y) {
  using complex_type = typename AxisDataType::complex_type;
  using T = typename complex_type::value_type;
  const std::size_t n = axis.length(0);
  const int nstages = axis.nstages();
  const auto yr = ComplexAsReal<T>{y.m_data, y.m_stride};

  if (nstages == 0) { // n = 1
    level.for_each(1,
                   [&](std::size_t) { y.store(0, complex_type(x.load(0))); });
    level.barrier();
    return;
  }

  std::size_t L = 1;
  for (int s = 0; s < nstages; ++s) {
    const std::size_t radix = axis.radix(s);
    if (s == nstages - 1) {
      if (s % 2 == 1) { // data is in y: copy it back to x
        level.for_each(n, [&](std::size_t i) { x.store(i, yr.load(i)); });
        level.barrier();
      }
      odd_r2c_stage<true>(level, axis, radix, L, x, y);
    } else if (s % 2 == 0) {
      odd_r2c_stage<false>(level, axis, radix, L, x, yr);
    } else {
      odd_r2c_stage<false>(level, axis, radix, L, yr, x);
    }
    level.barrier();
    L *= radix;
  }
}

/// \brief Odd-n C2R of one line: complex `x` ((n+1)/2) -> real `y` (n),
/// unnormalized (n times the inverse DFT). Stages run in reverse order and
/// alternate x -> y -> x (as reals) -> ...; if the last one writes `x`, the
/// result is copied to `y`. `x` is overwritten.
template <typename LevelType, typename AxisDataType, typename ComplexLineType,
          typename RealLineType>
KOKKOS_FUNCTION void
odd_c2r_line(const LevelType &level, const AxisDataType &axis,
             const ComplexLineType &x, const RealLineType &y) {
  using T = typename AxisDataType::float_type;
  const std::size_t n = axis.length(0);
  const int nstages = axis.nstages();
  const auto xr = ComplexAsReal<T>{x.m_data, x.m_stride};

  if (nstages == 0) { // n = 1
    level.for_each(1, [&](std::size_t) { y.store(0, x.load(0).real()); });
    level.barrier();
    return;
  }

  std::size_t Ls = n;
  for (int i = 0; i < nstages; ++i) {
    const int s = nstages - 1 - i;
    const std::size_t radix = axis.radix(s);
    const std::size_t L = Ls / radix;
    if (i == 0) {
      odd_c2r_stage<true>(level, axis, radix, L, x, y);
    } else if (i % 2 == 1) {
      odd_c2r_stage<false>(level, axis, radix, L, y, xr);
    } else {
      odd_c2r_stage<false>(level, axis, radix, L, xr, y);
    }
    level.barrier();
    Ls = L;
  }
  if (nstages % 2 == 0) { // the last stage wrote x
    level.for_each(n, [&](std::size_t i) { y.store(i, xr.load(i)); });
    level.barrier();
  }
}

} // namespace Impl
} // namespace Batched
} // namespace KokkosFFT

#endif
