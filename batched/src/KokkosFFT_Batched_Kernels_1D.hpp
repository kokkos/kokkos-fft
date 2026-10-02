#ifndef KOKKOSFFT_BATCHED_KERNELS_1D_HPP
#define KOKKOSFFT_BATCHED_KERNELS_1D_HPP

#include "KokkosFFT_Batched_Accessors.hpp"
#include "KokkosFFT_Batched_Butterflies.hpp"
#include "KokkosFFT_Batched_Levels.hpp"
#include <KokkosFFT_Common_Types.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief One Stockham stage of a 1D C2C transform, src -> dst.
/// The stage has radix r = axis.radix(stage), current length `len` and
/// stride `stride` (len * stride = n). The m * stride butterflies of the
/// stage are independent and distributed over the level.
template <KokkosFFT::Direction Dir, typename LevelType, typename AxisDataType,
          typename SrcLineType, typename DstLineType>
KOKKOS_FUNCTION void
stockham_stage(const LevelType &level, const AxisDataType &axis, int stage,
               std::size_t len, std::size_t stride, const SrcLineType &src,
               const DstLineType &dst) {
  using twiddle_view_type = typename AxisDataType::twiddle_view_type;
  const std::size_t radix = axis.radix(stage);
  const std::size_t m = len / radix;
  const StageTwiddles<Dir, twiddle_view_type> tw{
      axis.twiddles(), axis.tw_offset(stage), radix - 1};

  level.for_each(m * stride, [&](std::size_t idx) {
    const std::size_t p = idx / stride;
    const std::size_t q = idx - p * stride;
    switch (radix) {
    case 2:
      butterfly<2, Dir>(src, dst, tw, p, q, m, stride);
      break;
    case 3:
      butterfly<3, Dir>(src, dst, tw, p, q, m, stride);
      break;
    case 4:
      butterfly<4, Dir>(src, dst, tw, p, q, m, stride);
      break;
    case 5:
      butterfly<5, Dir>(src, dst, tw, p, q, m, stride);
      break;
    default:
      butterfly_generic<Dir>(src, dst, tw, axis.roots(),
                             axis.root_offset(stage), radix, p, q, m, stride);
      break;
    }
  });
}

/// \brief All the stages of a 1D complex transform of length axis.n_fft().
/// The data starts in `a`, and `b` is the work buffer. Stages ping-pong
/// a -> b -> a -> ...; the result is in `b` iff axis.nstages() is odd.
/// `a` and `b` may be different line types (e.g. a real line read as complex
/// pairs for R2C/C2R).
template <KokkosFFT::Direction Dir, typename LevelType, typename AxisDataType,
          typename LineTypeA, typename LineTypeB>
KOKKOS_FUNCTION void c2c_line(const LevelType &level, const AxisDataType &axis,
                              const LineTypeA &a, const LineTypeB &b) {
  std::size_t len = axis.n_fft();
  std::size_t stride = 1;
  for (int stage = 0; stage < axis.nstages(); ++stage) {
    if (stage % 2 == 0) {
      stockham_stage<Dir>(level, axis, stage, len, stride, a, b);
    } else {
      stockham_stage<Dir>(level, axis, stage, len, stride, b, a);
    }
    level.barrier();
    const std::size_t radix = axis.radix(stage);
    len /= radix;
    stride *= radix;
  }
}

/// \brief All the stages of the 1D complex transforms of `nlines` lines,
/// stage by stage: the butterflies of all the lines of a stage are
/// distributed over the level in one for_each (see for_each_in_lines for
/// `line_fastest`), followed by a barrier. lines_of(l) returns the line
/// accessors of line l as a pair (data, work buffer), by value; lines must
/// not share memory. As for c2c_line, the results are in the work buffers
/// iff axis.nstages() is odd.
///
/// The butterfly dispatch repeats the one of stockham_stage on purpose, so
/// that the per-line kernel keeps its code unchanged. The code size of the
/// merged kernels matters on the host: with g++ 12 at its default inlining
/// budget (inline-unit-growth), a translation unit that instantiates both the
/// serial and the team N-D passes inlines less, and the N-D serial plans ran
/// 15-25% slower than before (OpenMP); with a larger budget they do not.
template <KokkosFFT::Direction Dir, typename LevelType, typename AxisDataType,
          typename LinesOf>
KOKKOS_FUNCTION void c2c_lines(const LevelType &level, const AxisDataType &axis,
                               std::size_t nlines, bool line_fastest,
                               const LinesOf &lines_of) {
  using twiddle_view_type = typename AxisDataType::twiddle_view_type;
  std::size_t len = axis.n_fft();
  std::size_t stride = 1;
  for (int stage = 0; stage < axis.nstages(); ++stage) {
    const std::size_t radix = axis.radix(stage);
    const std::size_t m = len / radix;
    const StageTwiddles<Dir, twiddle_view_type> tw{
        axis.twiddles(), axis.tw_offset(stage), radix - 1};
    const bool a_to_b = stage % 2 == 0;
    // Butterfly `idx` of the stage, src -> dst
    const auto butterfly_at = [&](std::size_t idx, const auto &src,
                                  const auto &dst) {
      const std::size_t p = fast_div(idx, stride);
      const std::size_t q = idx - p * stride;
      switch (radix) {
      case 2:
        butterfly<2, Dir>(src, dst, tw, p, q, m, stride);
        break;
      case 3:
        butterfly<3, Dir>(src, dst, tw, p, q, m, stride);
        break;
      case 4:
        butterfly<4, Dir>(src, dst, tw, p, q, m, stride);
        break;
      case 5:
        butterfly<5, Dir>(src, dst, tw, p, q, m, stride);
        break;
      default:
        butterfly_generic<Dir>(src, dst, tw, axis.roots(),
                               axis.root_offset(stage), radix, p, q, m, stride);
        break;
      }
    };
    for_each_in_lines(level, nlines, m * stride, line_fastest,
                      [&](std::size_t l, std::size_t idx) {
                        const auto lines = lines_of(l);
                        if (a_to_b) {
                          butterfly_at(idx, lines.first, lines.second);
                        } else {
                          butterfly_at(idx, lines.second, lines.first);
                        }
                      });
    level.barrier();
    len /= radix;
    stride *= radix;
  }
}

// Even-n real transforms through a half-length complex FFT.
// With h = n/2, z[k] = x[2k] + i x[2k+1], Z = DFT_h(z) and W = exp(-2 pi i/n):
//   Z[k] = E[k] + i O[k], where E, O are the DFTs of the even/odd samples,
//   X[k] = E[k] + W^k O[k],  k = 0..h.
// Both passes below only couple k with h-k, so work item k handles the pair
// (k, h-k): it reads both entries before writing them, which makes the
// passes safe in place and in parallel.

/// \brief R2C post-processing: Z (length h, in `src`) -> X (length h+1, in
/// `dst`). `src` and `dst` may be the same line. Item k (k = 0..h/2) handles
/// the pair (k, h-k).
///
///   E = (Z[k] + conj(Z[h-k])) / 2,  O = -i (Z[k] - conj(Z[h-k])) / 2
///   X[k] = E + W^k O,  X[h-k] = conj(E - W^k O)          (k = 1..h/2)
///   X[0] = Re Z[0] + Im Z[0],  X[h] = Re Z[0] - Im Z[0]
template <typename TwiddleViewType, typename SrcLineType, typename DstLineType>
KOKKOS_INLINE_FUNCTION void
r2c_postprocess_item(std::size_t h, const TwiddleViewType &tw, std::size_t k,
                     const SrcLineType &src, const DstLineType &dst) {
  using value_type = typename SrcLineType::value_type;
  using T = typename value_type::value_type;
  if (k == 0) {
    const value_type z0 = src.load(0);
    dst.store(0, value_type(z0.real() + z0.imag(), T(0)));
    dst.store(h, value_type(z0.real() - z0.imag(), T(0)));
    return;
  }
  const value_type zk = src.load(k);
  const value_type zhk = Kokkos::conj(src.load(h - k));
  const value_type e = T(0.5) * (zk + zhk);
  const value_type o = T(-0.5) * mul_i(zk - zhk);
  const value_type t = tw(k) * o;
  dst.store(k, e + t);
  if (k != h - k)
    dst.store(h - k, Kokkos::conj(e - t));
}

/// \brief R2C post-processing of one line: the items k = 0..h/2 (see
/// r2c_postprocess_item) are distributed over the level
template <typename LevelType, typename AxisDataType, typename SrcLineType,
          typename DstLineType>
KOKKOS_FUNCTION void
r2c_postprocess(const LevelType &level, const AxisDataType &axis,
                const SrcLineType &src, const DstLineType &dst) {
  const std::size_t h = axis.n_fft();
  const auto &tw = axis.real_twiddles();
  level.for_each(h / 2 + 1, [&](std::size_t k) {
    r2c_postprocess_item(h, tw, k, src, dst);
  });
}

/// \brief C2R pre-processing, in place: X (length h+1) -> 2 Z (length h).
/// The imaginary parts of X[0] and X[h] are ignored. Item k (k = 0..h/2)
/// handles the pair (k, h-k).
///
///   Z'[k]   = (X[k] + conj(X[h-k])) + i (X[k] - conj(X[h-k])) conj(W^k)
///   Z'[h-k] = conj(X[k] + conj(X[h-k])) + i conj((X[k] - conj(X[h-k]))
///             conj(W^k))                                  (k = 1..h/2)
///   Z'[0]   = (Re X[0] + Re X[h]) + i (Re X[0] - Re X[h])
///
/// Z' = 2 Z, so the unnormalized backward DFT of length h gives n x, the
/// unnormalized C2R result; the normalization is applied afterwards.
template <typename TwiddleViewType, typename LineType>
KOKKOS_INLINE_FUNCTION void
c2r_preprocess_item(std::size_t h, const TwiddleViewType &tw, std::size_t k,
                    const LineType &x) {
  using value_type = typename LineType::value_type;
  if (k == 0) {
    const auto x0 = x.load(0).real();
    const auto xh = x.load(h).real();
    x.store(0, value_type(x0 + xh, x0 - xh));
    return;
  }
  const value_type xk = x.load(k);
  const value_type xhk = Kokkos::conj(x.load(h - k));
  const value_type e = xk + xhk;
  const value_type o = (xk - xhk) * Kokkos::conj(tw(k));
  x.store(k, e + mul_i(o));
  if (k != h - k)
    x.store(h - k, Kokkos::conj(e) + mul_i(Kokkos::conj(o)));
}

/// \brief C2R pre-processing of one line: the items k = 0..h/2 (see
/// c2r_preprocess_item) are distributed over the level
template <typename LevelType, typename AxisDataType, typename LineType>
KOKKOS_FUNCTION void c2r_preprocess(const LevelType &level,
                                    const AxisDataType &axis,
                                    const LineType &x) {
  const std::size_t h = axis.n_fft();
  const auto &tw = axis.real_twiddles();
  level.for_each(h / 2 + 1,
                 [&](std::size_t k) { c2r_preprocess_item(h, tw, k, x); });
}

/// \brief Scaling factor applied after a transform of total size n, following
/// KokkosFFT: `forward` scales the forward transform by 1/n, `backward` the
/// backward transform by 1/n, `ortho` both by 1/sqrt(n), `none` neither.
template <typename T, KokkosFFT::Direction Dir>
KOKKOS_INLINE_FUNCTION T normalization_factor(KokkosFFT::Normalization norm,
                                              std::size_t n) {
  switch (norm) {
  case KokkosFFT::Normalization::forward:
    return Dir == KokkosFFT::Direction::forward ? T(1) / static_cast<T>(n)
                                                : T(1);
  case KokkosFFT::Normalization::backward:
    return Dir == KokkosFFT::Direction::backward ? T(1) / static_cast<T>(n)
                                                 : T(1);
  case KokkosFFT::Normalization::ortho:
    return T(1) / Kokkos::sqrt(static_cast<T>(n));
  default:
    return T(1);
  }
}

} // namespace Impl
} // namespace Batched
} // namespace KokkosFFT

#endif
