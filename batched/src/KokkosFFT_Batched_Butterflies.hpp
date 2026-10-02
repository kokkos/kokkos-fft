#ifndef KOKKOSFFT_BATCHED_BUTTERFLIES_HPP
#define KOKKOSFFT_BATCHED_BUTTERFLIES_HPP

#include "KokkosFFT_Batched_Accessors.hpp"
#include "KokkosFFT_Batched_Traits.hpp"
#include <KokkosFFT_Common_Types.hpp>
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace KokkosFFT {
namespace Batched {
namespace Impl {

/// \brief Sign of the exponent: -1 for forward, +1 for backward
template <KokkosFFT::Direction Dir, typename T>
inline constexpr T direction_sign_v =
    Dir == KokkosFFT::Direction::forward ? T(-1) : T(1);

/// \brief i * z
template <typename ComplexType>
KOKKOS_INLINE_FUNCTION ComplexType mul_i(const ComplexType &z) {
  return ComplexType(-z.imag(), z.real());
}

/// \brief Forward table entry, conjugated for the backward transform
template <KokkosFFT::Direction Dir, typename ComplexType>
KOKKOS_INLINE_FUNCTION ComplexType directed(const ComplexType &w) {
  if constexpr (Dir == KokkosFFT::Direction::backward) {
    return Kokkos::conj(w);
  } else {
    return w;
  }
}

// In-place DFTs of size R, with s = -1 (forward) or +1 (backward):
//
//   a_j <- sum_{k=0}^{R-1} a_k w^{j k},  w = exp(s 2 pi i / R),  j = 0..R-1

/// \brief In-place radix-2 DFT
///
///   w = exp(s 2 pi i / 2) = -1 for both directions
///   a_0 <- a_0 + a_1
///   a_1 <- a_0 - a_1
template <KokkosFFT::Direction Dir, typename ComplexType>
KOKKOS_INLINE_FUNCTION void small_dft2(ComplexType (&a)[2]) {
  const ComplexType a0 = a[0];
  a[0] = a0 + a[1];
  a[1] = a0 - a[1];
}

/// \brief In-place radix-3 DFT
///
///   w = exp(s 2 pi i / 3) = c + i s_3, with w^2 = conj(w) and
///   c = cos(2 pi / 3) = -1/2, s_3 = s sin(2 pi / 3) = s sqrt(3) / 2
///
///   a_0 <- a_0 + a_1 + a_2
///   a_1 <- a_0 + w a_1 + w^2 a_2 = a_0 + c (a_1 + a_2) + i s_3 (a_1 - a_2)
///   a_2 <- a_0 + w^2 a_1 + w a_2 = a_0 + c (a_1 + a_2) - i s_3 (a_1 - a_2)
///
/// Computed as t1 = a_1 + a_2, t2 = a_0 + c t1, t3 = s_3 (a_1 - a_2):
///   a_0 = a_0 + t1,  a_1 = t2 + i t3,  a_2 = t2 - i t3
template <KokkosFFT::Direction Dir, typename ComplexType>
KOKKOS_INLINE_FUNCTION void small_dft3(ComplexType (&a)[3]) {
  using T = typename ComplexType::value_type;
  constexpr T sign = direction_sign_v<Dir, T>;
  constexpr T c = T(-0.5);                                    // cos(2 pi / 3)
  constexpr T s = sign * T(0.866025403784438646763723170753); // sin(2 pi / 3)
  const ComplexType t1 = a[1] + a[2];
  const ComplexType t2 = a[0] + c * t1;
  const ComplexType t3 = s * (a[1] - a[2]);
  a[0] = a[0] + t1;
  a[1] = t2 + mul_i(t3);
  a[2] = t2 - mul_i(t3);
}

/// \brief In-place radix-4 DFT
///
///   w = exp(s 2 pi i / 4) = s i, with w^2 = -1 and w^3 = -w
///
///   a_0 <- a_0 + a_1 + a_2 + a_3
///   a_1 <- a_0 + w a_1 - a_2 - w a_3 = (a_0 - a_2) + w (a_1 - a_3)
///   a_2 <- a_0 - a_1 + a_2 - a_3     = (a_0 + a_2) - (a_1 + a_3)
///   a_3 <- a_0 - w a_1 - a_2 + w a_3 = (a_0 - a_2) - w (a_1 - a_3)
///
/// Computed as t0 = a_0 + a_2, t1 = a_0 - a_2, t2 = a_1 + a_3,
/// t3 = s i (a_1 - a_3):
///   a_0 = t0 + t2,  a_1 = t1 + t3,  a_2 = t0 - t2,  a_3 = t1 - t3
template <KokkosFFT::Direction Dir, typename ComplexType>
KOKKOS_INLINE_FUNCTION void small_dft4(ComplexType (&a)[4]) {
  using T = typename ComplexType::value_type;
  constexpr T sign = direction_sign_v<Dir, T>;
  const ComplexType t0 = a[0] + a[2];
  const ComplexType t1 = a[0] - a[2];
  const ComplexType t2 = a[1] + a[3];
  const ComplexType t3 = sign * mul_i(a[1] - a[3]);
  a[0] = t0 + t2;
  a[1] = t1 + t3;
  a[2] = t0 - t2;
  a[3] = t1 - t3;
}

/// \brief In-place radix-5 DFT
///
///   w = exp(s 2 pi i / 5), with w^4 = conj(w) and w^3 = conj(w^2)
///   c1 = cos(2 pi / 5), c2 = cos(4 pi / 5),
///   s1 = s sin(2 pi / 5), s2 = s sin(4 pi / 5)
///
///   a_0 <- a_0 + a_1 + a_2 + a_3 + a_4
///   a_1 <- a_0 + w a_1 + w^2 a_2 + w^3 a_3 + w^4 a_4
///   a_2 <- a_0 + w^2 a_1 + w^4 a_2 + w a_3 + w^3 a_4
///   a_3 <- a_0 + w^3 a_1 + w a_2 + w^4 a_3 + w^2 a_4
///   a_4 <- a_0 + w^4 a_1 + w^3 a_2 + w^2 a_3 + w a_4
///
/// Pairing the conjugate terms, with b1 = a_1 + a_4, b2 = a_2 + a_3,
/// d1 = a_1 - a_4, d2 = a_2 - a_3:
///   u1 = a_0 + c1 b1 + c2 b2,  v1 = s1 d1 + s2 d2
///   u2 = a_0 + c2 b1 + c1 b2,  v2 = s2 d1 - s1 d2
///   a_0 = a_0 + b1 + b2
///   a_1 = u1 + i v1,  a_4 = u1 - i v1
///   a_2 = u2 + i v2,  a_3 = u2 - i v2
template <KokkosFFT::Direction Dir, typename ComplexType>
KOKKOS_INLINE_FUNCTION void small_dft5(ComplexType (&a)[5]) {
  using T = typename ComplexType::value_type;
  constexpr T sign = direction_sign_v<Dir, T>;
  constexpr T c1 = T(0.309016994374947424102293417183);        // cos(2 pi / 5)
  constexpr T c2 = T(-0.809016994374947424102293417183);       // cos(4 pi / 5)
  constexpr T s1 = sign * T(0.951056516295153572116439333379); // sin(2pi/5)
  constexpr T s2 = sign * T(0.587785252292473129168705954639); // sin(4pi/5)
  const ComplexType b1 = a[1] + a[4];
  const ComplexType b2 = a[2] + a[3];
  const ComplexType d1 = a[1] - a[4];
  const ComplexType d2 = a[2] - a[3];
  const ComplexType u1 = a[0] + c1 * b1 + c2 * b2;
  const ComplexType u2 = a[0] + c2 * b1 + c1 * b2;
  const ComplexType v1 = s1 * d1 + s2 * d2;
  const ComplexType v2 = s2 * d1 - s1 * d2;
  a[0] = a[0] + b1 + b2;
  a[1] = u1 + mul_i(v1);
  a[2] = u2 + mul_i(v2);
  a[3] = u2 - mul_i(v2);
  a[4] = u1 - mul_i(v1);
}

/// \brief In-place DFT of size R, dispatched to small_dft2/3/4/5
template <std::size_t R, KokkosFFT::Direction Dir, typename ComplexType>
KOKKOS_INLINE_FUNCTION void small_dft(ComplexType (&a)[R]) {
  if constexpr (R == 2) {
    small_dft2<Dir>(a);
  } else if constexpr (R == 3) {
    small_dft3<Dir>(a);
  } else if constexpr (R == 4) {
    small_dft4<Dir>(a);
  } else if constexpr (R == 5) {
    small_dft5<Dir>(a);
  } else {
    static_assert(always_false_v<ComplexType>,
                  "small_dft: only radix 2, 3, 4 and 5 are specialised");
  }
}

/// \brief Twiddles w_len^{j p} of one stage (p-major table, see
/// StageTables), conjugated for the backward transform.
template <KokkosFFT::Direction Dir, typename TwiddleViewType>
struct StageTwiddles {
  TwiddleViewType m_twiddles;
  std::size_t m_offset;
  std::size_t m_radix_minus_one;

  KOKKOS_INLINE_FUNCTION auto operator()(std::size_t j, std::size_t p) const {
    return directed<Dir>(m_twiddles(m_offset + p * m_radix_minus_one + j - 1));
  }
};

/// \brief One radix-R Stockham DIF butterfly:
///   a_k = src[q + stride (p + k m)],
///   dst[q + stride (R p + j)] = (sum_k a_k omega_R^{j k}) w_len^{j p}
/// src and dst may be different line types (StridedLine, RealPairAsComplex).
template <std::size_t R, KokkosFFT::Direction Dir, typename SrcLineType,
          typename DstLineType, typename TwiddlesType>
KOKKOS_INLINE_FUNCTION void
butterfly(const SrcLineType &src, const DstLineType &dst,
          const TwiddlesType &tw, std::size_t p, std::size_t q, std::size_t m,
          std::size_t stride) {
  using value_type = typename SrcLineType::value_type;
  value_type a[R];
  for (std::size_t k = 0; k < R; ++k) {
    a[k] = src.load(q + stride * (p + k * m));
  }
  small_dft<R, Dir>(a);
  const std::size_t base = q + stride * R * p;
  dst.store(base, a[0]);
  for (std::size_t j = 1; j < R; ++j) {
    dst.store(base + stride * j, a[j] * tw(j, p));
  }
}

/// \brief Generic radix-r butterfly for a prime r without a specialisation.
/// It reads the inputs from `src` for each output (O(r^2)) instead of keeping
/// r temporaries, so no scratch array is needed.
template <KokkosFFT::Direction Dir, typename SrcLineType, typename DstLineType,
          typename TwiddlesType, typename RootViewType>
KOKKOS_INLINE_FUNCTION void
butterfly_generic(const SrcLineType &src, const DstLineType &dst,
                  const TwiddlesType &tw, const RootViewType &roots,
                  std::size_t root_offset, std::size_t radix, std::size_t p,
                  std::size_t q, std::size_t m, std::size_t stride) {
  using value_type = typename SrcLineType::value_type;
  const std::size_t base = q + stride * radix * p;
  for (std::size_t j = 0; j < radix; ++j) {
    value_type acc(0);
    std::size_t t = 0; // (j * k) mod radix
    for (std::size_t k = 0; k < radix; ++k) {
      acc += src.load(q + stride * (p + k * m)) *
             directed<Dir>(roots(root_offset + t));
      t += j;
      if (t >= radix)
        t -= radix;
    }
    dst.store(base + stride * j, j == 0 ? acc : acc * tw(j, p));
  }
}

} // namespace Impl
} // namespace Batched
} // namespace KokkosFFT

#endif
