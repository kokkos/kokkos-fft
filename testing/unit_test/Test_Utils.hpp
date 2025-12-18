// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>

// Helper function for nextafter on fp16 types
template <typename fp16_t>
  requires((std::is_same_v<fp16_t, Kokkos::Experimental::half_t> ||
            std::is_same_v<fp16_t, Kokkos::Experimental::bhalf_t>) &&
           sizeof(fp16_t) == 2)
fp16_t nextafter_fp16(fp16_t from, fp16_t to) {
  constexpr std::uint16_t FP16_SIGN_MASK = 0x8000;
  constexpr std::uint16_t FP16_SMALLEST_POS_DN =
      0x0001;  // Smallest positive denormal
  constexpr std::uint16_t FP16_SMALLEST_NEG_DN =
      0x8001;  // Smallest negative denormal (magnitude)

  // Handle Nans
  if (Kokkos::isnan(from) || Kokkos::isnan(to)) {
    return Kokkos::Experimental::quiet_NaN<fp16_t>::value;
  }

  // Handle equality
  if (from == to) return to;

  // Get unsigned integer representation of from
  std::uint16_t uint_from = Kokkos::bit_cast<std::uint16_t>(from);

  // Handle zeros
  if (from == fp16_t(0)) {
    // from is +0.0 or -0.0
    // Return smallest magnitude number with the sign of 'to'.
    // nextafter(±0, negative) -> smallest_negative
    // nextafter(±0, positive) -> smallest_positive
    return Kokkos::bit_cast<fp16_t>((to > from) ? FP16_SMALLEST_POS_DN
                                                : FP16_SMALLEST_NEG_DN);
  }

  // Determine direction and sign of 'from'
  // True if moving to positive infinity
  bool to_positive_infinity = (to > from);
  bool from_is_negative     = (uint_from & FP16_SIGN_MASK);

  std::uint16_t uint_result =
      uint_from + 2 * (to_positive_infinity ^ from_is_negative) - 1;
  // This is equivalent to the following operations.
  // std::uint16_t uint_result;
  //
  // if (from_is_negative) {
  //  // For negative numbers, increasing magnitude means moving towards -inf
  //  // (larger uint value) Decreasing magnitude means moving towards zero
  //  // (smaller uint value)
  //  if (to_positive_infinity) {
  //    // Moving toward zero or positive
  //    uint_result = uint_from - 1;
  //  } else {
  //    // Moving toward negative infinity
  //    uint_result = uint_from + 1;
  //  }
  //} else {
  //  // For positive numbers, increasing magnitude means moving towards +inf
  //  // (larger uint value) Decreasing magnitude means moving towards zero
  //  // (smaller uint value)
  //  if (to_positive_infinity) {
  //    // Moving toward positive infinity
  //    uint_result = uint_from + 1;
  //  } else {
  //    // Moving toward zero or negative infinity
  //    uint_result = uint_from - 1;
  //  }
  //}
  return Kokkos::bit_cast<fp16_t>(uint_result);
}

template <typename T>
auto nextafter_wrapper(T from, T to) {
  if constexpr (std::is_same_v<T, Kokkos::Experimental::half_t>) {
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
    return nextafter_fp16<T>(from, to);
#else
    return Kokkos::nextafter(from, to);
#endif
  } else if constexpr (std::is_same_v<T, Kokkos::Experimental::bhalf_t>) {
#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
    return nextafter_fp16<T>(from, to);
#else
    return Kokkos::nextafter(from, to);
#endif
  } else {
    return Kokkos::nextafter(from, to);
  }
}

#endif
