// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ULPS_HPP
#define KOKKOSFFT_ULPS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Half.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_BitManipulation.hpp>
#include "KokkosFFT_Concepts.hpp"

namespace KokkosFFT {
namespace Testing {
namespace Impl {

// Helper structure to select the correct integer type for bit reinterpretation
template <typename T>
struct FloatIntMap;

template <>
struct FloatIntMap<float> {
  using IntType = std::int32_t;
};
template <>
struct FloatIntMap<double> {
  using IntType = std::int64_t;
};

// half_t and bhalf_t can be alias to float types
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
template <>
struct FloatIntMap<Kokkos::Experimental::half_t> {
  using IntType = std::int16_t;
};
#endif

#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
template <>
struct FloatIntMap<Kokkos::Experimental::bhalf_t> {
  using IntType = std::int16_t;
};
#endif

template <typename ScalarA, typename ScalarB>
KOKKOS_INLINE_FUNCTION bool almost_equal_ulps(ScalarA a, ScalarB b,
                                              std::size_t max_ulps_diff) {
  // We consider nans are identical
  if (Kokkos::isnan(a) && Kokkos::isnan(b)) return true;

  // Handle non-finite cases and exact equality first
  if (Kokkos::isnan(a) || Kokkos::isnan(b)) return false;

  // Exact equality check: Handles +0 == -0, inf == inf, -inf == -inf
  if (a == b) return true;

  // Since a != b, we only need to check if a or b is infinite
  if (Kokkos::isinf(a) || Kokkos::isinf(b)) return false;

  // Get the integer representation of the floats
  // Not sure how to compare if two types do not have the common type
  using CommonFloatType = std::common_type_t<ScalarA, ScalarB>;
  using IntType         = typename FloatIntMap<CommonFloatType>::IntType;

  // Reinterpret the bits using Kokkos::bit_cast
  IntType int_a = Kokkos::bit_cast<IntType>(a);
  IntType int_b = Kokkos::bit_cast<IntType>(b);

  // Calculate the ULP difference
  IntType ulps_diff = int_a - int_b;

  // Do not know how Kokkos::abs works on int16_t,
  // so hardcode the absolute value
  using UnsignedIntType         = typename std::make_unsigned<IntType>::type;
  UnsignedIntType abs_ulps_diff = (ulps_diff < 0)
                                      ? static_cast<UnsignedIntType>(-ulps_diff)
                                      : static_cast<UnsignedIntType>(ulps_diff);

  // Compare with the maximum allowed ULP difference
  return abs_ulps_diff <= static_cast<UnsignedIntType>(max_ulps_diff);
}

}  // namespace Impl
}  // namespace Testing
}  // namespace KokkosFFT

#endif
