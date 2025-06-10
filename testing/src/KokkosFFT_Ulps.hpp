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
  using IntType = std::uint32_t;
};
template <>
struct FloatIntMap<double> {
  using IntType = std::uint64_t;
};

// half_t and bhalf_t can be alias to float types
#if defined(KOKKOS_HALF_T_IS_FLOAT) && !KOKKOS_HALF_T_IS_FLOAT
template <>
struct FloatIntMap<Kokkos::Experimental::half_t> {
  using IntType = std::uint16_t;
};
#endif

#if defined(KOKKOS_BHALF_T_IS_FLOAT) && !KOKKOS_BHALF_T_IS_FLOAT
template <>
struct FloatIntMap<Kokkos::Experimental::bhalf_t> {
  using IntType = std::uint16_t;
};
#endif

/// \brief Convert a floating-point number to a biased integer representation.
/// Converts the float to unsigned integer representation, and then represents
/// them as their two's complement
/// See https://en.wikipedia.org/wiki/Two%27s_complement for detail
///
/// \tparam UIntType The unsigned integer type to convert to (e.g., uint32_t,
/// uint16_t)
/// \tparam FloatType The floating-point type to convert from (e.g.,
/// float, double)
///
/// \param[in] from The floating-point number to convert
/// \return The biased integer representation of the floating-point number
template <typename UIntType, typename FloatType>
  requires(sizeof(UIntType) == sizeof(FloatType))
KOKKOS_INLINE_FUNCTION UIntType float_to_biased_int(FloatType from) {
  UIntType to = Kokkos::bit_cast<UIntType>(from);

  // e.g., 0x80000000 for int32_t, 0x8000 for int16_t
  constexpr UIntType sign_mask = UIntType(1) << (sizeof(UIntType) * 8 - 1);

  if (to & sign_mask) {
    // Original float was negative (or -0.0f)
    // Bitwise NOT to reverse order and map to lower half conceptually
    return ~to + 1;
  } else {
    // Original float was positive (or +0.0f)
    // OR with sign_mask to shift positive values to the upper half conceptually
    return to | sign_mask;
  }
}

/// \brief Compare two floating-point numbers for approximate equality
/// using the ULP (Units in the Last Place) method.
///
/// \tparam ScalarA The type of the first floating-point number
/// \tparam ScalarB The type of the second floating-point number
///
/// \param[in] a The first floating-point number
/// \param[in] b The second floating-point number
/// \param[in] max_ulps_diff The maximum allowed difference in ULPs for the
/// numbers to be considered equal
/// \return True if the two numbers are approximately equal within the specified
/// ULP difference, false otherwise
template <typename ScalarA, typename ScalarB>
KOKKOS_INLINE_FUNCTION bool almost_equal_ulps(ScalarA a, ScalarB b,
                                              std::size_t max_ulps_diff) {
  // Handle non-finite cases and exact equality first
  if (Kokkos::isnan(a) || Kokkos::isnan(b)) return false;

  // Exact equality check: Handles +0 == -0, inf == inf, -inf == -inf
  if (a == b) return true;

  // Since a != b, we only need to check if a or b is infinite
  if (Kokkos::isinf(a) || Kokkos::isinf(b)) return false;

  // Get the integer representation of the floats
  // Not sure how to compare if two types do not have the common type
  using CommonFloatType = std::common_type_t<ScalarA, ScalarB>;
  using UIntType        = typename FloatIntMap<CommonFloatType>::IntType;

  UIntType biased_a = float_to_biased_int<UIntType>(a);
  UIntType biased_b = float_to_biased_int<UIntType>(b);

  UIntType abs_ulps_diff =
      (biased_a > biased_b) ? biased_a - biased_b : biased_b - biased_a;

  // Compare with the maximum allowed ULP difference
  return abs_ulps_diff <= static_cast<UIntType>(max_ulps_diff);
}

template <typename IntType>
struct UlpsComparisonOp {
 private:
  IntType m_max_ulps_diff;

 public:
  UlpsComparisonOp(IntType max_ulps_diff = 1)
      : m_max_ulps_diff(max_ulps_diff) {}

  template <typename ScalarA, typename ScalarB>
  KOKKOS_INLINE_FUNCTION bool operator()(ScalarA a, ScalarB b) const {
    return almost_equal_ulps(a, b, m_max_ulps_diff);
  }
};

}  // namespace Impl
}  // namespace Testing
}  // namespace KokkosFFT

#endif
