// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ALMOST_EQUAL_HPP
#define KOKKOSFFT_ALMOST_EQUAL_HPP

#include <Kokkos_Core.hpp>

namespace KokkosFFT {
namespace Testing {
namespace Impl {

template <typename ScalarA, typename ScalarB, typename ScalarTol>
KOKKOS_INLINE_FUNCTION bool are_almost_equal(ScalarA a, ScalarB b,
                                             ScalarTol rtol, ScalarTol atol) {
  auto abs_diff = Kokkos::abs(a - b);
  if (abs_diff <= atol) return true;

  // b is a reference
  return abs_diff <= rtol * Kokkos::abs(b);
}

template <typename ScalarTol>
struct AlmostEqualOp {
 private:
  ScalarTol m_rtol;
  ScalarTol m_atol;

 public:
  AlmostEqualOp(ScalarTol rtol = 1.0e-5, ScalarTol atol = 1.0e-8)
      : m_rtol(rtol), m_atol(atol) {}

  template <typename ScalarA, typename ScalarB>
  KOKKOS_INLINE_FUNCTION bool operator()(ScalarA a, ScalarB b) const {
    return are_almost_equal(a, b, m_rtol, m_atol);
  }
};

}  // namespace Impl
}  // namespace Testing
}  // namespace KokkosFFT

#endif
