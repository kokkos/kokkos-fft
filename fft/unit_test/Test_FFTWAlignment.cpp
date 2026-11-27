// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <stdexcept>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Plans.hpp"
#include "KokkosFFT_Transform.hpp"

namespace {
using execution_space = Kokkos::DefaultHostExecutionSpace;

// FFTW plans embed the SIMD alignment of the arrays used to create them:
// executing a plan created on aligned arrays on less-aligned ones is
// undefined (measured to segfault), and throws at run time thanks to the
// check in the exec_plan overloads. The opposite direction (plan created on
// misaligned arrays) is safe: FFTW uses non-SIMD codelets for such plans.
// A fresh Kokkos View is always 64-byte aligned, i.e. as aligned as FFTW
// ever requires; a subview shifted by one element is the misaligned case.

template <typename T>
struct FFTWAlignmentTyped : public ::testing::Test {
  using value_type = T;
};

using precision_types = ::testing::Types<float, double>;

TYPED_TEST_SUITE(FFTWAlignmentTyped, precision_types);

// The plan's alignment measure depends on its precision (fftwf_alignment_of
// for float, fftw_alignment_of for double): a float array shifted by one
// float (4 bytes) is misaligned but invisible to the double-based measure.
// These tests fail if either side of the comparison uses the wrong measure.

TYPED_TEST(FFTWAlignmentTyped, AlignedPlanOnShiftedInputThrows) {
  using value_type = TypeParam;
  const int n = 64;
  execution_space exec;

  Kokkos::View<value_type*, execution_space> x("x", n);
  Kokkos::View<value_type*, execution_space> big("big", n + 1);
  auto xs = Kokkos::subview(big, Kokkos::make_pair(1, n + 1));
  Kokkos::View<Kokkos::complex<value_type>*, execution_space> y("y",
                                                                n / 2 + 1);

  KokkosFFT::Plan plan(exec, x, y, KokkosFFT::Direction::forward,
                       /*axis=*/0);

  // shifted by one element: misaligned for FFTW SIMD
  EXPECT_THROW(KokkosFFT::execute(plan, xs, y), std::runtime_error);
}

TYPED_TEST(FFTWAlignmentTyped, AlignedPlanOnShiftedOutputThrows) {
  using value_type = TypeParam;
  const int n = 64;
  execution_space exec;

  Kokkos::View<Kokkos::complex<value_type>*, execution_space> x("x",
                                                                n / 2 + 1);
  Kokkos::View<value_type*, execution_space> y("y", n);
  Kokkos::View<value_type*, execution_space> big("big", n + 1);
  auto ys = Kokkos::subview(big, Kokkos::make_pair(1, n + 1));

  KokkosFFT::Plan plan(exec, x, y, KokkosFFT::Direction::backward,
                       /*axis=*/0);

  // the output array is checked like the input one
  EXPECT_THROW(KokkosFFT::execute(plan, x, ys), std::runtime_error);
}

// Plans created on misaligned arrays are never rejected, whichever the
// alignment of the execution arrays: the double tests below are the only
// ones whose outcome depends on the alignment recorded at plan creation
// being nonzero, so one precision is enough for them.

TEST(FFTWAlignment, ShiftedPlanOnAlignedDoubleInputDoesNotThrow) {
  const int n = 64;
  execution_space exec;

  Kokkos::View<double*, execution_space> big("big", n + 1);
  auto xs = Kokkos::subview(big, Kokkos::make_pair(1, n + 1));
  Kokkos::View<Kokkos::complex<double>*, execution_space> y("y", n / 2 + 1);

  // plan created on the misaligned subview: FFTW plans on misaligned
  // arrays use non-SIMD codelets and are valid on arrays of any alignment
  KokkosFFT::Plan plan(exec, xs, y, KokkosFFT::Direction::forward,
                       /*axis=*/0);

  Kokkos::View<double*, execution_space> x("x", n);
  EXPECT_NO_THROW(KokkosFFT::execute(plan, x, y));
}

TEST(FFTWAlignment, ShiftedPlanOnSameShiftedViewsDoesNotThrow) {
  const int n = 64;
  execution_space exec;

  Kokkos::View<double*, execution_space> big("big", n + 1);
  auto xs = Kokkos::subview(big, Kokkos::make_pair(1, n + 1));
  Kokkos::View<Kokkos::complex<double>*, execution_space> y("y", n / 2 + 1);

  // the reuse pattern on a fixed misaligned workspace: the plan is created
  // on the shifted subview and executed on the very same views, repeatedly
  KokkosFFT::Plan plan(exec, xs, y, KokkosFFT::Direction::forward,
                       /*axis=*/0);

  EXPECT_NO_THROW(KokkosFFT::execute(plan, xs, y));
  EXPECT_NO_THROW(KokkosFFT::execute(plan, xs, y));
}

TEST(FFTWAlignment, SameAlignedViewsStillWork) {
  // the check must not fire for the normal reuse pattern
  const int n = 64;
  execution_space exec;

  Kokkos::View<double*, execution_space> x("x", n);
  Kokkos::View<Kokkos::complex<double>*, execution_space> y("y", n / 2 + 1);

  KokkosFFT::Plan plan(exec, x, y, KokkosFFT::Direction::forward,
                       /*axis=*/0);
  EXPECT_NO_THROW(KokkosFFT::execute(plan, x, y));
  EXPECT_NO_THROW(KokkosFFT::execute(plan, x, y));
}

}  // namespace
