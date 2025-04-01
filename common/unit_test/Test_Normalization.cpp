// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_normalization.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, execution_space>;
}  // namespace

TEST(Normalization, Forward) {
  const int len = 30;
  View1D<double> x("x", len), ref_f("ref_f", len), ref_b("ref_b", len);

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_f, x);
  Kokkos::deep_copy(ref_b, x);

  double coef = 1.0 / static_cast<double>(len);
  multiply(execution_space(), ref_f, coef);

  Kokkos::fence();

  // Backward FFT with forward Normalization -> Do nothing
  KokkosFFT::Impl::normalize(execution_space(), x,
                             KokkosFFT::Direction::backward,
                             KokkosFFT::Normalization::forward, len);
  EXPECT_TRUE(allclose(execution_space(), x, ref_b, 1.e-5, 1.e-12));

  // Forward FFT with forward Normalization -> 1/N normalization
  KokkosFFT::Impl::normalize(execution_space(), x,
                             KokkosFFT::Direction::forward,
                             KokkosFFT::Normalization::forward, len);
  EXPECT_TRUE(allclose(execution_space(), x, ref_f, 1.e-5, 1.e-12));
}

TEST(Normalization, Backward) {
  const int len = 30;
  View1D<double> x("x", len), ref_f("ref_f", len), ref_b("ref_b", len);

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_f, x);
  Kokkos::deep_copy(ref_b, x);

  double coef = 1.0 / static_cast<double>(len);
  multiply(execution_space(), ref_b, coef);

  Kokkos::fence();

  // Forward FFT with backward Normalization -> Do nothing
  KokkosFFT::Impl::normalize(execution_space(), x,
                             KokkosFFT::Direction::forward,
                             KokkosFFT::Normalization::backward, len);
  EXPECT_TRUE(allclose(execution_space(), x, ref_f, 1.e-5, 1.e-12));

  // Backward FFT with backward Normalization -> 1/N normalization
  KokkosFFT::Impl::normalize(execution_space(), x,
                             KokkosFFT::Direction::backward,
                             KokkosFFT::Normalization::backward, len);
  EXPECT_TRUE(allclose(execution_space(), x, ref_b, 1.e-5, 1.e-12));
}

TEST(Normalization, Ortho) {
  const int len = 30;
  View1D<double> x_f("x_f", len), x_b("x_b", len);
  View1D<double> ref_f("ref_f", len), ref_b("ref_b", len);

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(x_f, random_pool, 1.0);

  Kokkos::deep_copy(x_b, x_f);
  Kokkos::deep_copy(ref_f, x_f);
  Kokkos::deep_copy(ref_b, x_f);

  double coef = 1.0 / Kokkos::sqrt(static_cast<double>(len));
  multiply(execution_space(), ref_f, coef);
  multiply(execution_space(), ref_b, coef);

  Kokkos::fence();

  // Forward FFT with ortho Normalization -> 1 / sqrt(N) normalization
  KokkosFFT::Impl::normalize(execution_space(), x_f,
                             KokkosFFT::Direction::forward,
                             KokkosFFT::Normalization::ortho, len);
  EXPECT_TRUE(allclose(execution_space(), x_f, ref_f, 1.e-5, 1.e-12));

  // Backward FFT with ortho Normalization -> 1 / sqrt(N) normalization
  KokkosFFT::Impl::normalize(execution_space(), x_b,
                             KokkosFFT::Direction::backward,
                             KokkosFFT::Normalization::ortho, len);
  EXPECT_TRUE(allclose(execution_space(), x_b, ref_b, 1.e-5, 1.e-12));
}

TEST(Normalization, None) {
  const int len = 30;
  View1D<double> x_f("x_f", len), x_b("x_b", len);
  View1D<double> ref_f("ref_f", len), ref_b("ref_b", len);

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(x_f, random_pool, 1.0);

  Kokkos::deep_copy(x_b, x_f);
  Kokkos::deep_copy(ref_f, x_f);
  Kokkos::deep_copy(ref_b, x_f);

  Kokkos::fence();

  // Forward FFT with none Normalization -> Do nothing
  KokkosFFT::Impl::normalize(execution_space(), x_f,
                             KokkosFFT::Direction::forward,
                             KokkosFFT::Normalization::none, len);
  EXPECT_TRUE(allclose(execution_space(), x_f, ref_f, 1.e-5, 1.e-12));

  // Backward FFT with none Normalization -> Do nothing
  KokkosFFT::Impl::normalize(execution_space(), x_b,
                             KokkosFFT::Direction::backward,
                             KokkosFFT::Normalization::none, len);
  EXPECT_TRUE(allclose(execution_space(), x_b, ref_b, 1.e-5, 1.e-12));
}
