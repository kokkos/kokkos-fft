// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Transform.hpp"
#include "Test_Utils.hpp"

namespace {
#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
using execution_space = Kokkos::DefaultExecutionSpace;
#else
using execution_space = Kokkos::DefaultHostExecutionSpace;
#endif

template <std::size_t DIM>
using shape_type = KokkosFFT::shape_type<DIM>;

using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct FFT1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct FFT2D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename T>
struct FFTND : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

// Tests for 1D FFT
template <typename T, typename LayoutType>
void test_fft1_identity(T atol = 1.0e-12) {
  const int maxlen     = 32;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;
  execution_space exec;
  for (int i = 1; i < maxlen; i++) {
    ComplexView1DType a("a", i), a_ref("a_ref", i);
    ComplexView1DType a_hat("a_hat", i), inv_a_hat("inv_a_hat", i);

    // Used for Real transforms
    RealView1DType ar("ar", i), inv_ar_hat("inv_ar_hat", i),
        ar_ref("ar_ref", i);
    ComplexView1DType ar_hat("ar_hat", i / 2 + 1);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(/*seed=*/12345);
    Kokkos::fill_random(exec, a, random_pool, z);
    Kokkos::fill_random(exec, ar, random_pool, 1.0);
    exec.fence();

    Kokkos::deep_copy(a_ref, a);
    Kokkos::deep_copy(ar_ref, ar);

    KokkosFFT::fft(exec, a, a_hat);
    KokkosFFT::ifft(exec, a_hat, inv_a_hat);

    KokkosFFT::rfft(exec, ar, ar_hat);
    KokkosFFT::irfft(exec, ar_hat, inv_ar_hat);

    EXPECT_TRUE(allclose(exec, inv_a_hat, a_ref, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, inv_ar_hat, ar_ref, 1.e-5, atol));
    exec.fence();
  }
}

template <typename T, typename LayoutType>
void test_fft1_identity_inplace(T atol = 1.0e-12) {
  const int maxlen     = 32;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;
  execution_space exec;
  for (int i = 1; i < maxlen; i++) {
    ComplexView1DType a("a", i), a_ref("a_ref", i);
    ComplexView1DType a_hat(a.data(), i), inv_a_hat(a.data(), i);

    // Used for Real transforms
    ComplexView1DType ar_hat("ar_hat", i / 2 + 1);
    RealView1DType ar(reinterpret_cast<T*>(ar_hat.data()), i),
        inv_ar_hat(reinterpret_cast<T*>(ar_hat.data()), i);
    RealView1DType ar_ref("ar_ref", i);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(/*seed=*/12345);
    Kokkos::fill_random(exec, a, random_pool, z);
    Kokkos::fill_random(exec, ar, random_pool, 1.0);
    exec.fence();

    Kokkos::deep_copy(a_ref, a);
    Kokkos::deep_copy(ar_ref, ar);

    KokkosFFT::fft(exec, a, a_hat);
    KokkosFFT::ifft(exec, a_hat, inv_a_hat);

    KokkosFFT::rfft(exec, ar, ar_hat);
    KokkosFFT::irfft(exec, ar_hat, inv_ar_hat);

    exec.fence();

    EXPECT_TRUE(allclose(exec, inv_a_hat, a_ref, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, inv_ar_hat, ar_ref, 1.e-5, atol));

    // Create a plan for inplace transform
    Kokkos::deep_copy(a_ref, a);
    Kokkos::deep_copy(ar_ref, ar);

    int axis = -1;
    KokkosFFT::Plan fft_plan(exec, a, a_hat, KokkosFFT::Direction::forward,
                             axis);
    KokkosFFT::execute(fft_plan, a, a_hat);

    KokkosFFT::Plan ifft_plan(exec, a_hat, inv_a_hat,
                              KokkosFFT::Direction::backward, axis);
    KokkosFFT::execute(ifft_plan, a_hat, inv_a_hat);

    KokkosFFT::Plan rfft_plan(exec, ar, ar_hat, KokkosFFT::Direction::forward,
                              axis);
    KokkosFFT::execute(rfft_plan, ar, ar_hat);

    KokkosFFT::Plan irfft_plan(exec, ar_hat, inv_ar_hat,
                               KokkosFFT::Direction::backward, axis);
    KokkosFFT::execute(irfft_plan, ar_hat, inv_ar_hat);

    EXPECT_TRUE(allclose(exec, inv_a_hat, a_ref, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, inv_ar_hat, ar_ref, 1.e-5, atol));

    // inplace Plan cannot be reused for out-of-place case
    ComplexView1DType a_hat_out("a_hat_out", i),
        inv_a_hat_out("inv_a_hat_out", i);

    RealView1DType inv_ar_hat_out("inv_ar_hat_out", i);
    ComplexView1DType ar_hat_out("ar_hat_out", i / 2 + 1);

    EXPECT_THROW(KokkosFFT::execute(fft_plan, a, a_hat_out),
                 std::runtime_error);
    EXPECT_THROW(KokkosFFT::execute(ifft_plan, a_hat_out, inv_a_hat_out),
                 std::runtime_error);
    EXPECT_THROW(KokkosFFT::execute(rfft_plan, ar, ar_hat_out),
                 std::runtime_error);
    EXPECT_THROW(KokkosFFT::execute(irfft_plan, ar_hat_out, inv_ar_hat_out),
                 std::runtime_error);
    exec.fence();
  }
}

template <typename T, typename LayoutType>
void test_fft1_identity_reuse_plan(T atol = 1.0e-12) {
  const int maxlen     = 32;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;
  execution_space exec;
  for (int i = 1; i < maxlen; i++) {
    ComplexView1DType a("a", i), a_ref("a_ref", i);
    ComplexView1DType a_hat("a_hat", i), inv_a_hat("inv_a_hat", i);

    // Used for Real transforms
    RealView1DType ar("ar", i), inv_ar_hat("inv_ar_hat", i),
        ar_ref("ar_ref", i);
    ComplexView1DType ar_hat("ar_hat", i / 2 + 1);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(/*seed=*/12345);
    Kokkos::fill_random(exec, a, random_pool, z);
    Kokkos::fill_random(exec, ar, random_pool, 1.0);
    exec.fence();

    Kokkos::deep_copy(a_ref, a);
    Kokkos::deep_copy(ar_ref, ar);

    int axis = -1;
    KokkosFFT::Plan fft_plan(exec, a, a_hat, KokkosFFT::Direction::forward,
                             axis);
    KokkosFFT::execute(fft_plan, a, a_hat);

    KokkosFFT::Plan ifft_plan(exec, a_hat, inv_a_hat,
                              KokkosFFT::Direction::backward, axis);
    KokkosFFT::execute(ifft_plan, a_hat, inv_a_hat);

    KokkosFFT::Plan rfft_plan(exec, ar, ar_hat, KokkosFFT::Direction::forward,
                              axis);
    KokkosFFT::execute(rfft_plan, ar, ar_hat);

    KokkosFFT::Plan irfft_plan(exec, ar_hat, inv_ar_hat,
                               KokkosFFT::Direction::backward, axis);
    KokkosFFT::execute(irfft_plan, ar_hat, inv_ar_hat);
    exec.fence();

    EXPECT_TRUE(allclose(exec, inv_a_hat, a_ref, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, inv_ar_hat, ar_ref, 1.e-5, atol));
    exec.fence();
  }

  ComplexView1DType a("a", maxlen), a_ref("a_ref", maxlen);
  ComplexView1DType a_hat("a_hat", maxlen), inv_a_hat("inv_a_hat", maxlen);

  // Used for Real transforms
  RealView1DType ar("ar", maxlen), inv_ar_hat("inv_ar_hat", maxlen),
      ar_ref("ar_ref", maxlen);
  ComplexView1DType ar_hat("ar_hat", maxlen / 2 + 1);

  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(/*seed=*/12345);
  Kokkos::fill_random(exec, a, random_pool, z);
  Kokkos::fill_random(exec, ar, random_pool, 1.0);
  exec.fence();

  Kokkos::deep_copy(a_ref, a);
  Kokkos::deep_copy(ar_ref, ar);

  // Create correct plans
  int axis = -1;
  KokkosFFT::Plan fft_plan(exec, a, a_hat, KokkosFFT::Direction::forward, axis);

  KokkosFFT::Plan ifft_plan(exec, a_hat, inv_a_hat,
                            KokkosFFT::Direction::backward, axis);

  KokkosFFT::Plan rfft_plan(exec, ar, ar_hat, KokkosFFT::Direction::forward,
                            axis);

  KokkosFFT::Plan irfft_plan(exec, ar_hat, inv_ar_hat,
                             KokkosFFT::Direction::backward, axis);

  // Check if errors are correctly raised against wrong extents
  const int maxlen_wrong = 32 * 2;
  ComplexView1DType a_wrong("a_wrong", maxlen_wrong),
      inv_a_hat_wrong("inv_a_hat_wrong", maxlen_wrong);
  ComplexView1DType a_hat_wrong("a_hat_wrong", maxlen_wrong),
      ar_hat_wrong("ar_hat_wrong", maxlen_wrong / 2 + 1);
  RealView1DType ar_wrong("ar_wrong", maxlen_wrong),
      inv_ar_hat_wrong("inv_ar_hat_wrong", maxlen_wrong);

  // fft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::execute(fft_plan, a_wrong, a_hat,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::execute(fft_plan, a, a_hat_wrong,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // ifft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::execute(ifft_plan, a_hat_wrong, inv_a_hat,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::execute(ifft_plan, a_hat, inv_a_hat_wrong,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // rfft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::execute(rfft_plan, ar_wrong, ar_hat,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::execute(rfft_plan, ar, ar_hat_wrong,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // irfft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::execute(irfft_plan, ar_hat_wrong, inv_ar_hat,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::execute(irfft_plan, ar_hat, inv_ar_hat_wrong,
                                  KokkosFFT::Normalization::backward),
               std::runtime_error);
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_1dview() {
  const int len = 30;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x("x", len), out("out", len), ref("ref", len);
  ComplexView1DType out_b("out_b", len), out_o("out_o", len),
      out_f("out_f", len);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  KokkosFFT::fft(exec, x, out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fft(exec, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::fft(exec, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::fft(exec, x, out_f, KokkosFFT::Normalization::forward);

  fft1(exec, x, ref);
  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(len)));
  multiply(exec, out_f, static_cast<T>(len));

  EXPECT_TRUE(allclose(exec, out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, ref, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft1_1difft_1dview() {
  const int len = 30;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x("x", len), out("out", len), ref("ref", len);
  ComplexView1DType out_b("out_b", len), out_o("out_o", len),
      out_f("out_f", len);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);

  Kokkos::fence();

  KokkosFFT::ifft(exec, x, out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifft(exec, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::ifft(exec, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::ifft(exec, x, out_f, KokkosFFT::Normalization::forward);

  ifft1(exec, x, ref);
  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(len)));
  multiply(exec, out_b, static_cast<T>(len));
  multiply(exec, out, static_cast<T>(len));

  EXPECT_TRUE(allclose(exec, out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, ref, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft1_1dhfft_1dview() {
  const int len_herm = 16, len = len_herm * 2 - 2;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x_herm("x_herm", len_herm),
      x_herm_ref("x_herm_ref", len_herm);
  ComplexView1DType x("x", len), ref("ref", len);
  RealView1DType out("out", len);
  RealView1DType out_b("out_b", len), out_o("out_o", len), out_f("out_f", len);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x_herm, random_pool, z);
  exec.fence();

  auto h_x      = Kokkos::create_mirror_view(x);
  auto h_x_herm = Kokkos::create_mirror_view(x_herm);
  Kokkos::deep_copy(h_x_herm, x_herm);

  auto last      = h_x_herm.extent(0) - 1;
  h_x_herm(0)    = h_x_herm(0).real();
  h_x_herm(last) = h_x_herm(last).real();

  for (int i = 0; i < len_herm; i++) {
    h_x(i) = h_x_herm(i);
  }

  // h_x_herm(last-1), h_x_herm(last-2), ..., h_x_herm(1)
  for (int i = last - 1; i > 0; i--) {
    h_x(len - i) = Kokkos::conj(h_x_herm(i));
  }

  Kokkos::deep_copy(x_herm, h_x_herm);
  Kokkos::deep_copy(x_herm_ref, h_x_herm);
  Kokkos::deep_copy(x, h_x);

  KokkosFFT::fft(exec, x, ref);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm,
                  out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(len)));
  multiply(exec, out_f, static_cast<T>(len));

  EXPECT_TRUE(allclose(exec, out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft1_1dihfft_1dview() {
  const int len_herm = 16, len = len_herm * 2 - 2;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x_herm("x_herm", len_herm),
      x_herm_ref("x_herm_ref", len_herm);
  RealView1DType out1("out1", len);
  RealView1DType out1_b("out1_b", len), out1_o("out1_o", len),
      out1_f("out1_f", len);
  ComplexView1DType out2("out2", len_herm), out2_b("out2_b", len_herm),
      out2_o("out2_o", len_herm), out2_f("out2_f", len_herm);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x_herm, random_pool, z);
  exec.fence();

  auto h_x_herm = Kokkos::create_mirror_view(x_herm);
  Kokkos::deep_copy(h_x_herm, x_herm);

  auto last      = h_x_herm.extent(0) - 1;
  h_x_herm(0)    = h_x_herm(0).real();
  h_x_herm(last) = h_x_herm(last).real();

  Kokkos::deep_copy(x_herm, h_x_herm);
  Kokkos::deep_copy(x_herm_ref, h_x_herm);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm,
                  out1);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ihfft(exec, out1,
                   out2);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm, out1_b, KokkosFFT::Normalization::backward);
  KokkosFFT::ihfft(exec, out1_b, out2_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm, out1_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::ihfft(exec, out1_o, out2_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(exec, x_herm, out1_f, KokkosFFT::Normalization::forward);
  KokkosFFT::ihfft(exec, out1_f, out2_f, KokkosFFT::Normalization::forward);

  EXPECT_TRUE(allclose(exec, out2, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out2_b, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out2_o, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out2_f, x_herm_ref, 1.e-5, 1.e-6));
}

template <typename T, typename LayoutType>
void test_fft1_shape(T atol = 1.0e-12) {
  const int n          = 32;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  RealView1DType xr("xr", n), xr_ref("xr_ref", n);
  ComplexView1DType x("x", n / 2 + 1), x_ref("x_ref", n / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(/*seed=*/12345);
  Kokkos::fill_random(exec, xr, random_pool, 1.0);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(x_ref, x);
  Kokkos::deep_copy(xr_ref, xr);

  std::vector<int> shapes = {n / 2, n, n * 2};
  for (auto&& shape : shapes) {
    // Real to complex
    ComplexView1DType outr("outr", shape / 2 + 1),
        outr_b("outr_b", shape / 2 + 1), outr_o("outr_o", shape / 2 + 1),
        outr_f("outr_f", shape / 2 + 1);

    Kokkos::deep_copy(xr, xr_ref);
    KokkosFFT::rfft(exec, xr, outr, KokkosFFT::Normalization::none, -1, shape);

    Kokkos::deep_copy(xr, xr_ref);
    KokkosFFT::rfft(exec, xr, outr_b, KokkosFFT::Normalization::backward, -1,
                    shape);

    Kokkos::deep_copy(xr, xr_ref);
    KokkosFFT::rfft(exec, xr, outr_o, KokkosFFT::Normalization::ortho, -1,
                    shape);

    Kokkos::deep_copy(xr, xr_ref);
    KokkosFFT::rfft(exec, xr, outr_f, KokkosFFT::Normalization::forward, -1,
                    shape);

    multiply(exec, outr_o, Kokkos::sqrt(static_cast<T>(shape)));
    multiply(exec, outr_f, static_cast<T>(shape));

    EXPECT_TRUE(allclose(exec, outr_b, outr, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, outr_o, outr, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, outr_f, outr, 1.e-5, atol));

    // Complex to real
    RealView1DType out("out", shape), out_b("out_b", shape),
        out_o("out_o", shape), out_f("out_f", shape);

    Kokkos::deep_copy(x, x_ref);
    KokkosFFT::irfft(exec, x, out, KokkosFFT::Normalization::none, -1, shape);

    Kokkos::deep_copy(x, x_ref);
    KokkosFFT::irfft(exec, x, out_b, KokkosFFT::Normalization::backward, -1,
                     shape);

    Kokkos::deep_copy(x, x_ref);
    KokkosFFT::irfft(exec, x, out_o, KokkosFFT::Normalization::ortho, -1,
                     shape);

    Kokkos::deep_copy(x, x_ref);
    KokkosFFT::irfft(exec, x, out_f, KokkosFFT::Normalization::forward, -1,
                     shape);

    multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(shape)));
    multiply(exec, out_b, static_cast<T>(shape));

    EXPECT_TRUE(allclose(exec, out_b, out, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, out_o, out, 1.e-5, atol));
    EXPECT_TRUE(allclose(exec, out_f, out, 1.e-5, atol));
    exec.fence();
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_2dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 12;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1), ref_x("ref_x", n0, n1);
  ComplexView2DType x_axis0("x_axis0", n0, n1), x_axis1("x_axis1", n0, n1);
  ComplexView2DType out_axis0("out_axis0", n0, n1),
      ref_out_axis0("ref_out_axis0", n0, n1);
  ComplexView2DType out_axis1("out_axis1", n0, n1),
      ref_out_axis1("ref_out_axis1", n0, n1);

  RealView2DType xr("xr", n0, n1), ref_xr("ref_xr", n0, n1);
  RealView2DType xr_axis0("xr_axis0", n0, n1), xr_axis1("xr_axis1", n0, n1);
  ComplexView2DType outr_axis0("outr_axis0", n0 / 2 + 1, n1),
      outr_axis1("outr_axis1", n0, n1 / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  Kokkos::fill_random(exec, xr, random_pool, 1);
  exec.fence();

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  // Along axis 0 (transpose needed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for (int i1 = 0; i1 < n1; i1++) {
    auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1);
    auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1);
    fft1(exec, sub_x, sub_ref);
  }

  KokkosFFT::fft(exec, x, out_axis0, KokkosFFT::Normalization::backward,
                 /*axis=*/0);
  EXPECT_TRUE(allclose(exec, out_axis0, ref_out_axis0, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis0, x_axis0, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  EXPECT_TRUE(allclose(exec, x_axis0, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis0, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  KokkosFFT::irfft(exec, outr_axis0, xr_axis0,
                   KokkosFFT::Normalization::backward, /*axis=*/0);

  EXPECT_TRUE(allclose(exec, xr_axis0, ref_xr, 1.e-5, atol));
  exec.fence();

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1
  // Perform batched 1D (along 1st axis) FFT sequentially
  for (int i0 = 0; i0 < n0; i0++) {
    auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL);
    auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL);
    fft1(exec, sub_x, sub_ref);
  }

  KokkosFFT::fft(exec, x, out_axis1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  EXPECT_TRUE(allclose(exec, out_axis1, ref_out_axis1, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis1, x_axis1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  EXPECT_TRUE(allclose(exec, x_axis1, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::irfft(exec, outr_axis1, xr_axis1,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  EXPECT_TRUE(allclose(exec, xr_axis1, ref_xr, 1.e-5, atol));
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_3dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 12, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2), ref_x("ref_x", n0, n1, n2);
  ComplexView3DType x_axis0("x_axis0", n0, n1, n2),
      x_axis1("x_axis1", n0, n1, n2), x_axis2("x_axis2", n0, n1, n2);
  ComplexView3DType out_axis0("out_axis0", n0, n1, n2),
      ref_out_axis0("ref_out_axis0", n0, n1, n2);
  ComplexView3DType out_axis1("out_axis1", n0, n1, n2),
      ref_out_axis1("ref_out_axis1", n0, n1, n2);
  ComplexView3DType out_axis2("out_axis2", n0, n1, n2),
      ref_out_axis2("ref_out_axis2", n0, n1, n2);

  RealView3DType xr("xr", n0, n1, n2), ref_xr("ref_xr", n0, n1, n2);
  RealView3DType xr_axis0("xr_axis0", n0, n1, n2),
      xr_axis1("xr_axis1", n0, n1, n2), xr_axis2("xr_axis2", n0, n1, n2);
  ComplexView3DType outr_axis0("outr_axis0", n0 / 2 + 1, n1, n2),
      outr_axis1("outr_axis1", n0, n1 / 2 + 1, n2),
      outr_axis2("outr_axis2", n0, n1, n2 / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  Kokkos::fill_random(exec, xr, random_pool, 1);
  exec.fence();

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  // Along axis 0 (transpose needed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1, i2);
      auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1, i2);
      fft1(exec, sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(exec, x, out_axis0, KokkosFFT::Normalization::backward,
                 /*axis=*/0);
  EXPECT_TRUE(allclose(exec, out_axis0, ref_out_axis0, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis0, x_axis0, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  EXPECT_TRUE(allclose(exec, x_axis0, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis0, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  KokkosFFT::irfft(exec, outr_axis0, xr_axis0,
                   KokkosFFT::Normalization::backward, /*axis=*/0);

  EXPECT_TRUE(allclose(exec, xr_axis0, ref_xr, 1.e-5, atol));
  exec.fence();

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1 (transpose needed)
  // Perform batched 1D (along 1st axis) FFT sequentially
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i0 = 0; i0 < n0; i0++) {
      auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL, i2);
      auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL, i2);
      fft1(exec, sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(exec, x, out_axis1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  EXPECT_TRUE(allclose(exec, out_axis1, ref_out_axis1, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis1, x_axis1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  EXPECT_TRUE(allclose(exec, x_axis1, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::irfft(exec, outr_axis1, xr_axis1,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  EXPECT_TRUE(allclose(exec, xr_axis1, ref_xr, 1.e-5, atol));
  exec.fence();

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 2
  // Perform batched 1D (along 2nd axis) FFT sequentially
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      auto sub_x   = Kokkos::subview(x, i0, i1, Kokkos::ALL);
      auto sub_ref = Kokkos::subview(ref_out_axis2, i0, i1, Kokkos::ALL);
      fft1(exec, sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(exec, x, out_axis2, KokkosFFT::Normalization::backward,
                 /*axis=*/2);
  EXPECT_TRUE(allclose(exec, out_axis2, ref_out_axis2, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis2, x_axis2, KokkosFFT::Normalization::backward,
                  /*axis=*/2);
  EXPECT_TRUE(allclose(exec, x_axis2, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis2, KokkosFFT::Normalization::backward,
                  /*axis=*/2);
  KokkosFFT::irfft(exec, outr_axis2, xr_axis2,
                   KokkosFFT::Normalization::backward, /*axis=*/2);

  EXPECT_TRUE(allclose(exec, xr_axis2, ref_xr, 1.e-5, atol));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_4dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 12, n2 = 8, n3 = 5;
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;

  ComplexView4DType x("x", n0, n1, n2, n3), ref_x("ref_x", n0, n1, n2, n3);
  ComplexView4DType x_axis0("x_axis0", n0, n1, n2, n3),
      x_axis1("x_axis1", n0, n1, n2, n3), x_axis2("x_axis2", n0, n1, n2, n3),
      x_axis3("x_axis3", n0, n1, n2, n3);
  ComplexView4DType out_axis0("out_axis0", n0, n1, n2, n3),
      ref_out_axis0("ref_out_axis0", n0, n1, n2, n3);
  ComplexView4DType out_axis1("out_axis1", n0, n1, n2, n3),
      ref_out_axis1("ref_out_axis1", n0, n1, n2, n3);
  ComplexView4DType out_axis2("out_axis2", n0, n1, n2, n3),
      ref_out_axis2("ref_out_axis2", n0, n1, n2, n3);
  ComplexView4DType out_axis3("out_axis3", n0, n1, n2, n3),
      ref_out_axis3("ref_out_axis3", n0, n1, n2, n3);

  RealView4DType xr("xr", n0, n1, n2, n3), ref_xr("ref_xr", n0, n1, n2, n3);
  RealView4DType xr_axis0("xr_axis0", n0, n1, n2, n3),
      xr_axis1("xr_axis1", n0, n1, n2, n3),
      xr_axis2("xr_axis2", n0, n1, n2, n3),
      xr_axis3("xr_axis3", n0, n1, n2, n3);
  ComplexView4DType outr_axis0("outr_axis0", n0 / 2 + 1, n1, n2, n3),
      outr_axis1("outr_axis1", n0, n1 / 2 + 1, n2, n3),
      outr_axis2("outr_axis2", n0, n1, n2 / 2 + 1, n3),
      outr_axis3("outr_axis3", n0, n1, n2, n3 / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  Kokkos::fill_random(exec, xr, random_pool, 1);
  exec.fence();

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  // Along axis 0 (transpose needed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1, i2, i3);
        auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1, i2, i3);
        fft1(exec, sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(exec, x, out_axis0, KokkosFFT::Normalization::backward,
                 /*axis=*/0);
  EXPECT_TRUE(allclose(exec, out_axis0, ref_out_axis0, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis0, x_axis0, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  EXPECT_TRUE(allclose(exec, x_axis0, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis0, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  KokkosFFT::irfft(exec, outr_axis0, xr_axis0,
                   KokkosFFT::Normalization::backward, /*axis=*/0);

  EXPECT_TRUE(allclose(exec, xr_axis0, ref_xr, 1.e-5, atol));
  exec.fence();

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1 (transpose needed)
  // Perform batched 1D (along 1st axis) FFT sequentially
  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i0 = 0; i0 < n0; i0++) {
        auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL, i2, i3);
        auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL, i2, i3);
        fft1(exec, sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(exec, x, out_axis1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  EXPECT_TRUE(allclose(exec, out_axis1, ref_out_axis1, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis1, x_axis1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  EXPECT_TRUE(allclose(exec, x_axis1, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::irfft(exec, outr_axis1, xr_axis1,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  EXPECT_TRUE(allclose(exec, xr_axis1, ref_xr, 1.e-5, atol));

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 2
  // Perform batched 1D (along 2nd axis) FFT sequentially
  for (int i3 = 0; i3 < n3; i3++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i0 = 0; i0 < n0; i0++) {
        auto sub_x   = Kokkos::subview(x, i0, i1, Kokkos::ALL, i3);
        auto sub_ref = Kokkos::subview(ref_out_axis2, i0, i1, Kokkos::ALL, i3);
        fft1(exec, sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(exec, x, out_axis2, KokkosFFT::Normalization::backward,
                 /*axis=*/2);
  EXPECT_TRUE(allclose(exec, out_axis2, ref_out_axis2, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis2, x_axis2, KokkosFFT::Normalization::backward,
                  /*axis=*/2);
  EXPECT_TRUE(allclose(exec, x_axis2, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis2, KokkosFFT::Normalization::backward,
                  /*axis=*/2);
  KokkosFFT::irfft(exec, outr_axis2, xr_axis2,
                   KokkosFFT::Normalization::backward, /*axis=*/2);

  EXPECT_TRUE(allclose(exec, xr_axis2, ref_xr, 1.e-5, atol));
  exec.fence();

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 3
  // Perform batched 1D (along 3rd axis) FFT sequentially
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      for (int i0 = 0; i0 < n0; i0++) {
        auto sub_x   = Kokkos::subview(x, i0, i1, i2, Kokkos::ALL);
        auto sub_ref = Kokkos::subview(ref_out_axis3, i0, i1, i2, Kokkos::ALL);
        fft1(exec, sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(exec, x, out_axis3, KokkosFFT::Normalization::backward,
                 /*axis=*/3);
  EXPECT_TRUE(allclose(exec, out_axis3, ref_out_axis3, 1.e-5, atol));

  KokkosFFT::ifft(exec, out_axis3, x_axis3, KokkosFFT::Normalization::backward,
                  /*axis=*/3);
  EXPECT_TRUE(allclose(exec, x_axis3, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(exec, xr, outr_axis3, KokkosFFT::Normalization::backward,
                  /*axis=*/3);
  KokkosFFT::irfft(exec, outr_axis3, xr_axis3,
                   KokkosFFT::Normalization::backward, /*axis=*/3);

  EXPECT_TRUE(allclose(exec, xr_axis3, ref_xr, 1.e-5, atol));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_5dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4;
  using RealView5DType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ComplexView5DType =
      Kokkos::View<Kokkos::complex<T>*****, LayoutType, execution_space>;

  constexpr int DIM = 5;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4});
  ComplexView5DType x("x", n0, n1, n2, n3, n4);

  execution_space exec;
  for (int axis = 0; axis < DIM; axis++) {
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<DIM> shape          = default_shape;
      shape_type<DIM> shape_c2r      = default_shape;
      const std::size_t n_new        = shape.at(axis) + i0;
      shape.at(axis)                 = n_new;
      shape_c2r.at(axis)             = n_new / 2 + 1;
      auto [s0, s1, s2, s3, s4]      = shape;
      auto [sr0, sr1, sr2, sr3, sr4] = shape_c2r;
      ComplexView5DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4),
          x_hat("x_hat", s0, s1, s2, s3, s4),
          ref_x("ref_x", s0, s1, s2, s3, s4);
      RealView5DType xr("xr", s0, s1, s2, s3, s4),
          inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4),
          ref_xr("ref_xr", s0, s1, s2, s3, s4);
      ComplexView5DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4);

      const Kokkos::complex<T> z(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
      Kokkos::fill_random(exec, x, random_pool, z);
      Kokkos::fill_random(exec, xr, random_pool, 1);

      KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
      KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft(exec, x, x_hat, KokkosFFT::Normalization::backward, axis,
                     n_new);
      KokkosFFT::ifft(exec, x_hat, inv_x_hat,
                      KokkosFFT::Normalization::backward, axis, n_new);
      EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                      axis, n_new);
      KokkosFFT::irfft(exec, xr_hat, inv_xr_hat,
                       KokkosFFT::Normalization::backward, axis, n_new);

      EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
      exec.fence();
    }
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_6dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3;
  using RealView6DType = Kokkos::View<T******, LayoutType, execution_space>;
  using ComplexView6DType =
      Kokkos::View<Kokkos::complex<T>******, LayoutType, execution_space>;

  constexpr int DIM = 6;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4, n5});
  ComplexView6DType x("x", n0, n1, n2, n3, n4, n5);

  execution_space exec;
  for (int axis = 0; axis < DIM; axis++) {
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<DIM> shape               = default_shape;
      shape_type<DIM> shape_c2r           = default_shape;
      const std::size_t n_new             = shape.at(axis) + i0;
      shape.at(axis)                      = n_new;
      shape_c2r.at(axis)                  = n_new / 2 + 1;
      auto [s0, s1, s2, s3, s4, s5]       = shape;
      auto [sr0, sr1, sr2, sr3, sr4, sr5] = shape_c2r;
      ComplexView6DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4, s5),
          x_hat("x_hat", s0, s1, s2, s3, s4, s5),
          ref_x("ref_x", s0, s1, s2, s3, s4, s5);
      RealView6DType xr("xr", s0, s1, s2, s3, s4, s5),
          inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4, s5),
          ref_xr("ref_xr", s0, s1, s2, s3, s4, s5);
      ComplexView6DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5);

      const Kokkos::complex<T> z(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
      Kokkos::fill_random(exec, x, random_pool, z);
      Kokkos::fill_random(exec, xr, random_pool, 1);

      KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
      KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft(exec, x, x_hat, KokkosFFT::Normalization::backward, axis,
                     n_new);
      KokkosFFT::ifft(exec, x_hat, inv_x_hat,
                      KokkosFFT::Normalization::backward, axis, n_new);
      EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                      axis, n_new);
      KokkosFFT::irfft(exec, xr_hat, inv_xr_hat,
                       KokkosFFT::Normalization::backward, axis, n_new);

      EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
      exec.fence();
    }
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_7dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4;
  using RealView7DType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ComplexView7DType =
      Kokkos::View<Kokkos::complex<T>*******, LayoutType, execution_space>;

  constexpr int DIM = 7;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4, n5, n6});
  ComplexView7DType x("x", n0, n1, n2, n3, n4, n5, n6);

  execution_space exec;
  for (int axis = 0; axis < DIM; axis++) {
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<DIM> shape                    = default_shape;
      shape_type<DIM> shape_c2r                = default_shape;
      const std::size_t n_new                  = shape.at(axis) + i0;
      shape.at(axis)                           = n_new;
      shape_c2r.at(axis)                       = n_new / 2 + 1;
      auto [s0, s1, s2, s3, s4, s5, s6]        = shape;
      auto [sr0, sr1, sr2, sr3, sr4, sr5, sr6] = shape_c2r;
      ComplexView7DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4, s5, s6),
          x_hat("x_hat", s0, s1, s2, s3, s4, s5, s6),
          ref_x("ref_x", s0, s1, s2, s3, s4, s5, s6);
      RealView7DType xr("xr", s0, s1, s2, s3, s4, s5, s6),
          inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4, s5, s6),
          ref_xr("ref_xr", s0, s1, s2, s3, s4, s5, s6);
      ComplexView7DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5, sr6);

      const Kokkos::complex<T> z(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
      Kokkos::fill_random(exec, x, random_pool, z);
      Kokkos::fill_random(exec, xr, random_pool, 1);

      KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
      KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft(exec, x, x_hat, KokkosFFT::Normalization::backward, axis,
                     n_new);
      KokkosFFT::ifft(exec, x_hat, inv_x_hat,
                      KokkosFFT::Normalization::backward, axis, n_new);
      EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                      axis, n_new);
      KokkosFFT::irfft(exec, xr_hat, inv_xr_hat,
                       KokkosFFT::Normalization::backward, axis, n_new);

      EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
      exec.fence();
    }
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_8dview(T atol = 1.e-12) {
  const int n0 = 4, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4, n7 = 3;
  using RealView8DType = Kokkos::View<T********, LayoutType, execution_space>;
  using ComplexView8DType =
      Kokkos::View<Kokkos::complex<T>********, LayoutType, execution_space>;

  constexpr int DIM = 8;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4, n5, n6, n7});
  ComplexView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  execution_space exec;
  for (int axis = 0; axis < DIM; axis++) {
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<DIM> shape                         = default_shape;
      shape_type<DIM> shape_c2r                     = default_shape;
      const std::size_t n_new                       = shape.at(axis) + i0;
      shape.at(axis)                                = n_new;
      shape_c2r.at(axis)                            = n_new / 2 + 1;
      auto [s0, s1, s2, s3, s4, s5, s6, s7]         = shape;
      auto [sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7] = shape_c2r;
      ComplexView8DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4, s5, s6, s7),
          x_hat("x_hat", s0, s1, s2, s3, s4, s5, s6, s7),
          ref_x("ref_x", s0, s1, s2, s3, s4, s5, s6, s7);
      RealView8DType xr("xr", s0, s1, s2, s3, s4, s5, s6, s7),
          inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4, s5, s6, s7),
          ref_xr("ref_xr", s0, s1, s2, s3, s4, s5, s6, s7);
      ComplexView8DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5, sr6,
                               sr7);

      const Kokkos::complex<T> z(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
      Kokkos::fill_random(exec, x, random_pool, z);
      Kokkos::fill_random(exec, xr, random_pool, 1);

      KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
      KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft(exec, x, x_hat, KokkosFFT::Normalization::backward, axis,
                     n_new);
      KokkosFFT::ifft(exec, x_hat, inv_x_hat,
                      KokkosFFT::Normalization::backward, axis, n_new);
      EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                      axis, n_new);
      KokkosFFT::irfft(exec, xr_hat, inv_xr_hat,
                       KokkosFFT::Normalization::backward, axis, n_new);

      EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
      exec.fence();
    }
  }
}

// Tests for FFT2
template <typename T, typename LayoutType>
void test_fft2_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1);
  const ComplexView2DType out1("out1", n0, n1), out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);

  // np.fft2 is identical to np.fft(np.fft(x, axis=1), axis=0)
  KokkosFFT::fft(exec, x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/0);

  KokkosFFT::fft2(exec, x,
                  out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fft2(exec, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::fft2(exec, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::fft2(exec, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Plan fft2_plan(exec, x, out, KokkosFFT::Direction::forward, axes);

  KokkosFFT::execute(fft2_plan, x, out);
  KokkosFFT::execute(fft2_plan, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::execute(fft2_plan, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::execute(fft2_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // np.fft2(axes=(-1, -2)) is identical to np.fft(np.fft(x, axis=0), axis=1)
  axes_type axes10 = {-1, -2};

  KokkosFFT::fft(exec, x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/0);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/1);

  KokkosFFT::fft2(exec, x, out_b, KokkosFFT::Normalization::backward, axes10);
  KokkosFFT::fft2(exec, x, out_o, KokkosFFT::Normalization::ortho, axes10);
  KokkosFFT::fft2(exec, x, out_f, KokkosFFT::Normalization::forward, axes10);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans np.fft2(axes=(-1, -2))
  KokkosFFT::Plan fft2_plan_axes10(exec, x, out, KokkosFFT::Direction::forward,
                                   axes10);

  KokkosFFT::execute(fft2_plan_axes10, x, out_b,
                     KokkosFFT::Normalization::backward);
  KokkosFFT::execute(fft2_plan_axes10, x, out_o,
                     KokkosFFT::Normalization::ortho);
  KokkosFFT::execute(fft2_plan_axes10, x, out_f,
                     KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft2_2difft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1),
      out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);

  // np.ifft2 is identical to np.ifft(np.ifft(x, axis=1), axis=0)
  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::ifft(exec, out1, out2, KokkosFFT::Normalization::backward,
                  /*axis=*/0);

  KokkosFFT::ifft2(exec, x,
                   out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifft2(exec, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::ifft2(exec, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::ifft2(exec, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Plan ifft2_plan(exec, x, out, KokkosFFT::Direction::backward,
                             axes);

  KokkosFFT::execute(ifft2_plan, x, out);
  KokkosFFT::execute(ifft2_plan, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::execute(ifft2_plan, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::execute(ifft2_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // np.ifft2(axes=(-1, -2)) is identical to np.ifft(np.ifft(x, axis=0), axis=1)
  axes_type axes10 = {-1, -2};
  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  KokkosFFT::ifft(exec, out1, out2, KokkosFFT::Normalization::backward,
                  /*axis=*/1);

  KokkosFFT::ifft2(exec, x, out_b, KokkosFFT::Normalization::backward, axes10);
  KokkosFFT::ifft2(exec, x, out_o, KokkosFFT::Normalization::ortho, axes10);
  KokkosFFT::ifft2(exec, x, out_f, KokkosFFT::Normalization::forward, axes10);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  KokkosFFT::Plan ifft2_plan_axes10(exec, x, out,
                                    KokkosFFT::Direction::backward, axes10);

  KokkosFFT::execute(ifft2_plan_axes10, x, out_b,
                     KokkosFFT::Normalization::backward);
  KokkosFFT::execute(ifft2_plan_axes10, x, out_o,
                     KokkosFFT::Normalization::ortho);
  KokkosFFT::execute(ifft2_plan_axes10, x, out_f,
                     KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft2_2drfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType x("x", n0, n1), x_ref("x_ref", n0, n1);
  ComplexView2DType out("out", n0, n1 / 2 + 1), out1("out1", n0, n1 / 2 + 1),
      out2("out2", n0, n1 / 2 + 1);
  ComplexView2DType out_b("out_b", n0, n1 / 2 + 1),
      out_o("out_o", n0, n1 / 2 + 1), out_f("out_f", n0, n1 / 2 + 1);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, 1);
  exec.fence();

  Kokkos::deep_copy(x_ref, x);

  // np.rfft2 is identical to np.fft(np.rfft(x, axis=1), axis=0)
  KokkosFFT::rfft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(exec, x,
                   out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(exec, x, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(exec, x, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(exec, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Plan rfft2_plan(exec, x, out, KokkosFFT::Direction::forward, axes);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfft2_plan, x, out);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfft2_plan, x, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfft2_plan, x, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfft2_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft2_2dirfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1 / 2 + 1), x_ref("x_ref", n0, n1 / 2 + 1);
  ComplexView2DType out1("out1", n0, n1 / 2 + 1);
  RealView2DType out2("out2", n0, n1), out("out", n0, n1);
  RealView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  Kokkos::deep_copy(x_ref, x);

  // np.irfft2 is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward, 0);
  KokkosFFT::irfft(exec, out1, out2, KokkosFFT::Normalization::backward, 1);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(exec, x,
                    out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(exec, x, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(exec, x, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(exec, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Plan irfft2_plan(exec, x, out, KokkosFFT::Direction::backward,
                              axes);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfft2_plan, x, out);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfft2_plan, x, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfft2_plan, x, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfft2_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_2dview_shape(T atol = 1.0e-12) {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType xr("xr", n0, n1), xr_ref("xr_ref", n0, n1);
  ComplexView2DType x("x", n0, n1 / 2 + 1), x_ref("x_ref", n0, n1 / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, xr, random_pool, 1.0);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  Kokkos::deep_copy(xr_ref, xr);
  Kokkos::deep_copy(x_ref, x);

  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  std::vector<std::size_t> shapes0 = {n0 / 2, n0, n0 * 2};
  std::vector<std::size_t> shapes1 = {n1 / 2, n1, n1 * 2};

  for (auto&& shape0 : shapes0) {
    for (auto&& shape1 : shapes1) {
      // Real to complex
      ComplexView2DType outr("outr", shape0, shape1 / 2 + 1),
          outr_b("outr_b", shape0, shape1 / 2 + 1),
          outr_o("outr_o", shape0, shape1 / 2 + 1),
          outr_f("outr_f", shape0, shape1 / 2 + 1);

      shape_type<2> new_shape = {shape0, shape1};

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfft2(exec, xr, outr, KokkosFFT::Normalization::none, axes,
                       new_shape);

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfft2(exec, xr, outr_b, KokkosFFT::Normalization::backward,
                       axes, new_shape);

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfft2(exec, xr, outr_o, KokkosFFT::Normalization::ortho, axes,
                       new_shape);

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfft2(exec, xr, outr_f, KokkosFFT::Normalization::forward,
                       axes, new_shape);

      multiply(exec, outr_o, Kokkos::sqrt(static_cast<T>(shape0 * shape1)));
      multiply(exec, outr_f, static_cast<T>(shape0 * shape1));

      EXPECT_TRUE(allclose(exec, outr_b, outr, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, outr_o, outr, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, outr_f, outr, 1.e-5, atol));

      // Complex to real
      RealView2DType out("out", shape0, shape1), out_b("out_b", shape0, shape1),
          out_o("out_o", shape0, shape1), out_f("out_f", shape0, shape1);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfft2(exec, x, out, KokkosFFT::Normalization::none, axes,
                        new_shape);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfft2(exec, x, out_b, KokkosFFT::Normalization::backward,
                        axes, new_shape);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfft2(exec, x, out_o, KokkosFFT::Normalization::ortho, axes,
                        new_shape);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfft2(exec, x, out_f, KokkosFFT::Normalization::forward, axes,
                        new_shape);

      multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(shape0 * shape1)));
      multiply(exec, out_b, static_cast<T>(shape0 * shape1));

      EXPECT_TRUE(allclose(exec, out_b, out, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, out_o, out, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, out_f, out, 1.e-5, atol));
      exec.fence();
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_2dview_inplace([[maybe_unused]] T atol = 1.0e-12) {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1), x_ref("x_ref", n0, n1);
  ComplexView2DType x_hat(x.data(), n0, n1), inv_x_hat(x.data(), n0, n1);

  // Used for real transforms
  ComplexView2DType xr_hat("xr_hat", n0, n1 / 2 + 1),
      xr_hat_ref("xr_hat_ref", n0, n1 / 2 + 1);
  RealView2DType xr_ref("xr_ref", n0, n1),
      inv_xr_hat_unpadded("inv_xr_hat_unpadded", n0, n1),
      inv_xr_hat_ref("inv_xr_hat_ref", n0, n1);

  // Unmanaged views for in-place transforms
  RealView2DType xr(reinterpret_cast<T*>(xr_hat.data()), n0, n1),
      inv_xr_hat(reinterpret_cast<T*>(xr_hat.data()), n0, n1);
  RealView2DType xr_padded(reinterpret_cast<T*>(xr_hat.data()), n0,
                           (n1 / 2 + 1) * 2),
      inv_xr_hat_padded(reinterpret_cast<T*>(xr_hat.data()), n0,
                        (n1 / 2 + 1) * 2);

  // Initialize xr_hat through xr_padded
  auto sub_xr_padded =
      Kokkos::subview(xr_padded, Kokkos::ALL, Kokkos::pair<int, int>(0, n1));

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, xr_ref, random_pool, 1.0);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  Kokkos::deep_copy(sub_xr_padded, xr_ref);
  Kokkos::deep_copy(x_ref, x);

  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    // in-place transforms are not supported if transpose is needed
    EXPECT_THROW(KokkosFFT::fft2(exec, x, x_hat,
                                 KokkosFFT::Normalization::backward, axes),
                 std::runtime_error);
    EXPECT_THROW(KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                                  KokkosFFT::Normalization::backward, axes),
                 std::runtime_error);
    EXPECT_THROW(KokkosFFT::rfft2(exec, xr, xr_hat,
                                  KokkosFFT::Normalization::backward, axes),
                 std::runtime_error);
    EXPECT_THROW(KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                                   KokkosFFT::Normalization::backward, axes),
                 std::runtime_error);
  } else {
    // Identity tests for complex transforms
    KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward, axes);
    KokkosFFT::ifft2(exec, x_hat, inv_x_hat, KokkosFFT::Normalization::backward,
                     axes);

    Kokkos::fence();
    EXPECT_TRUE(allclose(exec, inv_x_hat, x_ref, 1.e-5, atol));

    // In-place transforms
    KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                     axes);

    // Out-of-place transforms (reference)
    KokkosFFT::rfft2(exec, xr_ref, xr_hat_ref,
                     KokkosFFT::Normalization::backward, axes);

    Kokkos::fence();
    EXPECT_TRUE(allclose(exec, xr_hat, xr_hat_ref, 1.e-5, atol));

    // In-place transforms
    Kokkos::fill_random(exec, xr_hat, random_pool, z);
    Kokkos::deep_copy(xr_hat_ref, xr_hat);
    KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                      KokkosFFT::Normalization::backward, axes);

    // Out-of-place transforms (reference)
    KokkosFFT::irfft2(exec, xr_hat_ref, inv_xr_hat_ref,
                      KokkosFFT::Normalization::backward, axes);

    Kokkos::fence();
    auto sub_inv_xr_hat_padded = Kokkos::subview(inv_xr_hat_padded, Kokkos::ALL,
                                                 Kokkos::pair<int, int>(0, n1));
    Kokkos::deep_copy(inv_xr_hat_unpadded, sub_inv_xr_hat_padded);

    EXPECT_TRUE(
        allclose(exec, inv_xr_hat_unpadded, inv_xr_hat_ref, 1.e-5, atol));
  }
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_3dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  constexpr int DIM = 3;
  shape_type<DIM> default_shape({n0, n1, n2});
  ComplexView3DType x("x", n0, n1, n2), ref_x("ref_x", n0, n1, n2);

  execution_space exec;
  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<DIM> shape     = default_shape;
          shape_type<DIM> shape_c2r = default_shape;

          std::size_t n0_new = static_cast<std::size_t>(shape.at(axis0) + i0);
          std::size_t n1_new = static_cast<std::size_t>(shape.at(axis1) + i1);
          shape_type<2> new_shape = {n0_new, n1_new};

          shape.at(axis0)     = n0_new;
          shape.at(axis1)     = n1_new;
          shape_c2r.at(axis0) = n0_new;
          shape_c2r.at(axis1) = n1_new / 2 + 1;

          auto [s0, s1, s2]    = shape;
          auto [sr0, sr1, sr2] = shape_c2r;

          ComplexView3DType inv_x_hat("inv_x_hat", s0, s1, s2),
              x_hat("x_hat", s0, s1, s2), ref_x("ref_x", s0, s1, s2);
          RealView3DType xr("xr", s0, s1, s2),
              inv_xr_hat("inv_xr_hat", s0, s1, s2),
              ref_xr("ref_xr", s0, s1, s2);
          ComplexView3DType xr_hat("xr_hat", sr0, sr1, sr2);

          const Kokkos::complex<T> z(1.0, 1.0);
          Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
          Kokkos::fill_random(exec, x, random_pool, z);
          Kokkos::fill_random(exec, xr, random_pool, 1);

          KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
          KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

          // Along one axis
          // Simple identity tests
          KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward,
                          axes, new_shape);

          KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                           KokkosFFT::Normalization::backward, axes, new_shape);

          EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

          // Simple identity tests for r2c and c2r transforms
          KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                           axes, new_shape);

          KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                            KokkosFFT::Normalization::backward, axes,
                            new_shape);

          EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
          exec.fence();
        }
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_4dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5;
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;

  constexpr int DIM = 4;
  shape_type<DIM> default_shape({n0, n1, n2, n3});
  ComplexView4DType x("x", n0, n1, n2, n3);

  execution_space exec;
  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<DIM> shape     = default_shape;
          shape_type<DIM> shape_c2r = default_shape;

          std::size_t n0_new = static_cast<std::size_t>(shape.at(axis0) + i0);
          std::size_t n1_new = static_cast<std::size_t>(shape.at(axis1) + i1);
          shape_type<2> new_shape = {n0_new, n1_new};

          shape.at(axis0)     = n0_new;
          shape.at(axis1)     = n1_new;
          shape_c2r.at(axis0) = n0_new;
          shape_c2r.at(axis1) = n1_new / 2 + 1;

          auto [s0, s1, s2, s3]     = shape;
          auto [sr0, sr1, sr2, sr3] = shape_c2r;

          ComplexView4DType inv_x_hat("inv_x_hat", s0, s1, s2, s3),
              x_hat("x_hat", s0, s1, s2, s3), ref_x("ref_x", s0, s1, s2, s3);
          RealView4DType xr("xr", s0, s1, s2, s3),
              inv_xr_hat("inv_xr_hat", s0, s1, s2, s3),
              ref_xr("ref_xr", s0, s1, s2, s3);
          ComplexView4DType xr_hat("xr_hat", sr0, sr1, sr2, sr3);

          const Kokkos::complex<T> z(1.0, 1.0);
          Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
          Kokkos::fill_random(exec, x, random_pool, z);
          Kokkos::fill_random(exec, xr, random_pool, 1);

          KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
          KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

          // Along one axis
          // Simple identity tests
          KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward,
                          axes, new_shape);

          KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                           KokkosFFT::Normalization::backward, axes, new_shape);

          EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

          // Simple identity tests for r2c and c2r transforms
          KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                           axes, new_shape);

          KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                            KokkosFFT::Normalization::backward, axes,
                            new_shape);

          EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
          exec.fence();
        }
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_5dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4;
  using RealView5DType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ComplexView5DType =
      Kokkos::View<Kokkos::complex<T>*****, LayoutType, execution_space>;

  constexpr int DIM = 5;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4});
  ComplexView5DType x("x", n0, n1, n2, n3, n4);

  execution_space exec;
  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<DIM> shape     = default_shape;
          shape_type<DIM> shape_c2r = default_shape;

          std::size_t n0_new = static_cast<std::size_t>(shape.at(axis0) + i0);
          std::size_t n1_new = static_cast<std::size_t>(shape.at(axis1) + i1);
          shape_type<2> new_shape = {n0_new, n1_new};

          shape.at(axis0)     = n0_new;
          shape.at(axis1)     = n1_new;
          shape_c2r.at(axis0) = n0_new;
          shape_c2r.at(axis1) = n1_new / 2 + 1;

          auto [s0, s1, s2, s3, s4]      = shape;
          auto [sr0, sr1, sr2, sr3, sr4] = shape_c2r;

          ComplexView5DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4),
              x_hat("x_hat", s0, s1, s2, s3, s4),
              ref_x("ref_x", s0, s1, s2, s3, s4);
          RealView5DType xr("xr", s0, s1, s2, s3, s4),
              inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4),
              ref_xr("ref_xr", s0, s1, s2, s3, s4);
          ComplexView5DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4);

          const Kokkos::complex<T> z(1.0, 1.0);
          Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
          Kokkos::fill_random(exec, x, random_pool, z);
          Kokkos::fill_random(exec, xr, random_pool, 1);

          KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
          KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

          // Along one axis
          // Simple identity tests
          KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward,
                          axes, new_shape);

          KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                           KokkosFFT::Normalization::backward, axes, new_shape);

          EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

          // Simple identity tests for r2c and c2r transforms
          KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                           axes, new_shape);

          KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                            KokkosFFT::Normalization::backward, axes,
                            new_shape);

          EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
          exec.fence();
        }
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_6dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3;
  using RealView6DType = Kokkos::View<T******, LayoutType, execution_space>;
  using ComplexView6DType =
      Kokkos::View<Kokkos::complex<T>******, LayoutType, execution_space>;

  constexpr int DIM = 6;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4, n5});
  ComplexView6DType x("x", n0, n1, n2, n3, n4, n5);

  execution_space exec;
  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<DIM> shape     = default_shape;
          shape_type<DIM> shape_c2r = default_shape;

          std::size_t n0_new = static_cast<std::size_t>(shape.at(axis0) + i0);
          std::size_t n1_new = static_cast<std::size_t>(shape.at(axis1) + i1);
          shape_type<2> new_shape = {n0_new, n1_new};

          shape.at(axis0)     = n0_new;
          shape.at(axis1)     = n1_new;
          shape_c2r.at(axis0) = n0_new;
          shape_c2r.at(axis1) = n1_new / 2 + 1;

          auto [s0, s1, s2, s3, s4, s5]       = shape;
          auto [sr0, sr1, sr2, sr3, sr4, sr5] = shape_c2r;

          ComplexView6DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4, s5),
              x_hat("x_hat", s0, s1, s2, s3, s4, s5),
              ref_x("ref_x", s0, s1, s2, s3, s4, s5);
          RealView6DType xr("xr", s0, s1, s2, s3, s4, s5),
              inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4, s5),
              ref_xr("ref_xr", s0, s1, s2, s3, s4, s5);
          ComplexView6DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5);

          const Kokkos::complex<T> z(1.0, 1.0);
          Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
          Kokkos::fill_random(exec, x, random_pool, z);
          Kokkos::fill_random(exec, xr, random_pool, 1);

          KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
          KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

          // Along one axis
          // Simple identity tests
          KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward,
                          axes, new_shape);

          KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                           KokkosFFT::Normalization::backward, axes, new_shape);

          EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

          // Simple identity tests for r2c and c2r transforms
          KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                           axes, new_shape);

          KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                            KokkosFFT::Normalization::backward, axes,
                            new_shape);

          EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
          exec.fence();
        }
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_7dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4;
  using RealView7DType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ComplexView7DType =
      Kokkos::View<Kokkos::complex<T>*******, LayoutType, execution_space>;

  constexpr int DIM = 7;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4, n5, n6});
  ComplexView7DType x("x", n0, n1, n2, n3, n4, n5, n6);

  execution_space exec;
  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<DIM> shape     = default_shape;
          shape_type<DIM> shape_c2r = default_shape;

          std::size_t n0_new = static_cast<std::size_t>(shape.at(axis0) + i0);
          std::size_t n1_new = static_cast<std::size_t>(shape.at(axis1) + i1);
          shape_type<2> new_shape = {n0_new, n1_new};

          shape.at(axis0)     = n0_new;
          shape.at(axis1)     = n1_new;
          shape_c2r.at(axis0) = n0_new;
          shape_c2r.at(axis1) = n1_new / 2 + 1;

          auto [s0, s1, s2, s3, s4, s5, s6]        = shape;
          auto [sr0, sr1, sr2, sr3, sr4, sr5, sr6] = shape_c2r;

          ComplexView7DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4, s5, s6),
              x_hat("x_hat", s0, s1, s2, s3, s4, s5, s6),
              ref_x("ref_x", s0, s1, s2, s3, s4, s5, s6);

          RealView7DType xr("xr", s0, s1, s2, s3, s4, s5, s6),
              inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4, s5, s6),
              ref_xr("ref_xr", s0, s1, s2, s3, s4, s5, s6);

          ComplexView7DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5, sr6);

          const Kokkos::complex<T> z(1.0, 1.0);
          Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
          Kokkos::fill_random(exec, x, random_pool, z);
          Kokkos::fill_random(exec, xr, random_pool, 1);

          KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
          KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

          // Along one axis
          // Simple identity tests
          KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward,
                          axes, new_shape);

          KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                           KokkosFFT::Normalization::backward, axes, new_shape);

          EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

          // Simple identity tests for r2c and c2r transforms
          KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                           axes, new_shape);

          KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                            KokkosFFT::Normalization::backward, axes,
                            new_shape);

          EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
          exec.fence();
        }
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_8dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4, n7 = 3;
  using RealView8DType = Kokkos::View<T********, LayoutType, execution_space>;
  using ComplexView8DType =
      Kokkos::View<Kokkos::complex<T>********, LayoutType, execution_space>;

  constexpr int DIM = 8;
  shape_type<DIM> default_shape({n0, n1, n2, n3, n4, n5, n6, n7});
  ComplexView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  execution_space exec;
  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<DIM> shape     = default_shape;
          shape_type<DIM> shape_c2r = default_shape;

          std::size_t n0_new = static_cast<std::size_t>(shape.at(axis0) + i0);
          std::size_t n1_new = static_cast<std::size_t>(shape.at(axis1) + i1);
          shape_type<2> new_shape = {n0_new, n1_new};

          shape.at(axis0)     = n0_new;
          shape.at(axis1)     = n1_new;
          shape_c2r.at(axis0) = n0_new;
          shape_c2r.at(axis1) = n1_new / 2 + 1;

          auto [s0, s1, s2, s3, s4, s5, s6, s7]         = shape;
          auto [sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7] = shape_c2r;

          ComplexView8DType inv_x_hat("inv_x_hat", s0, s1, s2, s3, s4, s5, s6,
                                      s7),
              x_hat("x_hat", s0, s1, s2, s3, s4, s5, s6, s7),
              ref_x("ref_x", s0, s1, s2, s3, s4, s5, s6, s7);

          RealView8DType xr("xr", s0, s1, s2, s3, s4, s5, s6, s7),
              inv_xr_hat("inv_xr_hat", s0, s1, s2, s3, s4, s5, s6, s7),
              ref_xr("ref_xr", s0, s1, s2, s3, s4, s5, s6, s7);

          ComplexView8DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5, sr6,
                                   sr7);

          const Kokkos::complex<T> z(1.0, 1.0);
          Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
          Kokkos::fill_random(exec, x, random_pool, z);
          Kokkos::fill_random(exec, xr, random_pool, 1);

          KokkosFFT::Impl::crop_or_pad(exec, x, ref_x);
          KokkosFFT::Impl::crop_or_pad(exec, xr, ref_xr);

          // Along one axis
          // Simple identity tests
          KokkosFFT::fft2(exec, x, x_hat, KokkosFFT::Normalization::backward,
                          axes, new_shape);

          KokkosFFT::ifft2(exec, x_hat, inv_x_hat,
                           KokkosFFT::Normalization::backward, axes, new_shape);

          EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

          // Simple identity tests for r2c and c2r transforms
          KokkosFFT::rfft2(exec, xr, xr_hat, KokkosFFT::Normalization::backward,
                           axes, new_shape);

          KokkosFFT::irfft2(exec, xr_hat, inv_xr_hat,
                            KokkosFFT::Normalization::backward, axes,
                            new_shape);

          EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
          exec.fence();
        }
      }
    }
  }
}

// Tests for FFTN
template <typename T, typename LayoutType>
void test_fftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1),
      out2("out2", n0, n1), out_no_axes("out_no_axes", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);

  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  KokkosFFT::fft(exec, x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/0);

  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::fftn(exec, x,
                  out_no_axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(exec, x, out,
                  axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Plan fftn_plan(exec, x, out, KokkosFFT::Direction::forward, axes);

  KokkosFFT::execute(fftn_plan, x, out);
  KokkosFFT::execute(fftn_plan, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::execute(fftn_plan, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::execute(fftn_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_ifftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1),
      out2("out2", n0, n1), out_no_axes("out_no_axes", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  // np.ifftn for 2D array is identical to np.ifft(np.ifft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::ifft(exec, out1, out2, KokkosFFT::Normalization::backward,
                  /*axis=*/0);

  KokkosFFT::ifftn(exec, x,
                   out_no_axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(exec, x, out,
                   axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Plan ifftn_plan(exec, x, out, KokkosFFT::Direction::backward,
                             axes);

  KokkosFFT::execute(ifftn_plan, x, out);
  KokkosFFT::execute(ifftn_plan, x, out_b, KokkosFFT::Normalization::backward);
  KokkosFFT::execute(ifftn_plan, x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::execute(ifftn_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_rfftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType x("x", n0, n1), x_ref("x_ref", n0, n1);
  ComplexView2DType out("out", n0, n1 / 2 + 1), out1("out1", n0, n1 / 2 + 1),
      out2("out2", n0, n1 / 2 + 1), out_no_axes("out_no_axes", n0, n1 / 2 + 1);
  ComplexView2DType out_b("out_b", n0, n1 / 2 + 1),
      out_o("out_o", n0, n1 / 2 + 1), out_f("out_f", n0, n1 / 2 + 1);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, 1);
  exec.fence();
  Kokkos::deep_copy(x_ref, x);

  // np.rfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::rfft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x,
                   out_no_axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out,
                   axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Plan rfftn_plan(exec, x, out, KokkosFFT::Direction::forward, axes);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfftn_plan, x, out);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfftn_plan, x, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfftn_plan, x, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(rfftn_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_irfftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1 / 2 + 1), x_ref("x_ref", n0, n1 / 2 + 1);
  ComplexView2DType out1("out1", n0, n1 / 2 + 1);
  RealView2DType out2("out2", n0, n1), out("out", n0, n1);
  RealView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1), out_no_axes("out_no_axes", n0, n1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 2D array is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  KokkosFFT::irfft(exec, out1, out2, KokkosFFT::Normalization::backward,
                   /*axis=*/1);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(
      exec, x,
      out_no_axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out,
                    axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Plan irfftn_plan(exec, x, out, KokkosFFT::Direction::backward,
                              axes);
  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfftn_plan, x, out);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfftn_plan, x, out_b, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfftn_plan, x, out_o, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::execute(irfftn_plan, x, out_f, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(exec, out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out2, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fftn_2dfft_2dview_shape(T atol = 1.0e-12) {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType xr("xr", n0, n1), xr_ref("xr_ref", n0, n1);
  ComplexView2DType x("x", n0, n1 / 2 + 1), x_ref("x_ref", n0, n1 / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, xr, random_pool, 1.0);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  Kokkos::deep_copy(xr_ref, xr);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 2D array is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  std::vector<std::size_t> shapes0 = {n0 / 2, n0, n0 * 2};
  std::vector<std::size_t> shapes1 = {n1 / 2, n1, n1 * 2};

  for (auto&& shape0 : shapes0) {
    for (auto&& shape1 : shapes1) {
      // Real to complex
      ComplexView2DType outr("outr", shape0, shape1 / 2 + 1),
          outr_b("outr_b", shape0, shape1 / 2 + 1),
          outr_o("outr_o", shape0, shape1 / 2 + 1),
          outr_f("outr_f", shape0, shape1 / 2 + 1);

      shape_type<2> new_shape = {shape0, shape1};

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfftn(exec, xr, outr, axes, KokkosFFT::Normalization::none,
                       new_shape);

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfftn(exec, xr, outr_b, axes,
                       KokkosFFT::Normalization::backward, new_shape);

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfftn(exec, xr, outr_o, axes, KokkosFFT::Normalization::ortho,
                       new_shape);

      Kokkos::deep_copy(xr, xr_ref);
      KokkosFFT::rfftn(exec, xr, outr_f, axes,
                       KokkosFFT::Normalization::forward, new_shape);

      multiply(exec, outr_o, Kokkos::sqrt(static_cast<T>(shape0 * shape1)));
      multiply(exec, outr_f, static_cast<T>(shape0 * shape1));

      EXPECT_TRUE(allclose(exec, outr_b, outr, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, outr_o, outr, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, outr_f, outr, 1.e-5, atol));

      // Complex to real
      RealView2DType out("out", shape0, shape1), out_b("out_b", shape0, shape1),
          out_o("out_o", shape0, shape1), out_f("out_f", shape0, shape1);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfftn(exec, x, out, axes, KokkosFFT::Normalization::none,
                        new_shape);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfftn(exec, x, out_b, axes,
                        KokkosFFT::Normalization::backward, new_shape);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho,
                        new_shape);

      Kokkos::deep_copy(x, x_ref);
      KokkosFFT::irfftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward,
                        new_shape);

      multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(shape0 * shape1)));
      multiply(exec, out_b, static_cast<T>(shape0 * shape1));

      EXPECT_TRUE(allclose(exec, out_b, out, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, out_o, out, 1.e-5, atol));
      EXPECT_TRUE(allclose(exec, out_f, out, 1.e-5, atol));
      exec.fence();
    }
  }
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_3dview(T atol = 1.0e-6) {
  const int n0 = 4, n1 = 6, n2 = 8;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2);
  ComplexView3DType out("out", n0, n1, n2), out1("out1", n0, n1, n2),
      out2("out2", n0, n1, n2), out3("out3", n0, n1, n2);
  ComplexView3DType out_b("out_b", n0, n1, n2), out_o("out_o", n0, n1, n2),
      out_f("out_f", n0, n1, n2), out_no_axes("out_no_axes", n0, n1, n2);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  // np.fftn for 3D array is identical to np.fft(np.fft(np.fft(x, axis=2),
  // axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;
  axes_type axes  = {-3, -2, -1};

  KokkosFFT::fft(exec, x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/2);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  KokkosFFT::fft(exec, out2, out3, KokkosFFT::Normalization::backward,
                 /*axis=*/0);

  KokkosFFT::fftn(exec, x,
                  out_no_axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(exec, x, out,
                  axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(exec, out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(exec, out, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(exec, out_no_axes, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(exec, out_b, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(exec, out_o, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(exec, out_f, out3, 1.e-5, atol));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_ifftn_3dfft_3dview() {
  const int n0 = 4, n1 = 6, n2 = 8;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2);
  ComplexView3DType out("out", n0, n1, n2), out1("out1", n0, n1, n2),
      out2("out2", n0, n1, n2), out3("out3", n0, n1, n2);
  ComplexView3DType out_b("out_b", n0, n1, n2), out_o("out_o", n0, n1, n2),
      out_f("out_f", n0, n1, n2), out_no_axes("out_no_axes", n0, n1, n2);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  // np.ifftn for 3D array is identical to np.ifft(np.ifft(np.ifft(x, axis=2),
  // axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;
  axes_type axes  = {-3, -2, -1};

  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/2);
  KokkosFFT::ifft(exec, out1, out2, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::ifft(exec, out2, out3, KokkosFFT::Normalization::backward,
                  /*axis=*/0);

  KokkosFFT::ifftn(exec, x,
                   out_no_axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(exec, x, out,
                   axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(exec, out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out3, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_rfftn_3dfft_3dview() {
  const int n0 = 4, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2), x_ref("x_ref", n0, n1, n2);
  ComplexView3DType out("out", n0, n1, n2 / 2 + 1),
      out1("out1", n0, n1, n2 / 2 + 1), out2("out2", n0, n1, n2 / 2 + 1),
      out3("out3", n0, n1, n2 / 2 + 1);
  ComplexView3DType out_b("out_b", n0, n1, n2 / 2 + 1),
      out_o("out_o", n0, n1, n2 / 2 + 1), out_f("out_f", n0, n1, n2 / 2 + 1),
      out_no_axes("out_no_axes", n0, n1, n2 / 2 + 1);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, 1);
  exec.fence();

  Kokkos::deep_copy(x_ref, x);

  // np.rfftn for 3D array is identical to np.fft(np.fft(np.rfft(x, axis=2),
  // axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;
  axes_type axes  = {-3, -2, -1};

  KokkosFFT::rfft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/2);
  KokkosFFT::fft(exec, out1, out2, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  KokkosFFT::fft(exec, out2, out3, KokkosFFT::Normalization::backward,
                 /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x,
                   out_no_axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out,
                   axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, Kokkos::sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(exec, out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(exec, out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out3, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_irfftn_3dfft_3dview() {
  const int n0 = 4, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2 / 2 + 1),
      x_ref("x_ref", n0, n1, n2 / 2 + 1);
  ComplexView3DType out1("out1", n0, n1, n2 / 2 + 1),
      out2("out2", n0, n1, n2 / 2 + 1);
  RealView3DType out("out", n0, n1, n2), out3("out3", n0, n1, n2);
  RealView3DType out_b("out_b", n0, n1, n2), out_o("out_o", n0, n1, n2),
      out_f("out_f", n0, n1, n2), out_no_axes("out_no_axes", n0, n1, n2);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, z);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 3D array is identical to np.irfft(np.ifft(np.ifft(x, axis=0),
  // axis=1), axis=2)
  using axes_type = KokkosFFT::axis_type<3>;
  axes_type axes  = {-3, -2, -1};

  KokkosFFT::ifft(exec, x, out1, KokkosFFT::Normalization::backward,
                  /*axis=*/0);
  KokkosFFT::ifft(exec, out1, out2, KokkosFFT::Normalization::backward,
                  /*axis=*/1);
  KokkosFFT::irfft(exec, out2, out3, KokkosFFT::Normalization::backward,
                   /*axis=*/2);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(
      exec, x,
      out_no_axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out,
                    axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out_b, axes, KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(exec, x, out_f, axes, KokkosFFT::Normalization::forward);

  multiply(exec, out_o, 1.0 / Kokkos::sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(exec, out_f, 1.0 / static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(exec, out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_no_axes, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(exec, out_f, out3, 1.e-5, 1.e-6));
  exec.fence();
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_3dview_shape(T atol = 1.0e-12) {
  const int n0 = 4, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType xr("xr", n0, n1, n2), xr_ref("xr_ref", n0, n1, n2);
  ComplexView3DType x("x", n0, n1, n2 / 2 + 1),
      x_ref("x_ref", n0, n1, n2 / 2 + 1);

  execution_space exec;
  const Kokkos::complex<T> z(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, xr, random_pool, 1.0);
  Kokkos::fill_random(exec, x, random_pool, z);
  exec.fence();

  Kokkos::deep_copy(xr_ref, xr);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 3D array is identical to np.irfft(np.ifft(np.ifft(x, axis=0),
  // axis=1), axis=2)
  using axes_type = KokkosFFT::axis_type<3>;
  axes_type axes  = {-3, -2, -1};

  std::vector<std::size_t> shapes0 = {n0 / 2, n0, n0 * 2};
  std::vector<std::size_t> shapes1 = {n1 / 2, n1, n1 * 2};
  std::vector<std::size_t> shapes2 = {n2 / 2, n2, n2 * 2};

  for (auto&& shape0 : shapes0) {
    for (auto&& shape1 : shapes1) {
      for (auto&& shape2 : shapes2) {
        shape_type<3> new_shape = {shape0, shape1, shape2};

        // Real to complex
        ComplexView3DType outr("outr", shape0, shape1, shape2 / 2 + 1),
            outr_b("outr_b", shape0, shape1, shape2 / 2 + 1),
            outr_o("outr_o", shape0, shape1, shape2 / 2 + 1),
            outr_f("outr_f", shape0, shape1, shape2 / 2 + 1);

        Kokkos::deep_copy(xr, xr_ref);
        KokkosFFT::rfftn(exec, xr, outr, axes, KokkosFFT::Normalization::none,
                         new_shape);

        Kokkos::deep_copy(xr, xr_ref);
        KokkosFFT::rfftn(exec, xr, outr_b, axes,
                         KokkosFFT::Normalization::backward, new_shape);

        Kokkos::deep_copy(xr, xr_ref);
        KokkosFFT::rfftn(exec, xr, outr_o, axes,
                         KokkosFFT::Normalization::ortho, new_shape);

        Kokkos::deep_copy(xr, xr_ref);
        KokkosFFT::rfftn(exec, xr, outr_f, axes,
                         KokkosFFT::Normalization::forward, new_shape);

        multiply(exec, outr_o,
                 Kokkos::sqrt(static_cast<T>(shape0 * shape1 * shape2)));
        multiply(exec, outr_f, static_cast<T>(shape0 * shape1 * shape2));

        EXPECT_TRUE(allclose(exec, outr_b, outr, 1.e-5, atol));
        EXPECT_TRUE(allclose(exec, outr_o, outr, 1.e-5, atol));
        EXPECT_TRUE(allclose(exec, outr_f, outr, 1.e-5, atol));

        // Complex to real
        RealView3DType out("out", shape0, shape1, shape2),
            out_b("out_b", shape0, shape1, shape2),
            out_o("out_o", shape0, shape1, shape2),
            out_f("out_f", shape0, shape1, shape2);

        Kokkos::deep_copy(x, x_ref);
        KokkosFFT::irfftn(exec, x, out, axes, KokkosFFT::Normalization::none,
                          new_shape);

        Kokkos::deep_copy(x, x_ref);
        KokkosFFT::irfftn(exec, x, out_b, axes,
                          KokkosFFT::Normalization::backward, new_shape);

        Kokkos::deep_copy(x, x_ref);
        KokkosFFT::irfftn(exec, x, out_o, axes, KokkosFFT::Normalization::ortho,
                          new_shape);

        Kokkos::deep_copy(x, x_ref);
        KokkosFFT::irfftn(exec, x, out_f, axes,
                          KokkosFFT::Normalization::forward, new_shape);

        multiply(exec, out_o,
                 Kokkos::sqrt(static_cast<T>(shape0 * shape1 * shape2)));
        multiply(exec, out_b, static_cast<T>(shape0 * shape1 * shape2));

        EXPECT_TRUE(allclose(exec, out_b, out, 1.e-5, atol));
        EXPECT_TRUE(allclose(exec, out_o, out, 1.e-5, atol));
        EXPECT_TRUE(allclose(exec, out_f, out, 1.e-5, atol));
        exec.fence();
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_4dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5;
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;

  constexpr int DIM          = 4;
  std::array<int, DIM> shape = {n0, n1, n2, n3};
  ComplexView4DType x("x", n0, n1, n2, n3), ref_x("ref_x", n0, n1, n2, n3);

  using axes_type = KokkosFFT::axis_type<3>;

  execution_space exec;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;

        axes_type axes = {axis0, axis1, axis2};

        std::array<int, DIM> shape_c2r = shape;
        shape_c2r.at(axis2)            = shape_c2r.at(axis2) / 2 + 1;

        auto [sr0, sr1, sr2, sr3] = shape_c2r;

        ComplexView4DType inv_x_hat("inv_x_hat", n0, n1, n2, n3),
            x_hat("x_hat", n0, n1, n2, n3);
        RealView4DType xr("xr", n0, n1, n2, n3),
            ref_xr("ref_xr", n0, n1, n2, n3),
            inv_xr_hat("inv_xr_hat", n0, n1, n2, n3);
        ComplexView4DType xr_hat("xr_hat", sr0, sr1, sr2, sr3);

        const Kokkos::complex<T> z(1.0, 1.0);
        Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
        Kokkos::fill_random(exec, x, random_pool, z);
        Kokkos::fill_random(exec, xr, random_pool, 1);
        exec.fence();

        Kokkos::deep_copy(ref_x, x);
        Kokkos::deep_copy(ref_xr, xr);

        // Along one axis
        // Simple identity tests
        KokkosFFT::fftn(exec, x, x_hat, axes,
                        KokkosFFT::Normalization::backward);

        KokkosFFT::ifftn(exec, x_hat, inv_x_hat, axes,
                         KokkosFFT::Normalization::backward);

        EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

        // Simple identity tests for r2c and c2r transforms
        KokkosFFT::rfftn(exec, xr, xr_hat, axes,
                         KokkosFFT::Normalization::backward);

        KokkosFFT::irfftn(exec, xr_hat, inv_xr_hat, axes,
                          KokkosFFT::Normalization::backward);

        EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
        exec.fence();
      }
    }
  }
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_5dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4;
  using RealView5DType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ComplexView5DType =
      Kokkos::View<Kokkos::complex<T>*****, LayoutType, execution_space>;

  constexpr int DIM          = 5;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4};
  ComplexView5DType x("x", n0, n1, n2, n3, n4),
      ref_x("ref_x", n0, n1, n2, n3, n4);

  using axes_type = KokkosFFT::axis_type<3>;
  KokkosFFT::axis_type<DIM> default_axes({0, 1, 2, 3, 4});

  // Too many combinations, choose axes randomly
  std::vector<axes_type> list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    auto tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);

    // pickup 3 elements only
    axes_type trimmed_axes;
    std::copy(std::begin(tmp_axes) + DIM - 3, std::end(tmp_axes),
              std::begin(trimmed_axes));
    list_of_tested_axes.push_back(trimmed_axes);
  }

  execution_space exec;
  for (auto& tested_axes : list_of_tested_axes) {
    int last_axis                  = tested_axes.at(2);
    std::array<int, DIM> shape_c2r = shape;
    shape_c2r.at(last_axis)        = shape_c2r.at(last_axis) / 2 + 1;

    auto [sr0, sr1, sr2, sr3, sr4] = shape_c2r;
    ComplexView5DType inv_x_hat("inv_x_hat", n0, n1, n2, n3, n4),
        x_hat("x_hat", n0, n1, n2, n3, n4);
    RealView5DType xr("xr", n0, n1, n2, n3, n4),
        ref_xr("ref_xr", n0, n1, n2, n3, n4),
        inv_xr_hat("inv_xr_hat", n0, n1, n2, n3, n4);
    ComplexView5DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
    Kokkos::fill_random(exec, x, random_pool, z);
    Kokkos::fill_random(exec, xr, random_pool, 1);
    exec.fence();

    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    // Along one axis
    // Simple identity tests
    KokkosFFT::fftn(exec, x, x_hat, tested_axes,
                    KokkosFFT::Normalization::backward);

    KokkosFFT::ifftn(exec, x_hat, inv_x_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfftn(exec, xr, xr_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    KokkosFFT::irfftn(exec, xr_hat, inv_xr_hat, tested_axes,
                      KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
    exec.fence();
  }
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_6dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7;
  using RealView6DType = Kokkos::View<T******, LayoutType, execution_space>;
  using ComplexView6DType =
      Kokkos::View<Kokkos::complex<T>******, LayoutType, execution_space>;

  constexpr int DIM          = 6;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5};
  ComplexView6DType x("x", n0, n1, n2, n3, n4, n5),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5);

  using axes_type = KokkosFFT::axis_type<3>;
  KokkosFFT::axis_type<DIM> default_axes({0, 1, 2, 3, 4, 5});

  // Too many combinations, choose axes randomly
  std::vector<axes_type> list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    auto tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);

    // pickup 3 elements only
    axes_type trimmed_axes;
    std::copy(std::begin(tmp_axes) + DIM - 3, std::end(tmp_axes),
              std::begin(trimmed_axes));
    list_of_tested_axes.push_back(trimmed_axes);
  }

  execution_space exec;
  for (auto& tested_axes : list_of_tested_axes) {
    int last_axis                  = tested_axes.at(2);
    std::array<int, DIM> shape_c2r = shape;
    shape_c2r.at(last_axis)        = shape_c2r.at(last_axis) / 2 + 1;

    auto [sr0, sr1, sr2, sr3, sr4, sr5] = shape_c2r;
    ComplexView6DType inv_x_hat("inv_x_hat", n0, n1, n2, n3, n4, n5),
        x_hat("x_hat", n0, n1, n2, n3, n4, n5);
    RealView6DType xr("xr", n0, n1, n2, n3, n4, n5),
        ref_xr("ref_xr", n0, n1, n2, n3, n4, n5),
        inv_xr_hat("inv_xr_hat", n0, n1, n2, n3, n4, n5);
    ComplexView6DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
    Kokkos::fill_random(exec, x, random_pool, z);
    Kokkos::fill_random(exec, xr, random_pool, 1);
    exec.fence();

    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    // Along one axis
    // Simple identity tests
    KokkosFFT::fftn(exec, x, x_hat, tested_axes,
                    KokkosFFT::Normalization::backward);

    KokkosFFT::ifftn(exec, x_hat, inv_x_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfftn(exec, xr, xr_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    KokkosFFT::irfftn(exec, xr_hat, inv_xr_hat, tested_axes,
                      KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
    exec.fence();
  }
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_7dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8;
  using RealView7DType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ComplexView7DType =
      Kokkos::View<Kokkos::complex<T>*******, LayoutType, execution_space>;

  constexpr int DIM          = 7;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5, n6};
  ComplexView7DType x("x", n0, n1, n2, n3, n4, n5, n6),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5, n6);

  using axes_type = KokkosFFT::axis_type<3>;
  KokkosFFT::axis_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6});

  // Too many combinations, choose axes randomly
  std::vector<axes_type> list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    auto tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);

    // pickup 3 elements only
    axes_type trimmed_axes;
    std::copy(std::begin(tmp_axes) + DIM - 3, std::end(tmp_axes),
              std::begin(trimmed_axes));
    list_of_tested_axes.push_back(trimmed_axes);
  }

  execution_space exec;
  for (auto& tested_axes : list_of_tested_axes) {
    int last_axis                  = tested_axes.at(2);
    std::array<int, DIM> shape_c2r = shape;
    shape_c2r.at(last_axis)        = shape_c2r.at(last_axis) / 2 + 1;

    auto [sr0, sr1, sr2, sr3, sr4, sr5, sr6] = shape_c2r;
    ComplexView7DType inv_x_hat("inv_x_hat", n0, n1, n2, n3, n4, n5, n6),
        x_hat("x_hat", n0, n1, n2, n3, n4, n5, n6);
    RealView7DType xr("xr", n0, n1, n2, n3, n4, n5, n6),
        ref_xr("ref_xr", n0, n1, n2, n3, n4, n5, n6),
        inv_xr_hat("inv_xr_hat", n0, n1, n2, n3, n4, n5, n6);
    ComplexView7DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5, sr6);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
    Kokkos::fill_random(exec, x, random_pool, z);
    Kokkos::fill_random(exec, xr, random_pool, 1);
    exec.fence();

    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    // Along one axis
    // Simple identity tests
    KokkosFFT::fftn(exec, x, x_hat, tested_axes,
                    KokkosFFT::Normalization::backward);

    KokkosFFT::ifftn(exec, x_hat, inv_x_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfftn(exec, xr, xr_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    KokkosFFT::irfftn(exec, xr_hat, inv_xr_hat, tested_axes,
                      KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
    exec.fence();
  }
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_8dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 3, n2 = 4, n3 = 5, n4 = 6, n5 = 7, n6 = 8, n7 = 9;
  using RealView8DType = Kokkos::View<T********, LayoutType, execution_space>;
  using ComplexView8DType =
      Kokkos::View<Kokkos::complex<T>********, LayoutType, execution_space>;

  constexpr int DIM          = 8;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5, n6, n7};
  ComplexView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5, n6, n7);

  using axes_type = KokkosFFT::axis_type<3>;
  KokkosFFT::axis_type<DIM> default_axes({0, 1, 2, 3, 4, 5, 6, 7});

  // Too many combinations, choose axes randomly
  std::vector<axes_type> list_of_tested_axes;

  constexpr int nb_trials = 32;
  auto rng                = std::default_random_engine{};

  for (int i = 0; i < nb_trials; i++) {
    auto tmp_axes = default_axes;
    std::shuffle(std::begin(tmp_axes), std::end(tmp_axes), rng);

    // pickup 3 elements only
    axes_type trimmed_axes;
    std::copy(std::begin(tmp_axes) + DIM - 3, std::end(tmp_axes),
              std::begin(trimmed_axes));
    list_of_tested_axes.push_back(trimmed_axes);
  }

  execution_space exec;
  for (auto& tested_axes : list_of_tested_axes) {
    int last_axis                  = tested_axes.at(2);
    std::array<int, DIM> shape_c2r = shape;
    shape_c2r.at(last_axis)        = shape_c2r.at(last_axis) / 2 + 1;

    auto [sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7] = shape_c2r;
    ComplexView8DType inv_x_hat("inv_x_hat", n0, n1, n2, n3, n4, n5, n6, n7),
        x_hat("x_hat", n0, n1, n2, n3, n4, n5, n6, n7);
    RealView8DType xr("xr", n0, n1, n2, n3, n4, n5, n6, n7),
        ref_xr("ref_xr", n0, n1, n2, n3, n4, n5, n6, n7),
        inv_xr_hat("inv_xr_hat", n0, n1, n2, n3, n4, n5, n6, n7);
    ComplexView8DType xr_hat("xr_hat", sr0, sr1, sr2, sr3, sr4, sr5, sr6, sr7);

    const Kokkos::complex<T> z(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
    Kokkos::fill_random(exec, x, random_pool, z);
    Kokkos::fill_random(exec, xr, random_pool, 1);
    exec.fence();

    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    // Along one axis
    // Simple identity tests
    KokkosFFT::fftn(exec, x, x_hat, tested_axes,
                    KokkosFFT::Normalization::backward);

    KokkosFFT::ifftn(exec, x_hat, inv_x_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_x_hat, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfftn(exec, xr, xr_hat, tested_axes,
                     KokkosFFT::Normalization::backward);

    KokkosFFT::irfftn(exec, xr_hat, inv_xr_hat, tested_axes,
                      KokkosFFT::Normalization::backward);

    EXPECT_TRUE(allclose(exec, inv_xr_hat, ref_xr, 1.e-5, atol));
    exec.fence();
  }
}
}  // namespace

TYPED_TEST_SUITE(FFT1D, test_types);
TYPED_TEST_SUITE(FFT2D, test_types);
TYPED_TEST_SUITE(FFTND, test_types);

// Identity tests on 1D Views
TYPED_TEST(FFT1D, Identity_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_identity<float_type, layout_type>(atol);
}

// Identity tests on 1D Views for in-place transform
TYPED_TEST(FFT1D, Identity_1DView_inplace) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_identity_inplace<float_type, layout_type>(atol);
}

// Identity tests on 1D Views with plan reuse
TYPED_TEST(FFT1D, Identity_1DView_reuse_plans) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_identity_reuse_plan<float_type, layout_type>(atol);
}

// fft on 1D Views
TYPED_TEST(FFT1D, FFT_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft1_1dfft_1dview<float_type, layout_type>();
}

// ifft on 1D Views
TYPED_TEST(FFT1D, IFFT_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft1_1difft_1dview<float_type, layout_type>();
}

// hfft on 1D Views
TYPED_TEST(FFT1D, HFFT_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft1_1dhfft_1dview<float_type, layout_type>();
}

// ihfft on 1D Views
TYPED_TEST(FFT1D, IHFFT_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft1_1dihfft_1dview<float_type, layout_type>();
}

// fft1 on 1D Views with shape argument
TYPED_TEST(FFT1D, FFT_1DView_shape) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_shape<float_type, layout_type>(atol);
}

// batched fft1 on 2D Views
TYPED_TEST(FFT1D, FFT_batched_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_2dview<float_type, layout_type>(atol);
}

// batched fft1 on 3D Views
TYPED_TEST(FFT1D, FFT_batched_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_3dview<float_type, layout_type>(atol);
}

// batched fft1 on 4D Views
TYPED_TEST(FFT1D, FFT_batched_4DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_4dview<float_type, layout_type>(atol);
}

// batched fft1 on 5D Views
TYPED_TEST(FFT1D, FFT_batched_5DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_5dview<float_type, layout_type>(atol);
}

// batched fft1 on 6D Views
TYPED_TEST(FFT1D, FFT_batched_6DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_6dview<float_type, layout_type>(atol);
}

// batched fft1 on 7D Views
TYPED_TEST(FFT1D, FFT_batched_7DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_7dview<float_type, layout_type>(atol);
}

// batched fft1 on 8D Views
TYPED_TEST(FFT1D, FFT_batched_8DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_8dview<float_type, layout_type>(atol);
}

// fft2 on 2D Views
TYPED_TEST(FFT2D, FFT2_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft2_2dfft_2dview<float_type, layout_type>();
}

// ifft2 on 2D Views
TYPED_TEST(FFT2D, IFFT2_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft2_2difft_2dview<float_type, layout_type>();
}

// rfft2 on 2D Views
TYPED_TEST(FFT2D, RFFT2_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft2_2drfft_2dview<float_type, layout_type>();
}

// irfft2 on 2D Views
TYPED_TEST(FFT2D, IRFFT2_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft2_2dirfft_2dview<float_type, layout_type>();
}

// fft2 on 2D Views with shape argument
TYPED_TEST(FFT2D, 2DFFT_2DView_shape) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft2_2dfft_2dview_shape<float_type, layout_type>();
}

// fft2 on 2D Views with in-place transform
TYPED_TEST(FFT2D, 2DFFT_2DView_inplace) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fft2_2dfft_2dview_inplace<float_type, layout_type>();
}

// batched fft2 on 3D Views
TYPED_TEST(FFT2D, FFT_batched_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_3dview<float_type, layout_type>(atol);
}

// batched fft2 on 4D Views
TYPED_TEST(FFT2D, FFT_batched_4DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_4dview<float_type, layout_type>(atol);
}

// batched fft2 on 5D Views
TYPED_TEST(FFT2D, FFT_batched_5DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_5dview<float_type, layout_type>(atol);
}

// batched fft2 on 6D Views
TYPED_TEST(FFT2D, FFT_batched_6DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_6dview<float_type, layout_type>(atol);
}

// batched fft2 on 7D Views
TYPED_TEST(FFT2D, FFT_batched_7DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_7dview<float_type, layout_type>(atol);
}

// batched fft2 on 8D Views
TYPED_TEST(FFT2D, FFT_batched_8DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_8dview<float_type, layout_type>(atol);
}

// fftn on 2D Views
TYPED_TEST(FFTND, 2DFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fftn_2dfft_2dview<float_type, layout_type>();
}

// ifftn on 2D Views
TYPED_TEST(FFTND, 2DIFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_ifftn_2dfft_2dview<float_type, layout_type>();
}

// rfftn on 2D Views
TYPED_TEST(FFTND, 2DRFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_rfftn_2dfft_2dview<float_type, layout_type>();
}

// irfftn on 2D Views
TYPED_TEST(FFTND, 2DIRFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_irfftn_2dfft_2dview<float_type, layout_type>();
}

// fftn on 2D Views with shape argument
TYPED_TEST(FFTND, 2DFFT_2DView_shape) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fftn_2dfft_2dview_shape<float_type, layout_type>();
}

// fftn on 3D Views
TYPED_TEST(FFTND, 3DFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 5.0e-5 : 1.0e-10;
  test_fftn_3dfft_3dview<float_type, layout_type>(atol);
}

// ifftn on 3D Views
TYPED_TEST(FFTND, 3DIFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_ifftn_3dfft_3dview<float_type, layout_type>();
}

// rfftn on 3D Views
TYPED_TEST(FFTND, 3DRFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_rfftn_3dfft_3dview<float_type, layout_type>();
}

// irfftn on 3D Views
TYPED_TEST(FFTND, 3DIRFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_irfftn_3dfft_3dview<float_type, layout_type>();
}

// fftn on 3D Views with shape argument
TYPED_TEST(FFTND, 3DFFT_3DView_shape) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fftn_3dfft_3dview_shape<float_type, layout_type>();
}

// batched fftn on 4D Views
TYPED_TEST(FFTND, 3DFFT_batched_4DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-5 : 1.0e-10;
  test_fftn_3dfft_4dview<float_type, layout_type>(atol);
}

// batched fftn on 5D Views
TYPED_TEST(FFTND, 3DFFT_batched_5DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-5 : 1.0e-10;
  test_fftn_3dfft_5dview<float_type, layout_type>(atol);
}

// batched fftn on 6D Views
TYPED_TEST(FFTND, 3DFFT_batched_6DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-5 : 1.0e-10;
  test_fftn_3dfft_6dview<float_type, layout_type>(atol);
}

// batched fftn on 7D Views
TYPED_TEST(FFTND, 3DFFT_batched_7DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-5 : 1.0e-10;
  test_fftn_3dfft_7dview<float_type, layout_type>(atol);
}

// batched fftn on 8D Views
TYPED_TEST(FFTND, 3DFFT_batched_8DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-5 : 1.0e-10;
  test_fftn_3dfft_8dview<float_type, layout_type>(atol);
}
