#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_Transform.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

/// Kokkos equivalent of fft1 with numpy
/// def fft1(x):
///    L = len(x)
///    phase = -2j * np.pi * (np.arange(L) / L)
///    phase = np.arange(L).reshape(-1, 1) * phase
///    return np.sum(x*np.exp(phase), axis=1)
template <typename ViewType>
void fft1(ViewType& in, ViewType& out) {
  using value_type = typename ViewType::non_const_value_type;
  using real_value_type = KokkosFFT::real_type_t<value_type>;

  static_assert(KokkosFFT::is_complex<value_type>::value,
                "fft1: ViewType must be complex");

  const value_type I(0.0, 1.0);
  std::size_t L = in.size();

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<execution_space>(L, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<execution_space>::member_type& team_member) {
      const int j = team_member.league_rank();

      value_type sum = 0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, L),
        [&](const int i, value_type& lsum) {
          auto phase = -2 * I * M_PI *
            static_cast<real_value_type>(i) / static_cast<real_value_type>(L);

          auto tmp_in = in(i);
          lsum += tmp_in * Kokkos::exp( static_cast<real_value_type>(j) * phase );
        },
        sum
      );

      out(j) = sum;
    }
  );
}

/// Kokkos equivalent of ifft1 with numpy
/// def ifft1(x):
///    L = len(x)
///    phase = 2j * np.pi * (np.arange(L) / L)
///    phase = np.arange(L).reshape(-1, 1) * phase
///    return np.sum(x*np.exp(phase), axis=1)
template <typename ViewType>
void ifft1(ViewType& in, ViewType& out) {
  using value_type = typename ViewType::non_const_value_type;
  using real_value_type = KokkosFFT::real_type_t<value_type>;

  static_assert(KokkosFFT::is_complex<value_type>::value,
                "ifft1: ViewType must be complex");

  const value_type I(0.0, 1.0);
  std::size_t L = in.size();

  Kokkos::parallel_for(
    Kokkos::TeamPolicy<execution_space>(L, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<execution_space>::member_type& team_member) {
      const int j = team_member.league_rank();

      value_type sum = 0;
      Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(team_member, L),
        [&](const int i, value_type& lsum) {
          auto phase = 2 * I * M_PI *
            static_cast<real_value_type>(i) / static_cast<real_value_type>(L);

          auto tmp_in = in(i);
          lsum += tmp_in * Kokkos::exp( static_cast<real_value_type>(j) * phase );
        },
        sum
      );

      out(j) = sum;
    }
  );
}

template <typename T, typename LayoutType>
void test_fft1_identity(T atol=1.0e-12) {
  const int maxlen = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType = Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType a("a", maxlen), _a("_a", maxlen), a_ref("a_ref", maxlen);
  ComplexView1DType out("out", maxlen), outr("outr", maxlen/2+1);
  RealView1DType ar("ar", maxlen), _ar("_ar", maxlen), ar_ref("ar_ref", maxlen);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(a,  random_pool, I);
  Kokkos::fill_random(ar, random_pool, 1.0);
  Kokkos::deep_copy(a_ref, a);
  Kokkos::deep_copy(ar_ref, ar);

  Kokkos::fence();

  KokkosFFT::fft(a, out);
  KokkosFFT::ifft(out, _a);

  KokkosFFT::rfft(ar, outr);
  KokkosFFT::irfft(outr, _ar);

  EXPECT_TRUE( allclose(_a, a_ref, 1.e-5, atol) );
  EXPECT_TRUE( allclose(_ar, ar_ref, 1.e-5, atol) );
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_1dview() {
  const int len = 30;
  using ComplexView1DType = Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x("x", len), out("out", len), ref("ref", len);
  ComplexView1DType out_b("out_b", len), out_o("out_o", len), out_f("out_f", len);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  KokkosFFT::fft(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fft(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fft(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fft(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  fft1(x, ref);
  multiply(out_o, sqrt(static_cast<T>(len)));
  multiply(out_f, static_cast<T>(len));

  EXPECT_TRUE( allclose(out,   ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, ref, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_fft1_1difft_1dview() {
  const int len = 30;
  using ComplexView1DType = Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x("x", len), out("out", len), ref("ref", len);
  ComplexView1DType out_b("out_b", len), out_o("out_o", len), out_f("out_f", len);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  KokkosFFT::ifft(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifft(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifft(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifft(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  ifft1(x, ref);
  multiply(out_o, sqrt(static_cast<T>(len)));
  multiply(out_b, static_cast<T>(len));
  multiply(out, static_cast<T>(len));

  EXPECT_TRUE( allclose(out,   ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, ref, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_2dview(T atol=1.e-12) {
  const int n0 = 10, n1 = 12;
  using RealView2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1), ref_x("ref_x", n0, n1);
  ComplexView2DType x_axis0("x_axis0", n0, n1), x_axis1("x_axis1", n0, n1);
  ComplexView2DType out_axis0("out_axis0", n0, n1), ref_out_axis0("ref_out_axis0", n0, n1);
  ComplexView2DType out_axis1("out_axis1", n0, n1), ref_out_axis1("ref_out_axis1", n0, n1);

  RealView2DType xr("xr", n0, n1), ref_xr("ref_xr", n0, n1);
  RealView2DType xr_axis0("xr_axis0", n0, n1), xr_axis1("xr_axis1", n0, n1);
  ComplexView2DType outr_axis0("outr_axis0", n0/2+1, n1), outr_axis1("outr_axis1", n0, n1/2+1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  // Since HIP FFT destructs the input data, we need to keep the input data in different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  Kokkos::fence();

  // Along axis 0 (transpose neeed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for(int i1=0; i1<n1; i1++) {
    auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1);
    auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1);
    fft1(sub_x, sub_ref);
  }

  KokkosFFT::fft(x, out_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(out_axis0, ref_out_axis0, 1.e-5, atol) );

  KokkosFFT::ifft(out_axis0, x_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(x_axis0, ref_x, 1.e-5, atol) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(outr_axis0, xr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  EXPECT_TRUE( allclose(xr_axis0, ref_xr, 1.e-5, atol) );

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1
  // Perform batched 1D (along 1st axis) FFT sequentially
  for(int i0=0; i0<n0; i0++) {
    auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL);
    auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL);
    fft1(sub_x, sub_ref);
  }

  KokkosFFT::fft(x, out_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(out_axis1, ref_out_axis1, 1.e-5, atol) );

  KokkosFFT::ifft(out_axis1, x_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(x_axis1, ref_x, 1.e-5, atol) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::irfft(outr_axis1, xr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);

  EXPECT_TRUE( allclose(xr_axis1, ref_xr, 1.e-5, atol) );
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_3dview(T atol=1.e-12) {
  const int n0 = 10, n1 = 12, n2 = 8;
  using RealView3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2), ref_x("ref_x", n0, n1, n2);
  ComplexView3DType x_axis0("x_axis0", n0, n1, n2), x_axis1("x_axis1", n0, n1, n2), x_axis2("x_axis2", n0, n1, n2);
  ComplexView3DType out_axis0("out_axis0", n0, n1, n2), ref_out_axis0("ref_out_axis0", n0, n1, n2);
  ComplexView3DType out_axis1("out_axis1", n0, n1, n2), ref_out_axis1("ref_out_axis1", n0, n1, n2);
  ComplexView3DType out_axis2("out_axis2", n0, n1, n2), ref_out_axis2("ref_out_axis2", n0, n1, n2);

  RealView3DType xr("xr", n0, n1, n2), ref_xr("ref_xr", n0, n1, n2);
  RealView3DType xr_axis0("xr_axis0", n0, n1, n2), xr_axis1("xr_axis1", n0, n1, n2), xr_axis2("xr_axis2", n0, n1, n2);
  ComplexView3DType outr_axis0("outr_axis0", n0/2+1, n1, n2), outr_axis1("outr_axis1", n0, n1/2+1, n2), outr_axis2("outr_axis2", n0, n1, n2/2+1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  // Since HIP FFT destructs the input data, we need to keep the input data in different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  Kokkos::fence();

  // Along axis 0 (transpose neeed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for(int i2=0; i2<n2; i2++) {
    for(int i1=0; i1<n1; i1++) {
      auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1, i2);
      auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1, i2);
      fft1(sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(x, out_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(out_axis0, ref_out_axis0, 1.e-5, atol) );

  KokkosFFT::ifft(out_axis0, x_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(x_axis0, ref_x, 1.e-5, atol) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(outr_axis0, xr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  EXPECT_TRUE( allclose(xr_axis0, ref_xr, 1.e-5, atol) );

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1 (transpose neeed)
  // Perform batched 1D (along 1st axis) FFT sequentially
  for(int i2=0; i2<n2; i2++) {
    for(int i0=0; i0<n0; i0++) {
      auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL, i2);
      auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL, i2);
      fft1(sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(x, out_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(out_axis1, ref_out_axis1, 1.e-5, atol) );

  KokkosFFT::ifft(out_axis1, x_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(x_axis1, ref_x, 1.e-5, atol) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::irfft(outr_axis1, xr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);

  EXPECT_TRUE( allclose(xr_axis1, ref_xr, 1.e-5, atol) );

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 2
  // Perform batched 1D (along 2nd axis) FFT sequentially
  for(int i1=0; i1<n1; i1++) {
    for(int i0=0; i0<n0; i0++) {
      auto sub_x   = Kokkos::subview(x, i0, i1, Kokkos::ALL);
      auto sub_ref = Kokkos::subview(ref_out_axis2, i0, i1, Kokkos::ALL);
      fft1(sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(x, out_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  EXPECT_TRUE( allclose(out_axis2, ref_out_axis2, 1.e-5, atol) );

  KokkosFFT::ifft(out_axis2, x_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  EXPECT_TRUE( allclose(x_axis2, ref_x, 1.e-5, atol) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  KokkosFFT::irfft(outr_axis2, xr_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);

  EXPECT_TRUE( allclose(xr_axis2, ref_xr, 1.e-5, atol) );
}

// Identity tests of fft on 1D Views
TEST(FFT1D, Identity_1DLeftViewFloat) {
  test_fft1_identity<float, Kokkos::LayoutLeft>(1.0e-6);
}

TEST(FFT1D, Identity_1DLeftViewDouble) {
  test_fft1_identity<double, Kokkos::LayoutLeft>(1.0e-12);
}

TEST(FFT1D, Identity_1DRightViewFloat) {
  test_fft1_identity<float, Kokkos::LayoutRight>(1.0e-6);
}

TEST(FFT1D, Identity_1DRightViewDouble) {
  test_fft1_identity<double, Kokkos::LayoutRight>(1.0e-12);
}

// fft on 1D Views
TEST(FFT1D, 1DFFT_1DLeftViewFloat) {
  test_fft1_1dfft_1dview<float, Kokkos::LayoutLeft>();
}

TEST(FFT1D, 1DFFT_1DLeftViewDouble) {
  test_fft1_1dfft_1dview<double, Kokkos::LayoutLeft>();
}

TEST(FFT1D, 1DFFT_1DRightViewFloat) {
  test_fft1_1dfft_1dview<float, Kokkos::LayoutRight>();
}

TEST(FFT1D, 1DFFT_1DRightViewDouble) {
  test_fft1_1dfft_1dview<double, Kokkos::LayoutRight>();
}

// ifft on 1D Views
TEST(FFT1D, 1DIFFT_1DLeftViewFloat) {
  test_fft1_1difft_1dview<float, Kokkos::LayoutLeft>();
}

TEST(FFT1D, 1DIFFT_1DLeftViewDouble) {
  test_fft1_1difft_1dview<double, Kokkos::LayoutLeft>();
}

TEST(FFT1D, 1DIFFT_1DRightViewFloat) {
  test_fft1_1difft_1dview<float, Kokkos::LayoutRight>();
}

TEST(FFT1D, 1DIFFT_1DRightViewDouble) {
  test_fft1_1difft_1dview<double, Kokkos::LayoutRight>();
}

// batced fft1 on 2D Views
TEST(FFT1D, 1DBatchedFFT_2DLeftViewFloat) {
  test_fft1_1dfft_2dview<float, Kokkos::LayoutLeft>(1.0e-6);
}

TEST(FFT1D, 1DBatchedFFT_2DLeftViewDouble) {
  test_fft1_1dfft_2dview<double, Kokkos::LayoutLeft>(1.0e-12);
}

TEST(FFT1D, 1DBatchedFFT_2DRightViewFloat) {
  test_fft1_1dfft_2dview<float, Kokkos::LayoutRight>(1.0e-6);
}

TEST(FFT1D, 1DBatchedFFT_2DRightViewDouble) {
  test_fft1_1dfft_2dview<double, Kokkos::LayoutRight>(1.0e-12);
}

// batced fft1 on 3D Views
TEST(FFT1D, 1DBatchedFFT_3DLeftViewFloat) {
  test_fft1_1dfft_3dview<float, Kokkos::LayoutLeft>(1.0e-6);
}

TEST(FFT1D, 1DBatchedFFT_3DLeftViewDouble) {
  test_fft1_1dfft_3dview<double, Kokkos::LayoutLeft>(1.0e-12);
}

TEST(FFT1D, 1DBatchedFFT_3DRightViewFloat) {
  test_fft1_1dfft_3dview<float, Kokkos::LayoutRight>(1.0e-6);
}

TEST(FFT1D, 1DBatchedFFT_3DRightViewDouble) {
  test_fft1_1dfft_3dview<double, Kokkos::LayoutRight>(1.0e-12);
}

// Tests for FFT2
template <typename T, typename LayoutType>
void test_fft2_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.fft2 is identical to np.fft(np.fft(x, axis=1), axis=0)
  KokkosFFT::fft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  KokkosFFT::fft2(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fft2(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fft2(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fft2(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_fft2_2difft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.ifft2 is identical to np.ifft(np.ifft(x, axis=1), axis=0)
  KokkosFFT::ifft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::ifft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  KokkosFFT::ifft2(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifft2(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifft2(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifft2(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_fft2_2drfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType x("x", n0, n1), x_ref("x_ref", n0, n1);
  ComplexView2DType out("out", n0, n1/2+1), out1("out1", n0, n1/2+1), out2("out2", n0, n1/2+1);
  ComplexView2DType out_b("out_b", n0, n1/2+1), out_o("out_o", n0, n1/2+1), out_f("out_f", n0, n1/2+1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1);
  Kokkos::deep_copy(x_ref, x);
  Kokkos::fence();

  // np.rfft2 is identical to np.fft(np.rfft(x, axis=1), axis=0)
  KokkosFFT::rfft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_fft2_2dirfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1/2+1), x_ref("x_ref", n0, n1/2+1);
  ComplexView2DType out1("out1", n0, n1/2+1);
  RealView2DType out2("out2", n0, n1), out("out", n0, n1);
  RealView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  // np.irfft2 is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  KokkosFFT::ifft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, 0);
  KokkosFFT::irfft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, 1);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

// fft2 on 2D Views
TEST(FFT2D, 2DFFT_2DLeftViewFloat) {
  test_fft2_2dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DFFT_2DLeftViewDouble) {
  test_fft2_2dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DFFT_2DRightViewFloat) {
  test_fft2_2dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFT2D, 2DFFT_2DRightViewDouble) {
  test_fft2_2dfft_2dview<double, Kokkos::LayoutRight>();
}

// ifft2 on 2D Views
TEST(FFT2D, 2DIFFT_2DLeftViewFloat) {
  test_fft2_2difft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DIFFT_2DLeftViewDouble) {
  test_fft2_2difft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DIFFT_2DRightViewFloat) {
  test_fft2_2difft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFT2D, 2DIFFT_2DRightViewDouble) {
  test_fft2_2difft_2dview<double, Kokkos::LayoutRight>();
}

// rfft2 on 2D Views
TEST(FFT2D, 2DRFFT_2DLeftViewFloat) {
  test_fft2_2drfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DRFFT_2DLeftViewDouble) {
  test_fft2_2drfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DRFFT_2DRightViewFloat) {
  test_fft2_2drfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFT2D, 2DRFFT_2DRightViewDouble) {
  test_fft2_2drfft_2dview<double, Kokkos::LayoutRight>();
}

// irfft2 on 2D Views
TEST(FFT2D, 2DIRFFT_2DLeftViewFloat) {
  test_fft2_2dirfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DIRFFT_2DLeftViewDouble) {
  test_fft2_2dirfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFT2D, 2DIRFFT_2DRightViewFloat) {
  test_fft2_2dirfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFT2D, 2DIRFFT_2DRightViewDouble) {
  test_fft2_2dirfft_2dview<double, Kokkos::LayoutRight>();
}

// Tests for FFTN
template <typename T, typename LayoutType>
void test_fftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  KokkosFFT::fft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  KokkosFFT::fftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;

  KokkosFFT::fftn(x, out, axes_type{-2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fftn(x, out_b, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fftn(x, out_o, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fftn(x, out_f, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_fftn_3dfft_3dview(T atol=1.0e-6) {
  const int n0 = 4, n1 = 6, n2 = 8;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2);
  ComplexView3DType out("out", n0, n1, n2), out1("out1", n0, n1, n2), out2("out2", n0, n1, n2), out3("out3", n0, n1, n2);
  ComplexView3DType out_b("out_b", n0, n1, n2), out_o("out_o", n0, n1, n2), out_f("out_f", n0, n1, n2);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.fftn for 3D array is identical to np.fft(np.fft(np.fft(x, axis=2), axis=1), axis=0)
  KokkosFFT::fft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::fft(out2, out3, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  KokkosFFT::fftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, atol) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, atol) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, atol) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, atol) );

  // Same tests with specifying axes
  // np.fftn for 3D array is identical to np.fft(np.fft(np.fft(x, axis=2), axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;

  KokkosFFT::fftn(x, out, axes_type{-3, -2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fftn(x, out_b, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fftn(x, out_o, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fftn(x, out_f, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, atol) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, atol) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, atol) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, atol) );
}

template <typename T, typename LayoutType>
void test_ifftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.ifftn for 2D array is identical to np.ifft(np.ifft(x, axis=1), axis=0)
  KokkosFFT::ifft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::ifft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  KokkosFFT::ifftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;

  KokkosFFT::ifftn(x, out, axes_type{-2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifftn(x, out_b, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifftn(x, out_o, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifftn(x, out_f, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_ifftn_3dfft_3dview() {
  const int n0 = 4, n1 = 6, n2 = 8;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2);
  ComplexView3DType out("out", n0, n1, n2), out1("out1", n0, n1, n2), out2("out2", n0, n1, n2), out3("out3", n0, n1, n2);
  ComplexView3DType out_b("out_b", n0, n1, n2), out_o("out_o", n0, n1, n2), out_f("out_f", n0, n1, n2);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.ifftn for 3D array is identical to np.ifft(np.ifft(np.ifft(x, axis=2), axis=1), axis=0)
  KokkosFFT::ifft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  KokkosFFT::ifft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::ifft(out2, out3, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  KokkosFFT::ifftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.ifftn for 3D array is identical to np.ifft(np.ifft(np.ifft(x, axis=2), axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;

  KokkosFFT::ifftn(x, out, axes_type{-3, -2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifftn(x, out_b, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifftn(x, out_o, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifftn(x, out_f, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_rfftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType x("x", n0, n1), x_ref("x_ref", n0, n1);
  ComplexView2DType out("out", n0, n1/2+1), out1("out1", n0, n1/2+1), out2("out2", n0, n1/2+1);
  ComplexView2DType out_b("out_b", n0, n1/2+1), out_o("out_o", n0, n1/2+1), out_f("out_f", n0, n1/2+1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1);
  Kokkos::deep_copy(x_ref, x);
  Kokkos::fence();

  // np.rfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  KokkosFFT::rfft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.rfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out, axes_type{-2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_b, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_o, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_f, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_rfftn_3dfft_3dview() {
  const int n0 = 4, n1 = 6, n2 = 8;
  using RealView3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  RealView3DType x("x", n0, n1, n2), x_ref("x_ref", n0, n1, n2);
  ComplexView3DType out("out", n0, n1, n2/2+1), out1("out1", n0, n1, n2/2+1), out2("out2", n0, n1, n2/2+1), out3("out3", n0, n1, n2/2+1);
  ComplexView3DType out_b("out_b", n0, n1, n2/2+1), out_o("out_o", n0, n1, n2/2+1), out_f("out_f", n0, n1, n2/2+1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1);
  Kokkos::deep_copy(x_ref, x);
  Kokkos::fence();

  // np.rfftn for 3D array is identical to np.fft(np.fft(np.rfft(x, axis=2), axis=1), axis=0)
  KokkosFFT::rfft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::fft(out2, out3, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.rfftn for 3D array is identical to np.fft(np.fft(np.rfft(x, axis=2), axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out, axes_type{-3, -2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_b, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_o, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(x, out_f, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_irfftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType    = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType = Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1/2+1), x_ref("x_ref", n0, n1/2+1);
  ComplexView2DType out1("out1", n0, n1/2+1);
  RealView2DType out2("out2", n0, n1), out("out", n0, n1);
  RealView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 2D array is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  KokkosFFT::ifft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.irfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out, axes_type{-2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_b, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_o, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_f, axes_type{-2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

template <typename T, typename LayoutType>
void test_irfftn_3dfft_3dview() {
  const int n0 = 4, n1 = 6, n2 = 8;
  using RealView3DType    = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType = Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  ComplexView3DType x("x", n0, n1, n2/2+1), x_ref("x_ref", n0, n1, n2/2+1);
  ComplexView3DType out1("out1", n0, n1, n2/2+1), out2("out2", n0, n1, n2/2+1);
  RealView3DType out("out", n0, n1, n2), out3("out3", n0, n1, n2);
  RealView3DType out_b("out_b", n0, n1, n2), out_o("out_o", n0, n1, n2), out_f("out_f", n0, n1, n2);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 3D array is identical to np.irfft(np.ifft(np.ifft(x, axis=0), axis=1), axis=2)
  KokkosFFT::ifft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::ifft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::irfft(out2, out3, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, 1.e-6) );

  // Same tests with specifying axes
  // np.irfftn for 3D array is identical to np.irfft(np.ifft(np.ifft(x, axis=0), axis=1), axis=2)
  using axes_type = KokkosFFT::axis_type<3>;

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out, axes_type{-3, -2, -1}); // default: KokkosFFT::FFT_Normalization::BACKWARD

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_b, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::BACKWARD);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_o, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::ORTHO);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(x, out_f, axes_type{-3, -2, -1}, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, 1.0/sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0/static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE( allclose(out,   out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out3, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out3, 1.e-5, 1.e-6) );
}

// fftn on 2D Views
TEST(FFTN, 2DFFT_2DLeftViewFloat) {
  test_fftn_2dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DFFT_2DLeftViewDouble) {
  test_fftn_2dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DFFT_2DRightViewFloat) {
  test_fftn_2dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 2DFFT_2DRightViewDouble) {
  test_fftn_2dfft_2dview<double, Kokkos::LayoutRight>();
}

// fftn on 3D Views
TEST(FFTN, 3DFFT_3DLeftViewFloat) {
  test_fftn_3dfft_3dview<float, Kokkos::LayoutLeft>(5.0e-5);
}

TEST(FFTN, 3DFFT_3DLeftViewDouble) {
  test_fftn_3dfft_3dview<double, Kokkos::LayoutLeft>(1.0e-6);
}

TEST(FFTN, 3DFFT_3DRightViewFloat) {
  test_fftn_3dfft_3dview<float, Kokkos::LayoutRight>(5.0e-5);
}

TEST(FFTN, 3DFFT_3DRightViewDouble) {
  test_fftn_3dfft_3dview<double, Kokkos::LayoutRight>(1.0e-6);
}

// ifftn on 2D Views
TEST(FFTN, 2DIFFT_2DLeftViewFloat) {
  test_ifftn_2dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DIFFT_2DLeftViewDouble) {
  test_ifftn_2dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DIFFT_2DRightViewFloat) {
  test_ifftn_2dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 2DIFFT_2DRightViewDouble) {
  test_ifftn_2dfft_2dview<double, Kokkos::LayoutRight>();
}

// ifftn on 3D Views
TEST(FFTN, 3DIFFT_3DLeftViewFloat) {
  test_ifftn_3dfft_3dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 3DIFFT_3DLeftViewDouble) {
  test_ifftn_3dfft_3dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 3DIFFT_3DRightViewFloat) {
  test_ifftn_3dfft_3dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 3DIFFT_3DRightViewDouble) {
  test_ifftn_3dfft_3dview<double, Kokkos::LayoutRight>();
}

// rfftn on 2D Views
TEST(FFTN, 2DRFFT_2DLeftViewFloat) {
  test_rfftn_2dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DRFFT_2DLeftViewDouble) {
  test_rfftn_2dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DRFFT_2DRightViewFloat) {
  test_rfftn_2dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 2DRFFT_2DRightViewDouble) {
  test_rfftn_2dfft_2dview<double, Kokkos::LayoutRight>();
}

// rfftn on 3D Views
TEST(FFTN, 3DRFFT_3DLeftViewFloat) {
  test_rfftn_3dfft_3dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 3DRFFT_3DLeftViewDouble) {
  test_rfftn_3dfft_3dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 3DRFFT_3DRightViewFloat) {
  test_rfftn_3dfft_3dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 3DRFFT_3DRightViewDouble) {
  test_rfftn_3dfft_3dview<double, Kokkos::LayoutRight>();
}

// irfftn on 2D Views
TEST(FFTN, 2DIRFFT_2DLeftViewFloat) {
  test_irfftn_2dfft_2dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DIRFFT_2DLeftViewDouble) {
  test_irfftn_2dfft_2dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 2DIRFFT_2DRightViewFloat) {
  test_irfftn_2dfft_2dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 2DIRFFT_2DRightViewDouble) {
  test_irfftn_2dfft_2dview<double, Kokkos::LayoutRight>();
}

// irfftn on 3D Views
TEST(FFTN, 3DIRFFT_3DLeftViewFloat) {
  test_irfftn_3dfft_3dview<float, Kokkos::LayoutLeft>();
}

TEST(FFTN, 3DIRFFT_3DLeftViewDouble) {
  test_irfftn_3dfft_3dview<double, Kokkos::LayoutLeft>();
}

TEST(FFTN, 3DIRFFT_3DRightViewFloat) {
  test_irfftn_3dfft_3dview<float, Kokkos::LayoutRight>();
}

TEST(FFTN, 3DIRFFT_3DRightViewDouble) {
  test_irfftn_3dfft_3dview<double, Kokkos::LayoutRight>();
}