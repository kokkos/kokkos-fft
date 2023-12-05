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

TEST(FFT1D, Identity) {
  const int maxlen = 8;

  View1D<Kokkos::complex<double> > a("a", maxlen), _a("_a", maxlen);
  View1D<Kokkos::complex<double> > out("out", maxlen), outr("outr", maxlen/2+1);
  View1D<double> ar("ar", maxlen), _ar("_ar", maxlen);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(a,  random_pool, I);
  Kokkos::fill_random(ar, random_pool, 1.0);

  Kokkos::fence();

  KokkosFFT::fft(a, out);
  KokkosFFT::ifft(out, _a);

  KokkosFFT::rfft(ar, outr);
  KokkosFFT::irfft(outr, _ar);

  EXPECT_TRUE( allclose(_a, a, 1.e-5, 1.e-12) );
  EXPECT_TRUE( allclose(_ar, ar, 1.e-5, 1.e-12) );
}

TEST(FFT1D, FFT) {
  const int len = 30;

  View1D<Kokkos::complex<double> > x("x", len), out("out", len), ref("ref", len);
  View1D<Kokkos::complex<double> > out_b("out_b", len), out_o("out_o", len), out_f("out_f", len);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  KokkosFFT::fft(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::fft(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fft(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fft(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  fft1(x, ref);
  multiply(out_o, sqrt(static_cast<double>(len)));
  multiply(out_f, static_cast<double>(len));

  EXPECT_TRUE( allclose(out,   ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, ref, 1.e-5, 1.e-6) );
}

TEST(FFT1D, IFFT) {
  const int len = 30;

  View1D<Kokkos::complex<double> > x("x", len), out("out", len), ref("ref", len);
  View1D<Kokkos::complex<double> > out_b("out_b", len), out_o("out_o", len), out_f("out_f", len);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  KokkosFFT::ifft(x, out); // default: KokkosFFT::FFT_Normalization::BACKWARD
  KokkosFFT::ifft(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::ifft(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::ifft(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  ifft1(x, ref);
  multiply(out_o, sqrt(static_cast<double>(len)));
  multiply(out_b, static_cast<double>(len));
  multiply(out, static_cast<double>(len));

  EXPECT_TRUE( allclose(out,   ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, ref, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, ref, 1.e-5, 1.e-6) );
}

TEST(FFT1D, 1DbatchedFFT_2DLeftView) {
  const int n0 = 10, n1 = 12;

  LeftView2D<Kokkos::complex<double> > x("x", n0, n1), ref_x("ref_x", n0, n1);
  LeftView2D<Kokkos::complex<double> > x_axis0("x_axis0", n0, n1), x_axis1("x_axis1", n0, n1);
  LeftView2D<Kokkos::complex<double> > out_axis0("out_axis0", n0, n1), ref_out_axis0("ref_out_axis0", n0, n1);
  LeftView2D<Kokkos::complex<double> > out_axis1("out_axis1", n0, n1), ref_out_axis1("ref_out_axis1", n0, n1);

  LeftView2D<double> xr("xr", n0, n1), ref_xr("ref_xr", n0, n1);
  LeftView2D<double> xr_axis0("xr_axis0", n0, n1), xr_axis1("xr_axis1", n0, n1);
  LeftView2D<Kokkos::complex<double> > outr_axis0("outr_axis0", n0/2+1, n1), outr_axis1("outr_axis1", n0, n1/2+1);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  // Since HIP FFT destructs the input data, we need to keep the input data in different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  Kokkos::fence();

  // Along axis 0
  // Perform batched 1D (along 0th axis) FFT sequentially
  for(int i1=0; i1<n1; i1++) {
    auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1);
    auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1);
    fft1(sub_x, sub_ref);
  }

  KokkosFFT::fft(x, out_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(out_axis0, ref_out_axis0, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis0, x_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(x_axis0, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(outr_axis0, xr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  EXPECT_TRUE( allclose(xr_axis0, ref_xr, 1.e-5, 1.e-12) );

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1 (transpose neeed)
  // Perform batched 1D (along 1st axis) FFT sequentially
  for(int i0=0; i0<n0; i0++) {
    auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL);
    auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL);
    fft1(sub_x, sub_ref);
  }

  KokkosFFT::fft(x, out_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/-1);
  EXPECT_TRUE( allclose(out_axis1, ref_out_axis1, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis1, x_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/-1);
  EXPECT_TRUE( allclose(x_axis1, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/-1);
  KokkosFFT::irfft(outr_axis1, xr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/-1);

  EXPECT_TRUE( allclose(xr_axis1, ref_xr, 1.e-5, 1.e-12) );
}

TEST(FFT1D, 1DbatchedFFT_2DRightView) {
  const int n0 = 10, n1 = 12;

  RightView2D<Kokkos::complex<double> > x("x", n0, n1), ref_x("ref_x", n0, n1);
  RightView2D<Kokkos::complex<double> > x_axis0("x_axis0", n0, n1), x_axis1("x_axis1", n0, n1);
  RightView2D<Kokkos::complex<double> > out_axis0("out_axis0", n0, n1), ref_out_axis0("ref_out_axis0", n0, n1);
  RightView2D<Kokkos::complex<double> > out_axis1("out_axis1", n0, n1), ref_out_axis1("ref_out_axis1", n0, n1);

  RightView2D<double> xr("xr", n0, n1), ref_xr("ref_xr", n0, n1);
  RightView2D<double> xr_axis0("xr_axis0", n0, n1), xr_axis1("xr_axis1", n0, n1);
  RightView2D<Kokkos::complex<double> > outr_axis0("outr_axis0", n0/2+1, n1), outr_axis1("outr_axis1", n0, n1/2+1);

  const Kokkos::complex<double> I(1.0, 1.0);
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
  EXPECT_TRUE( allclose(out_axis0, ref_out_axis0, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis0, x_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(x_axis0, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(outr_axis0, xr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  EXPECT_TRUE( allclose(xr_axis0, ref_xr, 1.e-5, 1.e-12) );

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
  EXPECT_TRUE( allclose(out_axis1, ref_out_axis1, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis1, x_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(x_axis1, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::irfft(outr_axis1, xr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);

  EXPECT_TRUE( allclose(xr_axis1, ref_xr, 1.e-5, 1.e-12) );
}

TEST(FFT1D, 1DbatchedFFT_3DLeftView) {
  const int n0 = 10, n1 = 12, n2 = 8;

  LeftView3D<Kokkos::complex<double> > x("x", n0, n1, n2), ref_x("ref_x", n0, n1, n2);
  LeftView3D<Kokkos::complex<double> > x_axis0("x_axis0", n0, n1, n2), x_axis1("x_axis1", n0, n1, n2), x_axis2("x_axis2", n0, n1, n2);
  LeftView3D<Kokkos::complex<double> > out_axis0("out_axis0", n0, n1, n2), ref_out_axis0("ref_out_axis0", n0, n1, n2);
  LeftView3D<Kokkos::complex<double> > out_axis1("out_axis1", n0, n1, n2), ref_out_axis1("ref_out_axis1", n0, n1, n2);
  LeftView3D<Kokkos::complex<double> > out_axis2("out_axis2", n0, n1, n2), ref_out_axis2("ref_out_axis2", n0, n1, n2);

  LeftView3D<double> xr("xr", n0, n1, n2), ref_xr("ref_xr", n0, n1, n2);
  LeftView3D<double> xr_axis0("xr_axis0", n0, n1, n2), xr_axis1("xr_axis1", n0, n1, n2), xr_axis2("xr_axis2", n0, n1, n2);
  LeftView3D<Kokkos::complex<double> > outr_axis0("outr_axis0", n0/2+1, n1, n2), outr_axis1("outr_axis1", n0, n1/2+1, n2), outr_axis2("outr_axis2", n0, n1, n2/2+1);

  const Kokkos::complex<double> I(1.0, 1.0);
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
  EXPECT_TRUE( allclose(out_axis0, ref_out_axis0, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis0, x_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(x_axis0, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(outr_axis0, xr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  EXPECT_TRUE( allclose(xr_axis0, ref_xr, 1.e-5, 1.e-12) );

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
  EXPECT_TRUE( allclose(out_axis1, ref_out_axis1, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis1, x_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(x_axis1, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::irfft(outr_axis1, xr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);

  EXPECT_TRUE( allclose(xr_axis1, ref_xr, 1.e-5, 1.e-12) );

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
  EXPECT_TRUE( allclose(out_axis2, ref_out_axis2, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis2, x_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  EXPECT_TRUE( allclose(x_axis2, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  KokkosFFT::irfft(outr_axis2, xr_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);

  EXPECT_TRUE( allclose(xr_axis2, ref_xr, 1.e-5, 1.e-12) );
}

TEST(FFT1D, 1DbatchedFFT_3DRightView) {
  const int n0 = 10, n1 = 12, n2 = 8;

  RightView3D<Kokkos::complex<double> > x("x", n0, n1, n2), ref_x("ref_x", n0, n1, n2);
  RightView3D<Kokkos::complex<double> > x_axis0("x_axis0", n0, n1, n2), x_axis1("x_axis1", n0, n1, n2), x_axis2("x_axis2", n0, n1, n2);
  RightView3D<Kokkos::complex<double> > out_axis0("out_axis0", n0, n1, n2), ref_out_axis0("ref_out_axis0", n0, n1, n2);
  RightView3D<Kokkos::complex<double> > out_axis1("out_axis1", n0, n1, n2), ref_out_axis1("ref_out_axis1", n0, n1, n2);
  RightView3D<Kokkos::complex<double> > out_axis2("out_axis2", n0, n1, n2), ref_out_axis2("ref_out_axis2", n0, n1, n2);

  RightView3D<double> xr("xr", n0, n1, n2), ref_xr("ref_xr", n0, n1, n2);
  RightView3D<double> xr_axis0("xr_axis0", n0, n1, n2), xr_axis1("xr_axis1", n0, n1, n2), xr_axis2("xr_axis2", n0, n1, n2);
  RightView3D<Kokkos::complex<double> > outr_axis0("outr_axis0", n0/2+1, n1, n2), outr_axis1("outr_axis1", n0, n1/2+1, n2), outr_axis2("outr_axis2", n0, n1, n2/2+1);

  const Kokkos::complex<double> I(1.0, 1.0);
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
  EXPECT_TRUE( allclose(out_axis0, ref_out_axis0, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis0, x_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  EXPECT_TRUE( allclose(x_axis0, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);
  KokkosFFT::irfft(outr_axis0, xr_axis0, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/0);

  EXPECT_TRUE( allclose(xr_axis0, ref_xr, 1.e-5, 1.e-12) );

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
  EXPECT_TRUE( allclose(out_axis1, ref_out_axis1, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis1, x_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  EXPECT_TRUE( allclose(x_axis1, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);
  KokkosFFT::irfft(outr_axis1, xr_axis1, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/1);

  EXPECT_TRUE( allclose(xr_axis1, ref_xr, 1.e-5, 1.e-12) );

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
  EXPECT_TRUE( allclose(out_axis2, ref_out_axis2, 1.e-5, 1.e-12) );

  KokkosFFT::ifft(out_axis2, x_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  EXPECT_TRUE( allclose(x_axis2, ref_x, 1.e-5, 1.e-12) );

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(xr, outr_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);
  KokkosFFT::irfft(outr_axis2, xr_axis2, KokkosFFT::FFT_Normalization::BACKWARD, /*axis=*/2);

  EXPECT_TRUE( allclose(xr_axis2, ref_xr, 1.e-5, 1.e-12) );
}

TEST(FFT2D, 2DFFT_2DLeftView) {
  const int n0 = 4, n1 = 6;

  LeftView2D<Kokkos::complex<double> > x("x", n0, n1);
  LeftView2D<Kokkos::complex<double> > out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  LeftView2D<Kokkos::complex<double> > out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
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

  multiply(out_o, sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DFFT_2DRightView) {
  const int n0 = 4, n1 = 6;

  RightView2D<Kokkos::complex<double> > x("x", n0, n1);
  RightView2D<Kokkos::complex<double> > out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  RightView2D<Kokkos::complex<double> > out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
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

  multiply(out_o, sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DIFFT_2DLeftView) {
  const int n0 = 4, n1 = 6;

  LeftView2D<Kokkos::complex<double> > x("x", n0, n1);
  LeftView2D<Kokkos::complex<double> > out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  LeftView2D<Kokkos::complex<double> > out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
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

  multiply(out_o, 1.0/sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DIFFT_2DRightView) {
  const int n0 = 4, n1 = 6;

  RightView2D<Kokkos::complex<double> > x("x", n0, n1);
  RightView2D<Kokkos::complex<double> > out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  RightView2D<Kokkos::complex<double> > out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
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

  multiply(out_o, 1.0/sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DRFFT_2DLeftView) {
  const int n0 = 4, n1 = 6;

  LeftView2D<double> x("x", n0, n1), x_ref("x_ref", n0, n1);
  LeftView2D<Kokkos::complex<double> > out("out", n0, n1/2+1), out1("out1", n0, n1/2+1), out2("out2", n0, n1/2+1);
  LeftView2D<Kokkos::complex<double> > out_b("out_b", n0, n1/2+1), out_o("out_o", n0, n1/2+1), out_f("out_f", n0, n1/2+1);

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

  multiply(out_o, sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DRFFT_2DRightView) {
  const int n0 = 4, n1 = 6;

  RightView2D<double> x("x", n0, n1), x_ref("x_ref", n0, n1);
  RightView2D<Kokkos::complex<double> > out("out", n0, n1/2+1), out1("out1", n0, n1/2+1), out2("out2", n0, n1/2+1);
  RightView2D<Kokkos::complex<double> > out_b("out_b", n0, n1/2+1), out_o("out_o", n0, n1/2+1), out_f("out_f", n0, n1/2+1);

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

  multiply(out_o, sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DIRFFT_2DLeftView) {
  const int n0 = 4, n1 = 6;

  LeftView2D<Kokkos::complex<double> > x("x", n0, n1/2+1), x_ref("x_ref", n0, n1/2+1);
  LeftView2D<Kokkos::complex<double> > out1("out1", n0, n1/2+1);
  LeftView2D<double> out2("out2", n0, n1), out("out", n0, n1);
  LeftView2D<double> out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  Kokkos::fence();

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

  multiply(out_o, 1.0/sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}

TEST(FFT2D, 2DIRFFT_2DRightView) {
  const int n0 = 4, n1 = 6;

  RightView2D<Kokkos::complex<double> > x("x", n0, n1/2+1), x_ref("x_ref", n0, n1/2+1);
  RightView2D<Kokkos::complex<double> > out1("out1", n0, n1/2+1);
  RightView2D<double> out2("out2", n0, n1), out("out", n0, n1);
  RightView2D<double> out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  Kokkos::fence();

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

  multiply(out_o, 1.0/sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, 1.0/static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out,   out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_b, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_o, out2, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out_f, out2, 1.e-5, 1.e-6) );
}