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

TEST(FFT1D, 1DbatchedFFT_2DRightView) {
  const int n0 = 10, n1 = 12;

  RightView2D<Kokkos::complex<double> > x("x", n0, n1), _x("_x", n0, n1);
  RightView2D<Kokkos::complex<double> > out("out", n0, n1), outr("outr", n0, n1/2+1);
  RightView2D<double> xr("xr", n0, n1), _xr("_x", n0, n1);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  Kokkos::fence();

  KokkosFFT::fft(x, out, KokkosFFT::FFT_Normalization::BACKWARD, -1);
  KokkosFFT::ifft(out, _x, KokkosFFT::FFT_Normalization::BACKWARD, -1);

  KokkosFFT::rfft(xr, outr, KokkosFFT::FFT_Normalization::BACKWARD, -1);
  KokkosFFT::irfft(outr, _xr, KokkosFFT::FFT_Normalization::BACKWARD, -1);

  EXPECT_TRUE( allclose(_x, x, 1.e-5, 1.e-12) );
  EXPECT_TRUE( allclose(_xr, xr, 1.e-5, 1.e-12) );
}

TEST(FFT1D, 1DbatchedFFT_3DRightView) {
  const int n0 = 10, n1 = 12, n2 = 8;

  RightView3D<Kokkos::complex<double> > x("x", n0, n1, n2), _x("_x", n0, n1, n2);
  RightView3D<Kokkos::complex<double> > out("out", n0, n1, n2), outr("outr", n0, n1, n2/2+1);
  RightView3D<double> xr("xr", n0, n1, n2), _xr("_x", n0, n1, n2);

  const Kokkos::complex<double> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  Kokkos::fence();

  KokkosFFT::fft(x, out, KokkosFFT::FFT_Normalization::BACKWARD, -1);
  KokkosFFT::ifft(out, _x, KokkosFFT::FFT_Normalization::BACKWARD, -1);

  KokkosFFT::rfft(xr, outr, KokkosFFT::FFT_Normalization::BACKWARD, -1);
  KokkosFFT::irfft(outr, _xr, KokkosFFT::FFT_Normalization::BACKWARD, -1);

  EXPECT_TRUE( allclose(_x, x, 1.e-5, 1.e-12) );
  EXPECT_TRUE( allclose(_xr, xr, 1.e-5, 1.e-12) );
}

/*
TEST(FFT1D, FFT2Left) {
  const int n0 = 30, n1 = 20;

  LeftView2D<Kokkos::complex<double> > x("x", n0, n1);
  LeftView2D<Kokkos::complex<double> > out("out", n0, n1), out1("out1", n0, n1), out2("out2", n0, n1);
  LeftView2D<Kokkos::complex<double> > out_b("out_b", n0, n1), out_o("out_o", n0, n1), out_f("out_f", n0, n1);

  KokkosFFT::fft(x, out1, KokkosFFT::FFT_Normalization::BACKWARD, 1);
  KokkosFFT::fft(out1, out2, KokkosFFT::FFT_Normalization::BACKWARD, 0);
  KokkosFFT::fft2(x, out);
  KokkosFFT::fft2(x, out_b, KokkosFFT::FFT_Normalization::BACKWARD);
  KokkosFFT::fft2(x, out_o, KokkosFFT::FFT_Normalization::ORTHO);
  KokkosFFT::fft2(x, out_f, KokkosFFT::FFT_Normalization::FORWARD);

  multiply(out_o, sqrt(static_cast<double>(n0 * n1)));
  multiply(out_f, static_cast<double>(n0 * n1));

  EXPECT_TRUE( allclose(out2, out, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out2, out_b, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out2, out_o, 1.e-5, 1.e-6) );
  EXPECT_TRUE( allclose(out2, out_f, 1.e-5, 1.e-6) );
}
*/