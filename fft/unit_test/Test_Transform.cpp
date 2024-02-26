#include <gtest/gtest.h>
#include <algorithm>
#include <random>
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
  using value_type      = typename ViewType::non_const_value_type;
  using real_value_type = KokkosFFT::Impl::real_type_t<value_type>;

  static_assert(KokkosFFT::Impl::is_complex<value_type>::value,
                "fft1: ViewType must be complex");

  const value_type I(0.0, 1.0);
  std::size_t L = in.size();

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(L, Kokkos::AUTO),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<execution_space>::member_type& team_member) {
        const int j = team_member.league_rank();

        value_type sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, L),
            [&](const int i, value_type& lsum) {
              auto phase = -2 * I * M_PI * static_cast<real_value_type>(i) /
                           static_cast<real_value_type>(L);

              auto tmp_in = in(i);
              lsum +=
                  tmp_in * Kokkos::exp(static_cast<real_value_type>(j) * phase);
            },
            sum);

        out(j) = sum;
      });
}

/// Kokkos equivalent of ifft1 with numpy
/// def ifft1(x):
///    L = len(x)
///    phase = 2j * np.pi * (np.arange(L) / L)
///    phase = np.arange(L).reshape(-1, 1) * phase
///    return np.sum(x*np.exp(phase), axis=1)
template <typename ViewType>
void ifft1(ViewType& in, ViewType& out) {
  using value_type      = typename ViewType::non_const_value_type;
  using real_value_type = KokkosFFT::Impl::real_type_t<value_type>;

  static_assert(KokkosFFT::Impl::is_complex<value_type>::value,
                "ifft1: ViewType must be complex");

  const value_type I(0.0, 1.0);
  std::size_t L = in.size();

  Kokkos::parallel_for(
      Kokkos::TeamPolicy<execution_space>(L, Kokkos::AUTO),
      KOKKOS_LAMBDA(
          const Kokkos::TeamPolicy<execution_space>::member_type& team_member) {
        const int j = team_member.league_rank();

        value_type sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, L),
            [&](const int i, value_type& lsum) {
              auto phase = 2 * I * M_PI * static_cast<real_value_type>(i) /
                           static_cast<real_value_type>(L);

              auto tmp_in = in(i);
              lsum +=
                  tmp_in * Kokkos::exp(static_cast<real_value_type>(j) * phase);
            },
            sum);

        out(j) = sum;
      });
}

using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight> >;

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

TYPED_TEST_SUITE(FFT1D, test_types);
TYPED_TEST_SUITE(FFT2D, test_types);
TYPED_TEST_SUITE(FFTND, test_types);

// Tests for 1D FFT
template <typename T, typename LayoutType>
void test_fft1_identity(T atol = 1.0e-12) {
  const int maxlen     = 32;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  for (int i = 1; i < maxlen; i++) {
    ComplexView1DType a("a", i), _a("_a", i), a_ref("a_ref", i);
    ComplexView1DType out("out", i), outr("outr", i / 2 + 1);
    RealView1DType ar("ar", i), _ar("_ar", i), ar_ref("ar_ref", i);

    const Kokkos::complex<T> I(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::fill_random(a, random_pool, I);
    Kokkos::fill_random(ar, random_pool, 1.0);
    Kokkos::deep_copy(a_ref, a);
    Kokkos::deep_copy(ar_ref, ar);

    Kokkos::fence();

    KokkosFFT::fft(execution_space(), a, out);
    KokkosFFT::ifft(execution_space(), out, _a);

    KokkosFFT::rfft(execution_space(), ar, outr);
    KokkosFFT::irfft(execution_space(), outr, _ar);

    EXPECT_TRUE(allclose(_a, a_ref, 1.e-5, atol));
    EXPECT_TRUE(allclose(_ar, ar_ref, 1.e-5, atol));
  }
}

template <typename T, typename LayoutType>
void test_fft1_identity_reuse_plan(T atol = 1.0e-12) {
  const int maxlen     = 32;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  for (int i = 1; i < maxlen; i++) {
    ComplexView1DType a("a", i), _a("_a", i), a_ref("a_ref", i);
    ComplexView1DType out("out", i), outr("outr", i / 2 + 1);
    RealView1DType ar("ar", i), _ar("_ar", i), ar_ref("ar_ref", i);

    const Kokkos::complex<T> I(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::fill_random(a, random_pool, I);
    Kokkos::fill_random(ar, random_pool, 1.0);
    Kokkos::deep_copy(a_ref, a);
    Kokkos::deep_copy(ar_ref, ar);

    Kokkos::fence();

    int axis = -1;
    KokkosFFT::Impl::Plan fft_plan(execution_space(), a, out,
                                   KokkosFFT::Direction::forward, axis);
    KokkosFFT::fft(execution_space(), a, out, fft_plan);

    KokkosFFT::Impl::Plan ifft_plan(execution_space(), out, _a,
                                    KokkosFFT::Direction::backward, axis);
    KokkosFFT::ifft(execution_space(), out, _a, ifft_plan);

    KokkosFFT::Impl::Plan rfft_plan(execution_space(), ar, outr,
                                    KokkosFFT::Direction::forward, axis);
    KokkosFFT::rfft(execution_space(), ar, outr, rfft_plan);

    KokkosFFT::Impl::Plan irfft_plan(execution_space(), outr, _ar,
                                     KokkosFFT::Direction::backward, axis);
    KokkosFFT::irfft(execution_space(), outr, _ar, irfft_plan);

    EXPECT_TRUE(allclose(_a, a_ref, 1.e-5, atol));
    EXPECT_TRUE(allclose(_ar, ar_ref, 1.e-5, atol));
  }

  ComplexView1DType a("a", maxlen), _a("_a", maxlen), a_ref("a_ref", maxlen);
  ComplexView1DType out("out", maxlen), outr("outr", maxlen / 2 + 1);
  RealView1DType ar("ar", maxlen), _ar("_ar", maxlen), ar_ref("ar_ref", maxlen);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(a, random_pool, I);
  Kokkos::fill_random(ar, random_pool, 1.0);
  Kokkos::deep_copy(a_ref, a);
  Kokkos::deep_copy(ar_ref, ar);

  Kokkos::fence();

  // Create correct plans
  int axis = -1;
  KokkosFFT::Impl::Plan fft_plan(execution_space(), a, out,
                                 KokkosFFT::Direction::forward, axis);

  KokkosFFT::Impl::Plan ifft_plan(execution_space(), out, _a,
                                  KokkosFFT::Direction::backward, axis);

  KokkosFFT::Impl::Plan rfft_plan(execution_space(), ar, outr,
                                  KokkosFFT::Direction::forward, axis);

  KokkosFFT::Impl::Plan irfft_plan(execution_space(), outr, _ar,
                                   KokkosFFT::Direction::backward, axis);

  // Check if errors are correctly raised aginst wrong axis
  int wrong_axis = 0;
  EXPECT_THROW(KokkosFFT::fft(execution_space(), a, out, fft_plan,
                              KokkosFFT::Normalization::backward, wrong_axis),
               std::runtime_error);

  EXPECT_THROW(KokkosFFT::ifft(execution_space(), out, _a, ifft_plan,
                               KokkosFFT::Normalization::backward, wrong_axis),
               std::runtime_error);

  EXPECT_THROW(KokkosFFT::rfft(execution_space(), ar, outr, rfft_plan,
                               KokkosFFT::Normalization::backward, wrong_axis),
               std::runtime_error);

  EXPECT_THROW(KokkosFFT::irfft(execution_space(), outr, _ar, irfft_plan,
                                KokkosFFT::Normalization::backward, wrong_axis),
               std::runtime_error);

  // Check if errors are correctly raised aginst wrong dirction
  KokkosFFT::Direction wrong_fft_direction = KokkosFFT::Direction::backward;
  KokkosFFT::Impl::Plan wrong_fft_plan(execution_space(), a, out,
                                       wrong_fft_direction, axis);

  KokkosFFT::Direction wrong_ifft_direction = KokkosFFT::Direction::forward;
  KokkosFFT::Impl::Plan wrong_ifft_plan(execution_space(), out, _a,
                                        wrong_ifft_direction, axis);

  KokkosFFT::Impl::Plan wrong_rfft_plan(execution_space(), ar, outr,
                                        wrong_fft_direction, axis);
  KokkosFFT::Impl::Plan wrong_irfft_plan(execution_space(), outr, _ar,
                                         wrong_ifft_direction, axis);

  EXPECT_THROW(KokkosFFT::fft(execution_space(), a, out, wrong_fft_plan,
                              KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  EXPECT_THROW(KokkosFFT::ifft(execution_space(), out, _a, wrong_ifft_plan,
                               KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  EXPECT_THROW(KokkosFFT::rfft(execution_space(), ar, outr, wrong_rfft_plan,
                               KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  EXPECT_THROW(KokkosFFT::irfft(execution_space(), outr, _ar, wrong_irfft_plan,
                                KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // Check if errors are correctly raised aginst wrong extents
  const int maxlen_wrong = 32 * 2;
  ComplexView1DType a_wrong("a", maxlen_wrong), _a_wrong("_a", maxlen_wrong);
  ComplexView1DType out_wrong("out", maxlen_wrong),
      outr_wrong("outr", maxlen_wrong / 2 + 1);
  RealView1DType ar_wrong("ar", maxlen_wrong), _ar_wrong("_ar", maxlen_wrong);

  // fft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::fft(execution_space(), a_wrong, out, fft_plan,
                              KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::fft(execution_space(), a, out_wrong, fft_plan,
                              KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // ifft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::ifft(execution_space(), out_wrong, _a, ifft_plan,
                               KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::ifft(execution_space(), out, _a_wrong, ifft_plan,
                               KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // rfft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::rfft(execution_space(), ar_wrong, outr, rfft_plan,
                               KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::rfft(execution_space(), ar, out_wrong, rfft_plan,
                               KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // irfft
  // With incorrect input shape
  EXPECT_THROW(KokkosFFT::irfft(execution_space(), outr_wrong, _ar, irfft_plan,
                                KokkosFFT::Normalization::backward, axis),
               std::runtime_error);

  // With incorrect output shape
  EXPECT_THROW(KokkosFFT::irfft(execution_space(), outr, _ar_wrong, irfft_plan,
                                KokkosFFT::Normalization::backward, axis),
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  KokkosFFT::fft(execution_space(), x,
                 out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fft(execution_space(), x, out_b,
                 KokkosFFT::Normalization::backward);
  KokkosFFT::fft(execution_space(), x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::fft(execution_space(), x, out_f,
                 KokkosFFT::Normalization::forward);

  fft1(x, ref);
  multiply(out_o, sqrt(static_cast<T>(len)));
  multiply(out_f, static_cast<T>(len));

  EXPECT_TRUE(allclose(out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, ref, 1.e-5, 1.e-6));
}

template <typename T, typename LayoutType>
void test_fft1_1difft_1dview() {
  const int len = 30;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType x("x", len), out("out", len), ref("ref", len);
  ComplexView1DType out_b("out_b", len), out_o("out_o", len),
      out_f("out_f", len);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  KokkosFFT::ifft(execution_space(), x,
                  out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifft(execution_space(), x, out_b,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::ifft(execution_space(), x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::ifft(execution_space(), x, out_f,
                  KokkosFFT::Normalization::forward);

  ifft1(x, ref);
  multiply(out_o, sqrt(static_cast<T>(len)));
  multiply(out_b, static_cast<T>(len));
  multiply(out, static_cast<T>(len));

  EXPECT_TRUE(allclose(out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, ref, 1.e-5, 1.e-6));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x_herm, random_pool, I);

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

  Kokkos::fence();

  KokkosFFT::fft(execution_space(), x, ref);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm,
                  out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out_b,
                  KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out_o,
                  KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out_f,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(len)));
  multiply(out_f, static_cast<T>(len));

  EXPECT_TRUE(allclose(out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out, 1.e-5, 1.e-6));

  // Reuse plans
  int axis = -1;

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::Impl::Plan hfft_plan(execution_space(), x_herm, out,
                                  KokkosFFT::Direction::backward, axis);
  KokkosFFT::hfft(execution_space(), x_herm, out, hfft_plan);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out_b, hfft_plan,
                  KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out_o, hfft_plan,
                  KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out_f, hfft_plan,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(len)));
  multiply(out_f, static_cast<T>(len));

  EXPECT_TRUE(allclose(out, ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out, 1.e-5, 1.e-6));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x_herm, random_pool, I);

  auto h_x_herm = Kokkos::create_mirror_view(x_herm);
  Kokkos::deep_copy(h_x_herm, x_herm);

  auto last      = h_x_herm.extent(0) - 1;
  h_x_herm(0)    = h_x_herm(0).real();
  h_x_herm(last) = h_x_herm(last).real();

  Kokkos::deep_copy(x_herm, h_x_herm);
  Kokkos::deep_copy(x_herm_ref, h_x_herm);
  Kokkos::fence();

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm,
                  out1);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ihfft(execution_space(), out1,
                   out2);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1_b,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::ihfft(execution_space(), out1_b, out2_b,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1_o,
                  KokkosFFT::Normalization::ortho);
  KokkosFFT::ihfft(execution_space(), out1_o, out2_o,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1_f,
                  KokkosFFT::Normalization::forward);
  KokkosFFT::ihfft(execution_space(), out1_f, out2_f,
                   KokkosFFT::Normalization::forward);

  EXPECT_TRUE(allclose(out2, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out2_b, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out2_o, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out2_f, x_herm_ref, 1.e-5, 1.e-6));

  // Reuse plans
  int axis = -1;
  KokkosFFT::Impl::Plan hfft_plan(execution_space(), x_herm, out1,
                                  KokkosFFT::Direction::backward, axis);
  KokkosFFT::Impl::Plan ihfft_plan(execution_space(), out1, out2,
                                   KokkosFFT::Direction::forward, axis);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1,
                  hfft_plan);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ihfft(execution_space(), out1, out2,
                   ihfft_plan);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1_b, hfft_plan,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::ihfft(execution_space(), out1_b, out2_b, ihfft_plan,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1_o, hfft_plan,
                  KokkosFFT::Normalization::ortho);
  KokkosFFT::ihfft(execution_space(), out1_o, out2_o, ihfft_plan,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x_herm, x_herm_ref);
  KokkosFFT::hfft(execution_space(), x_herm, out1_f, hfft_plan,
                  KokkosFFT::Normalization::forward);
  KokkosFFT::ihfft(execution_space(), out1_f, out2_f, ihfft_plan,
                   KokkosFFT::Normalization::forward);

  EXPECT_TRUE(allclose(out2, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out2_b, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out2_o, x_herm_ref, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out2_f, x_herm_ref, 1.e-5, 1.e-6));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  Kokkos::fence();

  // Along axis 0 (transpose neeed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for (int i1 = 0; i1 < n1; i1++) {
    auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1);
    auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1);
    fft1(sub_x, sub_ref);
  }

  KokkosFFT::fft(execution_space(), x, out_axis0,
                 KokkosFFT::Normalization::backward, /*axis=*/0);
  EXPECT_TRUE(allclose(out_axis0, ref_out_axis0, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis0, x_axis0,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  EXPECT_TRUE(allclose(x_axis0, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis0,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  KokkosFFT::irfft(execution_space(), outr_axis0, xr_axis0,
                   KokkosFFT::Normalization::backward, /*axis=*/0);

  EXPECT_TRUE(allclose(xr_axis0, ref_xr, 1.e-5, atol));

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1
  // Perform batched 1D (along 1st axis) FFT sequentially
  for (int i0 = 0; i0 < n0; i0++) {
    auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL);
    auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL);
    fft1(sub_x, sub_ref);
  }

  KokkosFFT::fft(execution_space(), x, out_axis1,
                 KokkosFFT::Normalization::backward, /*axis=*/1);
  EXPECT_TRUE(allclose(out_axis1, ref_out_axis1, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis1, x_axis1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  EXPECT_TRUE(allclose(x_axis1, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::irfft(execution_space(), outr_axis1, xr_axis1,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  EXPECT_TRUE(allclose(xr_axis1, ref_xr, 1.e-5, atol));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  Kokkos::fence();

  // Along axis 0 (transpose neeed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1, i2);
      auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1, i2);
      fft1(sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis0,
                 KokkosFFT::Normalization::backward, /*axis=*/0);
  EXPECT_TRUE(allclose(out_axis0, ref_out_axis0, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis0, x_axis0,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  EXPECT_TRUE(allclose(x_axis0, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis0,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  KokkosFFT::irfft(execution_space(), outr_axis0, xr_axis0,
                   KokkosFFT::Normalization::backward, /*axis=*/0);

  EXPECT_TRUE(allclose(xr_axis0, ref_xr, 1.e-5, atol));

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1 (transpose neeed)
  // Perform batched 1D (along 1st axis) FFT sequentially
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i0 = 0; i0 < n0; i0++) {
      auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL, i2);
      auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL, i2);
      fft1(sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis1,
                 KokkosFFT::Normalization::backward, /*axis=*/1);
  EXPECT_TRUE(allclose(out_axis1, ref_out_axis1, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis1, x_axis1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  EXPECT_TRUE(allclose(x_axis1, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::irfft(execution_space(), outr_axis1, xr_axis1,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  EXPECT_TRUE(allclose(xr_axis1, ref_xr, 1.e-5, atol));

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 2
  // Perform batched 1D (along 2nd axis) FFT sequentially
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      auto sub_x   = Kokkos::subview(x, i0, i1, Kokkos::ALL);
      auto sub_ref = Kokkos::subview(ref_out_axis2, i0, i1, Kokkos::ALL);
      fft1(sub_x, sub_ref);
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis2,
                 KokkosFFT::Normalization::backward, /*axis=*/2);
  EXPECT_TRUE(allclose(out_axis2, ref_out_axis2, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis2, x_axis2,
                  KokkosFFT::Normalization::backward, /*axis=*/2);
  EXPECT_TRUE(allclose(x_axis2, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis2,
                  KokkosFFT::Normalization::backward, /*axis=*/2);
  KokkosFFT::irfft(execution_space(), outr_axis2, xr_axis2,
                   KokkosFFT::Normalization::backward, /*axis=*/2);

  EXPECT_TRUE(allclose(xr_axis2, ref_xr, 1.e-5, atol));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::fill_random(xr, random_pool, 1);

  // Since HIP FFT destructs the input data, we need to keep the input data in
  // different place
  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_xr, xr);

  Kokkos::fence();

  // Along axis 0 (transpose neeed)
  // Perform batched 1D (along 0th axis) FFT sequentially
  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        auto sub_x   = Kokkos::subview(x, Kokkos::ALL, i1, i2, i3);
        auto sub_ref = Kokkos::subview(ref_out_axis0, Kokkos::ALL, i1, i2, i3);
        fft1(sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis0,
                 KokkosFFT::Normalization::backward, /*axis=*/0);
  EXPECT_TRUE(allclose(out_axis0, ref_out_axis0, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis0, x_axis0,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  EXPECT_TRUE(allclose(x_axis0, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis0,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  KokkosFFT::irfft(execution_space(), outr_axis0, xr_axis0,
                   KokkosFFT::Normalization::backward, /*axis=*/0);

  EXPECT_TRUE(allclose(xr_axis0, ref_xr, 1.e-5, atol));

  // Recover input from reference
  Kokkos::deep_copy(x, ref_x);
  Kokkos::deep_copy(xr, ref_xr);

  // Along axis 1 (transpose neeed)
  // Perform batched 1D (along 1st axis) FFT sequentially
  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i0 = 0; i0 < n0; i0++) {
        auto sub_x   = Kokkos::subview(x, i0, Kokkos::ALL, i2, i3);
        auto sub_ref = Kokkos::subview(ref_out_axis1, i0, Kokkos::ALL, i2, i3);
        fft1(sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis1,
                 KokkosFFT::Normalization::backward, /*axis=*/1);
  EXPECT_TRUE(allclose(out_axis1, ref_out_axis1, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis1, x_axis1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  EXPECT_TRUE(allclose(x_axis1, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::irfft(execution_space(), outr_axis1, xr_axis1,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  EXPECT_TRUE(allclose(xr_axis1, ref_xr, 1.e-5, atol));

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
        fft1(sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis2,
                 KokkosFFT::Normalization::backward, /*axis=*/2);
  EXPECT_TRUE(allclose(out_axis2, ref_out_axis2, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis2, x_axis2,
                  KokkosFFT::Normalization::backward, /*axis=*/2);
  EXPECT_TRUE(allclose(x_axis2, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis2,
                  KokkosFFT::Normalization::backward, /*axis=*/2);
  KokkosFFT::irfft(execution_space(), outr_axis2, xr_axis2,
                   KokkosFFT::Normalization::backward, /*axis=*/2);

  EXPECT_TRUE(allclose(xr_axis2, ref_xr, 1.e-5, atol));

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
        fft1(sub_x, sub_ref);
      }
    }
  }

  KokkosFFT::fft(execution_space(), x, out_axis3,
                 KokkosFFT::Normalization::backward, /*axis=*/3);
  EXPECT_TRUE(allclose(out_axis3, ref_out_axis3, 1.e-5, atol));

  KokkosFFT::ifft(execution_space(), out_axis3, x_axis3,
                  KokkosFFT::Normalization::backward, /*axis=*/3);
  EXPECT_TRUE(allclose(x_axis3, ref_x, 1.e-5, atol));

  // Simple identity tests for r2c and c2r transforms
  KokkosFFT::rfft(execution_space(), xr, outr_axis3,
                  KokkosFFT::Normalization::backward, /*axis=*/3);
  KokkosFFT::irfft(execution_space(), outr_axis3, xr_axis3,
                   KokkosFFT::Normalization::backward, /*axis=*/3);

  EXPECT_TRUE(allclose(xr_axis3, ref_xr, 1.e-5, atol));
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_5dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4;
  using RealView5DType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ComplexView5DType =
      Kokkos::View<Kokkos::complex<T>*****, LayoutType, execution_space>;

  constexpr int DIM          = 5;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4};
  ComplexView5DType x("x", n0, n1, n2, n3, n4),
      ref_x("ref_x", n0, n1, n2, n3, n4);

  for (int axis = 0; axis < DIM; axis++) {
    std::array<int, DIM> shape_c2r = shape;
    shape_c2r.at(axis)             = shape_c2r.at(axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4] = shape_c2r;
    ComplexView5DType _x("_x", n0, n1, n2, n3, n4),
        out("out", n0, n1, n2, n3, n4), ref_out("ref_out", n0, n1, n2, n3, n4);
    RealView5DType xr("xr", n0, n1, n2, n3, n4),
        ref_xr("ref_xr", n0, n1, n2, n3, n4), _xr("_xr", n0, n1, n2, n3, n4);
    ComplexView5DType outr("outr", _n0, _n1, _n2, _n3, _n4);

    const Kokkos::complex<T> I(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, I);
    Kokkos::fill_random(xr, random_pool, 1);

    // Since HIP FFT destructs the input data, we need to keep the input data in
    // different place
    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    Kokkos::fence();

    // Along one axis
    // Simple identity tests
    KokkosFFT::fft(execution_space(), x, out,
                   KokkosFFT::Normalization::backward, axis);
    KokkosFFT::ifft(execution_space(), out, _x,
                    KokkosFFT::Normalization::backward, axis);
    EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfft(execution_space(), xr, outr,
                    KokkosFFT::Normalization::backward, axis);
    KokkosFFT::irfft(execution_space(), outr, _xr,
                     KokkosFFT::Normalization::backward, axis);

    EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_6dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3;
  using RealView6DType = Kokkos::View<T******, LayoutType, execution_space>;
  using ComplexView6DType =
      Kokkos::View<Kokkos::complex<T>******, LayoutType, execution_space>;

  constexpr int DIM          = 6;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5};
  ComplexView6DType x("x", n0, n1, n2, n3, n4, n5),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5);

  for (int axis = 0; axis < DIM; axis++) {
    std::array<int, DIM> shape_c2r      = shape;
    shape_c2r.at(axis)                  = shape_c2r.at(axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4, _n5] = shape_c2r;
    ComplexView6DType _x("_x", n0, n1, n2, n3, n4, n5),
        out("out", n0, n1, n2, n3, n4, n5),
        ref_out("ref_out", n0, n1, n2, n3, n4, n5);
    RealView6DType xr("xr", n0, n1, n2, n3, n4, n5),
        ref_xr("ref_xr", n0, n1, n2, n3, n4, n5),
        _xr("_xr", n0, n1, n2, n3, n4, n5);
    ComplexView6DType outr("outr", _n0, _n1, _n2, _n3, _n4, _n5);

    const Kokkos::complex<T> I(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, I);
    Kokkos::fill_random(xr, random_pool, 1);

    // Since HIP FFT destructs the input data, we need to keep the input data in
    // different place
    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    Kokkos::fence();

    // Along one axis
    // Simple identity tests
    KokkosFFT::fft(execution_space(), x, out,
                   KokkosFFT::Normalization::backward, axis);
    KokkosFFT::ifft(execution_space(), out, _x,
                    KokkosFFT::Normalization::backward, axis);
    EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfft(execution_space(), xr, outr,
                    KokkosFFT::Normalization::backward, axis);
    KokkosFFT::irfft(execution_space(), outr, _xr,
                     KokkosFFT::Normalization::backward, axis);

    EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_7dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4;
  using RealView7DType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ComplexView7DType =
      Kokkos::View<Kokkos::complex<T>*******, LayoutType, execution_space>;

  constexpr int DIM          = 7;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5, n6};
  ComplexView7DType x("x", n0, n1, n2, n3, n4, n5, n6),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5, n6);

  for (int axis = 0; axis < DIM; axis++) {
    std::array<int, DIM> shape_c2r           = shape;
    shape_c2r.at(axis)                       = shape_c2r.at(axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6] = shape_c2r;
    ComplexView7DType _x("_x", n0, n1, n2, n3, n4, n5, n6),
        out("out", n0, n1, n2, n3, n4, n5, n6),
        ref_out("ref_out", n0, n1, n2, n3, n4, n5, n6);
    RealView7DType xr("xr", n0, n1, n2, n3, n4, n5, n6),
        ref_xr("ref_xr", n0, n1, n2, n3, n4, n5, n6),
        _xr("_xr", n0, n1, n2, n3, n4, n5, n6);
    ComplexView7DType outr("outr", _n0, _n1, _n2, _n3, _n4, _n5, _n6);

    const Kokkos::complex<T> I(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, I);
    Kokkos::fill_random(xr, random_pool, 1);

    // Since HIP FFT destructs the input data, we need to keep the input data in
    // different place
    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    Kokkos::fence();

    // Along one axis
    // Simple identity tests
    KokkosFFT::fft(execution_space(), x, out,
                   KokkosFFT::Normalization::backward, axis);
    KokkosFFT::ifft(execution_space(), out, _x,
                    KokkosFFT::Normalization::backward, axis);
    EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfft(execution_space(), xr, outr,
                    KokkosFFT::Normalization::backward, axis);
    KokkosFFT::irfft(execution_space(), outr, _xr,
                     KokkosFFT::Normalization::backward, axis);

    EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
  }
}

template <typename T, typename LayoutType>
void test_fft1_1dfft_8dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4, n7 = 3;
  using RealView8DType = Kokkos::View<T********, LayoutType, execution_space>;
  using ComplexView8DType =
      Kokkos::View<Kokkos::complex<T>********, LayoutType, execution_space>;

  constexpr int DIM          = 8;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5, n6, n7};
  ComplexView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5, n6, n7);

  for (int axis = 0; axis < DIM; axis++) {
    std::array<int, DIM> shape_c2r                = shape;
    shape_c2r.at(axis)                            = shape_c2r.at(axis) / 2 + 1;
    auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7] = shape_c2r;
    ComplexView8DType _x("_x", n0, n1, n2, n3, n4, n5, n6, n7),
        out("out", n0, n1, n2, n3, n4, n5, n6, n7),
        ref_out("ref_out", n0, n1, n2, n3, n4, n5, n6, n7);
    RealView8DType xr("xr", n0, n1, n2, n3, n4, n5, n6, n7),
        ref_xr("ref_xr", n0, n1, n2, n3, n4, n5, n6, n7),
        _xr("_xr", n0, n1, n2, n3, n4, n5, n6, n7);
    ComplexView8DType outr("outr", _n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7);

    const Kokkos::complex<T> I(1.0, 1.0);
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, I);
    Kokkos::fill_random(xr, random_pool, 1);

    // Since HIP FFT destructs the input data, we need to keep the input data in
    // different place
    Kokkos::deep_copy(ref_x, x);
    Kokkos::deep_copy(ref_xr, xr);

    Kokkos::fence();

    // Along one axis
    // Simple identity tests
    KokkosFFT::fft(execution_space(), x, out,
                   KokkosFFT::Normalization::backward, axis);
    KokkosFFT::ifft(execution_space(), out, _x,
                    KokkosFFT::Normalization::backward, axis);
    EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

    // Simple identity tests for r2c and c2r transforms
    KokkosFFT::rfft(execution_space(), xr, outr,
                    KokkosFFT::Normalization::backward, axis);
    KokkosFFT::irfft(execution_space(), outr, _xr,
                     KokkosFFT::Normalization::backward, axis);

    EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
  }
}

// Identity tests on 1D Views
TYPED_TEST(FFT1D, Identity_1DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_identity<float_type, layout_type>(atol);
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

// batced fft1 on 2D Views
TYPED_TEST(FFT1D, FFT_batched_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_2dview<float_type, layout_type>(atol);
}

// batced fft1 on 3D Views
TYPED_TEST(FFT1D, FFT_batched_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_3dview<float_type, layout_type>(atol);
}

// batced fft1 on 4D Views
TYPED_TEST(FFT1D, FFT_batched_4DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_4dview<float_type, layout_type>(atol);
}

// batced fft1 on 5D Views
TYPED_TEST(FFT1D, FFT_batched_5DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_5dview<float_type, layout_type>(atol);
}

// batced fft1 on 6D Views
TYPED_TEST(FFT1D, FFT_batched_6DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_6dview<float_type, layout_type>(atol);
}

// batced fft1 on 7D Views
TYPED_TEST(FFT1D, FFT_batched_7DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_7dview<float_type, layout_type>(atol);
}

// batced fft1 on 8D Views
TYPED_TEST(FFT1D, FFT_batched_8DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft1_1dfft_8dview<float_type, layout_type>(atol);
}

// Tests for FFT2
template <typename T, typename LayoutType>
void test_fft2_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1),
      out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.fft2 is identical to np.fft(np.fft(x, axis=1), axis=0)
  KokkosFFT::fft(execution_space(), x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/0);

  KokkosFFT::fft2(execution_space(), x,
                  out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fft2(execution_space(), x, out_b,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fft2(execution_space(), x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::fft2(execution_space(), x, out_f,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Impl::Plan fft2_plan(execution_space(), x, out,
                                  KokkosFFT::Direction::forward, axes);
  KokkosFFT::fft2(execution_space(), x, out,
                  fft2_plan);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fft2(execution_space(), x, out_b, fft2_plan,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fft2(execution_space(), x, out_o, fft2_plan,
                  KokkosFFT::Normalization::ortho);
  KokkosFFT::fft2(execution_space(), x, out_f, fft2_plan,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // np.fft2(axes=(-1, -2)) is identical to np.fft(np.fft(x, axis=0), axis=1)
  axes_type axes10 = {-1, -2};

  KokkosFFT::fft(execution_space(), x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/0);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/1);

  KokkosFFT::fft2(execution_space(), x, out_b,
                  KokkosFFT::Normalization::backward, axes10);
  KokkosFFT::fft2(execution_space(), x, out_o, KokkosFFT::Normalization::ortho,
                  axes10);
  KokkosFFT::fft2(execution_space(), x, out_f,
                  KokkosFFT::Normalization::forward, axes10);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans np.fft2(axes=(-1, -2))
  KokkosFFT::Impl::Plan fft2_plan_axes10(execution_space(), x, out,
                                         KokkosFFT::Direction::forward, axes10);
  KokkosFFT::fft2(execution_space(), x, out_b, fft2_plan_axes10,
                  KokkosFFT::Normalization::backward, axes10);
  KokkosFFT::fft2(execution_space(), x, out_o, fft2_plan_axes10,
                  KokkosFFT::Normalization::ortho, axes10);
  KokkosFFT::fft2(execution_space(), x, out_f, fft2_plan_axes10,
                  KokkosFFT::Normalization::forward, axes10);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.ifft2 is identical to np.ifft(np.ifft(x, axis=1), axis=0)
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::ifft(execution_space(), out1, out2,
                  KokkosFFT::Normalization::backward, /*axis=*/0);

  KokkosFFT::ifft2(execution_space(), x,
                   out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifft2(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifft2(execution_space(), x, out_o,
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifft2(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Impl::Plan ifft2_plan(execution_space(), x, out,
                                   KokkosFFT::Direction::backward, axes);

  KokkosFFT::ifft2(execution_space(), x, out,
                   ifft2_plan);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifft2(execution_space(), x, out_b, ifft2_plan,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifft2(execution_space(), x, out_o, ifft2_plan,
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifft2(execution_space(), x, out_f, ifft2_plan,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // np.ifft2(axes=(-1, -2)) is identical to np.ifft(np.ifft(x, axis=0), axis=1)
  axes_type axes10 = {-1, -2};
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  KokkosFFT::ifft(execution_space(), out1, out2,
                  KokkosFFT::Normalization::backward, /*axis=*/1);

  KokkosFFT::ifft2(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward, axes10);
  KokkosFFT::ifft2(execution_space(), x, out_o, KokkosFFT::Normalization::ortho,
                   axes10);
  KokkosFFT::ifft2(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward, axes10);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  KokkosFFT::Impl::Plan ifft2_plan_axes10(
      execution_space(), x, out, KokkosFFT::Direction::backward, axes10);

  KokkosFFT::ifft2(execution_space(), x, out_b, ifft2_plan_axes10,
                   KokkosFFT::Normalization::backward, axes10);
  KokkosFFT::ifft2(execution_space(), x, out_o, ifft2_plan_axes10,
                   KokkosFFT::Normalization::ortho, axes10);
  KokkosFFT::ifft2(execution_space(), x, out_f, ifft2_plan_axes10,
                   KokkosFFT::Normalization::forward, axes10);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1);
  Kokkos::deep_copy(x_ref, x);
  Kokkos::fence();

  // np.rfft2 is identical to np.fft(np.rfft(x, axis=1), axis=0)
  KokkosFFT::rfft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x,
                   out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out_o,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Impl::Plan rfft2_plan(execution_space(), x, out,
                                   KokkosFFT::Direction::forward, axes);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out,
                   rfft2_plan);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out_b, rfft2_plan,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out_o, rfft2_plan,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfft2(execution_space(), x, out_f, rfft2_plan,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  // np.irfft2 is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, 0);
  KokkosFFT::irfft(execution_space(), out1, out2,
                   KokkosFFT::Normalization::backward, 1);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x,
                    out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x, out_b,
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x, out_o,
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x, out_f,
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};
  KokkosFFT::Impl::Plan irfft2_plan(execution_space(), x, out,
                                    KokkosFFT::Direction::backward, axes);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(
      execution_space(), x, out,
      irfft2_plan);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x, out_b, irfft2_plan,
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x, out_o, irfft2_plan,
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfft2(execution_space(), x, out_f, irfft2_plan,
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_3dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8;
  using RealView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using ComplexView3DType =
      Kokkos::View<Kokkos::complex<T>***, LayoutType, execution_space>;

  constexpr int DIM          = 3;
  std::array<int, DIM> shape = {n0, n1, n2};
  ComplexView3DType x("x", n0, n1, n2), ref_x("ref_x", n0, n1, n2);

  using axes_type = KokkosFFT::axis_type<2>;

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      std::array<int, DIM> shape_c2r = shape;
      shape_c2r.at(axis1)            = shape_c2r.at(axis1) / 2 + 1;

      auto [_n0, _n1, _n2] = shape_c2r;

      ComplexView3DType _x("_x", n0, n1, n2), out("out", n0, n1, n2),
          ref_out("ref_out", n0, n1, n2);
      RealView3DType xr("xr", n0, n1, n2), ref_xr("ref_xr", n0, n1, n2),
          _xr("_xr", n0, n1, n2);
      ComplexView3DType outr("outr", _n0, _n1, _n2);

      const Kokkos::complex<T> I(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<> random_pool(12345);
      Kokkos::fill_random(x, random_pool, I);
      Kokkos::fill_random(xr, random_pool, 1);

      Kokkos::deep_copy(ref_x, x);
      Kokkos::deep_copy(ref_xr, xr);

      Kokkos::fence();

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft2(execution_space(), x, out,
                      KokkosFFT::Normalization::backward, axes);

      KokkosFFT::ifft2(execution_space(), out, _x,
                       KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft2(execution_space(), xr, outr,
                       KokkosFFT::Normalization::backward, axes);

      KokkosFFT::irfft2(execution_space(), outr, _xr,
                        KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_4dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5;
  using RealView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using ComplexView4DType =
      Kokkos::View<Kokkos::complex<T>****, LayoutType, execution_space>;

  constexpr int DIM          = 4;
  std::array<int, DIM> shape = {n0, n1, n2, n3};
  ComplexView4DType x("x", n0, n1, n2, n3), ref_x("ref_x", n0, n1, n2, n3);

  using axes_type = KokkosFFT::axis_type<2>;

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      std::array<int, DIM> shape_c2r = shape;
      shape_c2r.at(axis1)            = shape_c2r.at(axis1) / 2 + 1;

      auto [_n0, _n1, _n2, _n3] = shape_c2r;
      ComplexView4DType _x("_x", n0, n1, n2, n3), out("out", n0, n1, n2, n3),
          ref_out("ref_out", n0, n1, n2, n3);
      RealView4DType xr("xr", n0, n1, n2, n3), ref_xr("ref_xr", n0, n1, n2, n3),
          _xr("_xr", n0, n1, n2, n3);
      ComplexView4DType outr("outr", _n0, _n1, _n2, _n3);

      const Kokkos::complex<T> I(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<> random_pool(12345);
      Kokkos::fill_random(x, random_pool, I);
      Kokkos::fill_random(xr, random_pool, 1);

      Kokkos::deep_copy(ref_x, x);
      Kokkos::deep_copy(ref_xr, xr);

      Kokkos::fence();

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft2(execution_space(), x, out,
                      KokkosFFT::Normalization::backward, axes);

      KokkosFFT::ifft2(execution_space(), out, _x,
                       KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft2(execution_space(), xr, outr,
                       KokkosFFT::Normalization::backward, axes);

      KokkosFFT::irfft2(execution_space(), outr, _xr,
                        KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_5dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4;
  using RealView5DType = Kokkos::View<T*****, LayoutType, execution_space>;
  using ComplexView5DType =
      Kokkos::View<Kokkos::complex<T>*****, LayoutType, execution_space>;

  constexpr int DIM          = 5;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4};
  ComplexView5DType x("x", n0, n1, n2, n3, n4),
      ref_x("ref_x", n0, n1, n2, n3, n4);

  using axes_type = KokkosFFT::axis_type<2>;

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      std::array<int, DIM> shape_c2r = shape;
      shape_c2r.at(axis1)            = shape_c2r.at(axis1) / 2 + 1;

      auto [_n0, _n1, _n2, _n3, _n4] = shape_c2r;
      ComplexView5DType _x("_x", n0, n1, n2, n3, n4),
          out("out", n0, n1, n2, n3, n4),
          ref_out("ref_out", n0, n1, n2, n3, n4);
      RealView5DType xr("xr", n0, n1, n2, n3, n4),
          ref_xr("ref_xr", n0, n1, n2, n3, n4), _xr("_xr", n0, n1, n2, n3, n4);
      ComplexView5DType outr("outr", _n0, _n1, _n2, _n3, _n4);

      const Kokkos::complex<T> I(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<> random_pool(12345);
      Kokkos::fill_random(x, random_pool, I);
      Kokkos::fill_random(xr, random_pool, 1);

      Kokkos::deep_copy(ref_x, x);
      Kokkos::deep_copy(ref_xr, xr);

      Kokkos::fence();

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft2(execution_space(), x, out,
                      KokkosFFT::Normalization::backward, axes);

      KokkosFFT::ifft2(execution_space(), out, _x,
                       KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft2(execution_space(), xr, outr,
                       KokkosFFT::Normalization::backward, axes);

      KokkosFFT::irfft2(execution_space(), outr, _xr,
                        KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_6dview(T atol = 1.e-12) {
  const int n0 = 10, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3;
  using RealView6DType = Kokkos::View<T******, LayoutType, execution_space>;
  using ComplexView6DType =
      Kokkos::View<Kokkos::complex<T>******, LayoutType, execution_space>;

  constexpr int DIM          = 6;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5};
  ComplexView6DType x("x", n0, n1, n2, n3, n4, n5),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5);

  using axes_type = KokkosFFT::axis_type<2>;

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      std::array<int, DIM> shape_c2r = shape;
      shape_c2r.at(axis1)            = shape_c2r.at(axis1) / 2 + 1;

      auto [_n0, _n1, _n2, _n3, _n4, _n5] = shape_c2r;
      ComplexView6DType _x("_x", n0, n1, n2, n3, n4, n5),
          out("out", n0, n1, n2, n3, n4, n5),
          ref_out("ref_out", n0, n1, n2, n3, n4, n5);
      RealView6DType xr("xr", n0, n1, n2, n3, n4, n5),
          ref_xr("ref_xr", n0, n1, n2, n3, n4, n5),
          _xr("_xr", n0, n1, n2, n3, n4, n5);
      ComplexView6DType outr("outr", _n0, _n1, _n2, _n3, _n4, _n5);

      const Kokkos::complex<T> I(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<> random_pool(12345);
      Kokkos::fill_random(x, random_pool, I);
      Kokkos::fill_random(xr, random_pool, 1);

      Kokkos::deep_copy(ref_x, x);
      Kokkos::deep_copy(ref_xr, xr);

      Kokkos::fence();

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft2(execution_space(), x, out,
                      KokkosFFT::Normalization::backward, axes);

      KokkosFFT::ifft2(execution_space(), out, _x,
                       KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft2(execution_space(), xr, outr,
                       KokkosFFT::Normalization::backward, axes);

      KokkosFFT::irfft2(execution_space(), outr, _xr,
                        KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_7dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4;
  using RealView7DType = Kokkos::View<T*******, LayoutType, execution_space>;
  using ComplexView7DType =
      Kokkos::View<Kokkos::complex<T>*******, LayoutType, execution_space>;

  constexpr int DIM          = 7;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5, n6};
  ComplexView7DType x("x", n0, n1, n2, n3, n4, n5, n6),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5, n6);

  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      std::array<int, DIM> shape_c2r = shape;
      shape_c2r.at(axis1)            = shape_c2r.at(axis1) / 2 + 1;

      auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6] = shape_c2r;
      ComplexView7DType _x("_x", n0, n1, n2, n3, n4, n5, n6),
          out("out", n0, n1, n2, n3, n4, n5, n6),
          ref_out("ref_out", n0, n1, n2, n3, n4, n5, n6);
      RealView7DType xr("xr", n0, n1, n2, n3, n4, n5, n6),
          ref_xr("ref_xr", n0, n1, n2, n3, n4, n5, n6),
          _xr("_xr", n0, n1, n2, n3, n4, n5, n6);
      ComplexView7DType outr("outr", _n0, _n1, _n2, _n3, _n4, _n5, _n6);

      const Kokkos::complex<T> I(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<> random_pool(12345);
      Kokkos::fill_random(x, random_pool, I);
      Kokkos::fill_random(xr, random_pool, 1);

      Kokkos::deep_copy(ref_x, x);
      Kokkos::deep_copy(ref_xr, xr);

      Kokkos::fence();

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft2(execution_space(), x, out,
                      KokkosFFT::Normalization::backward, axes);

      KokkosFFT::ifft2(execution_space(), out, _x,
                       KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft2(execution_space(), xr, outr,
                       KokkosFFT::Normalization::backward, axes);

      KokkosFFT::irfft2(execution_space(), outr, _xr,
                        KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
    }
  }
}

template <typename T, typename LayoutType>
void test_fft2_2dfft_8dview(T atol = 1.e-12) {
  const int n0 = 2, n1 = 6, n2 = 8, n3 = 5, n4 = 4, n5 = 3, n6 = 4, n7 = 3;
  using RealView8DType = Kokkos::View<T********, LayoutType, execution_space>;
  using ComplexView8DType =
      Kokkos::View<Kokkos::complex<T>********, LayoutType, execution_space>;

  constexpr int DIM          = 8;
  std::array<int, DIM> shape = {n0, n1, n2, n3, n4, n5, n6, n7};
  ComplexView8DType x("x", n0, n1, n2, n3, n4, n5, n6, n7),
      ref_x("ref_x", n0, n1, n2, n3, n4, n5, n6, n7);

  using axes_type = KokkosFFT::axis_type<2>;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      // make all the combinations of axes = {axis0, axis1}
      // axis0 and axis1 must be different
      if (axis0 == axis1) continue;

      axes_type axes = {axis0, axis1};

      std::array<int, DIM> shape_c2r = shape;
      shape_c2r.at(axis1)            = shape_c2r.at(axis1) / 2 + 1;

      auto [_n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7] = shape_c2r;
      ComplexView8DType _x("_x", n0, n1, n2, n3, n4, n5, n6, n7),
          out("out", n0, n1, n2, n3, n4, n5, n6, n7),
          ref_out("ref_out", n0, n1, n2, n3, n4, n5, n6, n7);
      RealView8DType xr("xr", n0, n1, n2, n3, n4, n5, n6, n7),
          ref_xr("ref_xr", n0, n1, n2, n3, n4, n5, n6, n7),
          _xr("_xr", n0, n1, n2, n3, n4, n5, n6, n7);
      ComplexView8DType outr("outr", _n0, _n1, _n2, _n3, _n4, _n5, _n6, _n7);

      const Kokkos::complex<T> I(1.0, 1.0);
      Kokkos::Random_XorShift64_Pool<> random_pool(12345);
      Kokkos::fill_random(x, random_pool, I);
      Kokkos::fill_random(xr, random_pool, 1);

      Kokkos::deep_copy(ref_x, x);
      Kokkos::deep_copy(ref_xr, xr);

      Kokkos::fence();

      // Along one axis
      // Simple identity tests
      KokkosFFT::fft2(execution_space(), x, out,
                      KokkosFFT::Normalization::backward, axes);

      KokkosFFT::ifft2(execution_space(), out, _x,
                       KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, atol));

      // Simple identity tests for r2c and c2r transforms
      KokkosFFT::rfft2(execution_space(), xr, outr,
                       KokkosFFT::Normalization::backward, axes);

      KokkosFFT::irfft2(execution_space(), outr, _xr,
                        KokkosFFT::Normalization::backward, axes);

      EXPECT_TRUE(allclose(_xr, ref_xr, 1.e-5, atol));
    }
  }
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

// batced fft2 on 3D Views
TYPED_TEST(FFT2D, FFT_batched_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_3dview<float_type, layout_type>(atol);
}

// batced fft2 on 4D Views
TYPED_TEST(FFT2D, FFT_batched_4DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_4dview<float_type, layout_type>(atol);
}

// batced fft2 on 5D Views
TYPED_TEST(FFT2D, FFT_batched_5DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_5dview<float_type, layout_type>(atol);
}

// batced fft2 on 6D Views
TYPED_TEST(FFT2D, FFT_batched_6DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_6dview<float_type, layout_type>(atol);
}

// batced fft2 on 7D Views
TYPED_TEST(FFT2D, FFT_batched_7DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_7dview<float_type, layout_type>(atol);
}

// batced fft2 on 8D Views
TYPED_TEST(FFT2D, FFT_batched_8DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 1.0e-6 : 1.0e-12;
  test_fft2_2dfft_8dview<float_type, layout_type>(atol);
}

// Tests for FFTN
template <typename T, typename LayoutType>
void test_fftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1),
      out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  KokkosFFT::fft(execution_space(), x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/1);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/0);

  KokkosFFT::fftn(execution_space(), x,
                  out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(execution_space(), x, out_b,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(execution_space(), x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(execution_space(), x, out_f,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  KokkosFFT::fftn(execution_space(), x, out,
                  axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(execution_space(), x, out_b, axes,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(execution_space(), x, out_o, axes,
                  KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(execution_space(), x, out_f, axes,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Impl::Plan fftn_plan(execution_space(), x, out,
                                  KokkosFFT::Direction::forward, axes);

  KokkosFFT::fftn(execution_space(), x, out, fftn_plan,
                  axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(execution_space(), x, out_b, fftn_plan, axes,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(execution_space(), x, out_o, fftn_plan, axes,
                  KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(execution_space(), x, out_f, fftn_plan, axes,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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
      out_f("out_f", n0, n1, n2);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.fftn for 3D array is identical to np.fft(np.fft(np.fft(x, axis=2),
  // axis=1), axis=0)
  KokkosFFT::fft(execution_space(), x, out1, KokkosFFT::Normalization::backward,
                 /*axis=*/2);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::fft(execution_space(), out2, out3,
                 KokkosFFT::Normalization::backward, /*axis=*/0);

  KokkosFFT::fftn(execution_space(), x,
                  out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(execution_space(), x, out_b,
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(execution_space(), x, out_o, KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(execution_space(), x, out_f,
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, atol));

  // Same tests with specifying axes
  // np.fftn for 3D array is identical to np.fft(np.fft(np.fft(x, axis=2),
  // axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;

  KokkosFFT::fftn(
      execution_space(), x, out,
      axes_type{-3, -2, -1});  // default: KokkosFFT::Normalization::backward
  KokkosFFT::fftn(execution_space(), x, out_b, axes_type{-3, -2, -1},
                  KokkosFFT::Normalization::backward);
  KokkosFFT::fftn(execution_space(), x, out_o, axes_type{-3, -2, -1},
                  KokkosFFT::Normalization::ortho);
  KokkosFFT::fftn(execution_space(), x, out_f, axes_type{-3, -2, -1},
                  KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, atol));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, atol));
}

template <typename T, typename LayoutType>
void test_ifftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  ComplexView2DType x("x", n0, n1);
  ComplexView2DType out("out", n0, n1), out1("out1", n0, n1),
      out2("out2", n0, n1);
  ComplexView2DType out_b("out_b", n0, n1), out_o("out_o", n0, n1),
      out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.ifftn for 2D array is identical to np.ifft(np.ifft(x, axis=1), axis=0)
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::ifft(execution_space(), out1, out2,
                  KokkosFFT::Normalization::backward, /*axis=*/0);

  KokkosFFT::ifftn(execution_space(), x,
                   out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(execution_space(), x, out_o,
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.fftn for 2D array is identical to np.fft(np.fft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  KokkosFFT::ifftn(execution_space(), x, out,
                   axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(execution_space(), x, out_b, axes,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(execution_space(), x, out_o, axes,
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(execution_space(), x, out_f, axes,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Impl::Plan ifftn_plan(execution_space(), x, out,
                                   KokkosFFT::Direction::backward, axes);
  KokkosFFT::ifftn(execution_space(), x, out, ifftn_plan,
                   axes);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(execution_space(), x, out_b, ifftn_plan, axes,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(execution_space(), x, out_o, ifftn_plan, axes,
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(execution_space(), x, out_f, ifftn_plan, axes,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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
      out_f("out_f", n0, n1, n2);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);

  Kokkos::fence();

  // np.ifftn for 3D array is identical to np.ifft(np.ifft(np.ifft(x, axis=2),
  // axis=1), axis=0)
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/2);
  KokkosFFT::ifft(execution_space(), out1, out2,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::ifft(execution_space(), out2, out3,
                  KokkosFFT::Normalization::backward, /*axis=*/0);

  KokkosFFT::ifftn(execution_space(), x,
                   out);  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(execution_space(), x, out_o,
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.ifftn for 3D array is identical to np.ifft(np.ifft(np.ifft(x, axis=2),
  // axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;

  KokkosFFT::ifftn(
      execution_space(), x, out,
      axes_type{-3, -2, -1});  // default: KokkosFFT::Normalization::backward
  KokkosFFT::ifftn(execution_space(), x, out_b, axes_type{-3, -2, -1},
                   KokkosFFT::Normalization::backward);
  KokkosFFT::ifftn(execution_space(), x, out_o, axes_type{-3, -2, -1},
                   KokkosFFT::Normalization::ortho);
  KokkosFFT::ifftn(execution_space(), x, out_f, axes_type{-3, -2, -1},
                   KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, 1.e-6));
}

template <typename T, typename LayoutType>
void test_rfftn_2dfft_2dview() {
  const int n0 = 4, n1 = 6;
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  RealView2DType x("x", n0, n1), x_ref("x_ref", n0, n1);
  ComplexView2DType out("out", n0, n1 / 2 + 1), out1("out1", n0, n1 / 2 + 1),
      out2("out2", n0, n1 / 2 + 1);
  ComplexView2DType out_b("out_b", n0, n1 / 2 + 1),
      out_o("out_o", n0, n1 / 2 + 1), out_f("out_f", n0, n1 / 2 + 1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1);
  Kokkos::deep_copy(x_ref, x);
  Kokkos::fence();

  // np.rfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  KokkosFFT::rfft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x,
                   out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_o,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.rfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out,
                   axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_b, axes,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_o, axes,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_f, axes,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Impl::Plan rfftn_plan(execution_space(), x, out,
                                   KokkosFFT::Direction::forward, axes);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out, rfftn_plan,
                   axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_b, rfftn_plan, axes,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_o, rfftn_plan, axes,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_f, rfftn_plan, axes,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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
      out_o("out_o", n0, n1, n2 / 2 + 1), out_f("out_f", n0, n1, n2 / 2 + 1);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1);
  Kokkos::deep_copy(x_ref, x);
  Kokkos::fence();

  // np.rfftn for 3D array is identical to np.fft(np.fft(np.rfft(x, axis=2),
  // axis=1), axis=0)
  KokkosFFT::rfft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/2);
  KokkosFFT::fft(execution_space(), out1, out2,
                 KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::fft(execution_space(), out2, out3,
                 KokkosFFT::Normalization::backward, /*axis=*/0);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x,
                   out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_b,
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_o,
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_f,
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.rfftn for 3D array is identical to np.fft(np.fft(np.rfft(x, axis=2),
  // axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<3>;

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(
      execution_space(), x, out,
      axes_type{-3, -2, -1});  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_b, axes_type{-3, -2, -1},
                   KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_o, axes_type{-3, -2, -1},
                   KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::rfftn(execution_space(), x, out_f, axes_type{-3, -2, -1},
                   KokkosFFT::Normalization::forward);

  multiply(out_o, sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, 1.e-6));
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
      out_f("out_f", n0, n1);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 2D array is identical to np.irfft(np.ifft(x, axis=0), axis=1)
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  KokkosFFT::irfft(execution_space(), out1, out2,
                   KokkosFFT::Normalization::backward, /*axis=*/1);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x,
                    out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_b,
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_o,
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_f,
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.irfftn for 2D array is identical to np.fft(np.rfft(x, axis=1), axis=0)
  using axes_type = KokkosFFT::axis_type<2>;
  axes_type axes  = {-2, -1};

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out,
                    axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_b, axes,
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_o, axes,
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_f, axes,
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));

  // Reuse plans
  KokkosFFT::Impl::Plan irfftn_plan(execution_space(), x, out,
                                    KokkosFFT::Direction::backward, axes);
  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out, irfftn_plan,
                    axes);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_b, irfftn_plan, axes,
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_o, irfftn_plan, axes,
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_f, irfftn_plan, axes,
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1));

  EXPECT_TRUE(allclose(out, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out2, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out2, 1.e-5, 1.e-6));
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
      out_f("out_f", n0, n1, n2);

  const Kokkos::complex<T> I(1.0, 1.0);
  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, I);
  Kokkos::deep_copy(x_ref, x);

  // np.irfftn for 3D array is identical to np.irfft(np.ifft(np.ifft(x, axis=0),
  // axis=1), axis=2)
  KokkosFFT::ifft(execution_space(), x, out1,
                  KokkosFFT::Normalization::backward, /*axis=*/0);
  KokkosFFT::ifft(execution_space(), out1, out2,
                  KokkosFFT::Normalization::backward, /*axis=*/1);
  KokkosFFT::irfft(execution_space(), out2, out3,
                   KokkosFFT::Normalization::backward, /*axis=*/2);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x,
                    out);  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_b,
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_o,
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_f,
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, 1.e-6));

  // Same tests with specifying axes
  // np.irfftn for 3D array is identical to np.irfft(np.ifft(np.ifft(x, axis=0),
  // axis=1), axis=2)
  using axes_type = KokkosFFT::axis_type<3>;

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(
      execution_space(), x, out,
      axes_type{-3, -2, -1});  // default: KokkosFFT::Normalization::backward

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_b, axes_type{-3, -2, -1},
                    KokkosFFT::Normalization::backward);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_o, axes_type{-3, -2, -1},
                    KokkosFFT::Normalization::ortho);

  Kokkos::deep_copy(x, x_ref);
  KokkosFFT::irfftn(execution_space(), x, out_f, axes_type{-3, -2, -1},
                    KokkosFFT::Normalization::forward);

  multiply(out_o, 1.0 / sqrt(static_cast<T>(n0 * n1 * n2)));
  multiply(out_f, 1.0 / static_cast<T>(n0 * n1 * n2));

  EXPECT_TRUE(allclose(out, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_b, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_o, out3, 1.e-5, 1.e-6));
  EXPECT_TRUE(allclose(out_f, out3, 1.e-5, 1.e-6));
}

// fftn on 2D Views
TYPED_TEST(FFTND, 2DFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_fftn_2dfft_2dview<float_type, layout_type>();
}

// fftn on 3D Views
TYPED_TEST(FFTND, 3DFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  float_type atol = std::is_same_v<float_type, float> ? 5.0e-5 : 1.0e-6;
  test_fftn_3dfft_3dview<float_type, layout_type>(atol);
}

// ifftn on 2D Views
TYPED_TEST(FFTND, 2DIFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_ifftn_2dfft_2dview<float_type, layout_type>();
}

// ifftn on 3D Views
TYPED_TEST(FFTND, 3DIFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_ifftn_3dfft_3dview<float_type, layout_type>();
}

// rfftn on 2D Views
TYPED_TEST(FFTND, 2DRFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_rfftn_2dfft_2dview<float_type, layout_type>();
}

// rfftn on 3D Views
TYPED_TEST(FFTND, 3DRFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_rfftn_3dfft_3dview<float_type, layout_type>();
}

// irfftn on 2D Views
TYPED_TEST(FFTND, 2DIRFFT_2DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_irfftn_2dfft_2dview<float_type, layout_type>();
}

// irfftn on 3D Views
TYPED_TEST(FFTND, 3DIRFFT_3DView) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  test_irfftn_3dfft_3dview<float_type, layout_type>();
}