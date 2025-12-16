// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_Layout.hpp"
#include "KokkosFFT_Normalization.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using real_and_int_types =
    ::testing::Types<std::pair<float, int>, std::pair<float, std::size_t>,
                     std::pair<double, int>, std::pair<double, std::size_t> >;
using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight> >;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestNormalizationCoefficient : public ::testing::Test {
  using float_type = typename T::first_type;
  using index_type = typename T::second_type;
};

template <typename T>
struct TestGetCoefficient : public ::testing::Test {
  using float_type = typename T::first_type;
  using index_type = typename T::second_type;
};

template <typename T>
struct TestNormalization : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

struct ParamTestsNormalization
    : public ::testing::TestWithParam<KokkosFFT::Normalization> {};

template <typename RealType, typename IndexType, std::size_t DIM>
auto get_extents_and_coef(bool sqrt = false) {
  std::array<IndexType, DIM> extents{};
  for (std::size_t i = 0; i < extents.size(); i++) {
    extents.at(i) = 100;
  }

  RealType coef = 1;
  for (auto extent : extents) {
    // sqrt(100) or 100
    auto temp = sqrt ? extent / 10 : extent;
    coef *= (RealType{1} / static_cast<RealType>(temp));
  }
  return std::make_pair(extents, coef);
}

/// \brief A test function to check the normalization_coefficient
/// \tparam RealType
/// \tparam IndexType
/// \tparam DIM
/// \param[in] norm Normalization type
template <typename RealType, typename IndexType, std::size_t DIM>
void test_one_over_N() {
  auto [extents, ref_coef] = get_extents_and_coef<RealType, IndexType, DIM>();
  auto vec_extents         = KokkosFFT::Impl::to_vector(extents);
  auto one_over_N          = KokkosFFT::Impl::one_over_N<RealType>(extents);
  auto one_over_N_vec      = KokkosFFT::Impl::one_over_N<RealType>(vec_extents);

  RealType epsilon = std::numeric_limits<RealType>::epsilon() * 10;
  EXPECT_NEAR(one_over_N, ref_coef, epsilon);
  EXPECT_NEAR(one_over_N_vec, ref_coef, epsilon);
}

/// \brief A test function to check the normalization_coefficient
/// \tparam RealType
/// \tparam IndexType
/// \tparam DIM
/// \param[in] norm Normalization type
template <typename RealType, typename IndexType, std::size_t DIM>
void test_one_over_sqrt_N() {
  auto [extents, ref_coef] =
      get_extents_and_coef<RealType, IndexType, DIM>(true);
  auto vec_extents     = KokkosFFT::Impl::to_vector(extents);
  auto one_over_sqrt_N = KokkosFFT::Impl::one_over_sqrt_N<RealType>(extents);
  auto one_over_sqrt_N_vec =
      KokkosFFT::Impl::one_over_sqrt_N<RealType>(vec_extents);

  RealType epsilon = std::numeric_limits<RealType>::epsilon() * 10;
  EXPECT_NEAR(one_over_sqrt_N, ref_coef, epsilon);
  EXPECT_NEAR(one_over_sqrt_N_vec, ref_coef, epsilon);
}

/// \brief A test function to check the normalization_coefficient
/// \tparam RealType
/// \tparam IndexType
/// \tparam DIM
/// \param[in] norm Normalization type
template <typename RealType, typename IndexType, std::size_t DIM>
void test_get_coefficients(KokkosFFT::Normalization norm) {
  auto [extents, one_over_N] = get_extents_and_coef<RealType, IndexType, DIM>();
  auto [coef_f, to_normalize_f] = KokkosFFT::Impl::get_coefficients<RealType>(
      KokkosFFT::Direction::forward, norm, extents);
  auto [coef_b, to_normalize_b] = KokkosFFT::Impl::get_coefficients<RealType>(
      KokkosFFT::Direction::backward, norm, extents);
  auto vec_extents = KokkosFFT::Impl::to_vector(extents);
  auto [coef_f_vec, to_normalize_f_vec] =
      KokkosFFT::Impl::get_coefficients<RealType>(KokkosFFT::Direction::forward,
                                                  norm, vec_extents);
  auto [coef_b_vec, to_normalize_b_vec] =
      KokkosFFT::Impl::get_coefficients<RealType>(
          KokkosFFT::Direction::backward, norm, vec_extents);

  RealType epsilon = std::numeric_limits<RealType>::epsilon() * 10;
  RealType one     = 1;
  if (norm == KokkosFFT::Normalization::forward) {
    EXPECT_NEAR(coef_f, one_over_N, epsilon);
    EXPECT_TRUE(to_normalize_f);

    EXPECT_NEAR(coef_b, one, epsilon);
    EXPECT_FALSE(to_normalize_b);

    EXPECT_NEAR(coef_f_vec, one_over_N, epsilon);
    EXPECT_TRUE(to_normalize_f_vec);

    EXPECT_NEAR(coef_b_vec, one, epsilon);
    EXPECT_FALSE(to_normalize_b_vec);
  } else if (norm == KokkosFFT::Normalization::backward) {
    EXPECT_NEAR(coef_f, one, epsilon);
    EXPECT_FALSE(to_normalize_f);

    EXPECT_NEAR(coef_b, one_over_N, epsilon);
    EXPECT_TRUE(to_normalize_b);

    EXPECT_NEAR(coef_f_vec, one, epsilon);
    EXPECT_FALSE(to_normalize_f_vec);

    EXPECT_NEAR(coef_b_vec, one_over_N, epsilon);
    EXPECT_TRUE(to_normalize_b_vec);
  } else if (norm == KokkosFFT::Normalization::ortho) {
    EXPECT_NEAR(coef_f, Kokkos::sqrt(one_over_N), epsilon);
    EXPECT_TRUE(to_normalize_f);

    EXPECT_NEAR(coef_b, Kokkos::sqrt(one_over_N), epsilon);
    EXPECT_TRUE(to_normalize_b);

    EXPECT_NEAR(coef_f_vec, Kokkos::sqrt(one_over_N), epsilon);
    EXPECT_TRUE(to_normalize_f_vec);

    EXPECT_NEAR(coef_b_vec, Kokkos::sqrt(one_over_N), epsilon);
    EXPECT_TRUE(to_normalize_b_vec);
  } else {
    // norm == Normalization::none
    EXPECT_NEAR(coef_f, one, epsilon);
    EXPECT_FALSE(to_normalize_f);

    EXPECT_NEAR(coef_b, one, epsilon);
    EXPECT_FALSE(to_normalize_b);

    EXPECT_NEAR(coef_f_vec, one, epsilon);
    EXPECT_FALSE(to_normalize_f_vec);

    EXPECT_NEAR(coef_b_vec, one, epsilon);
    EXPECT_FALSE(to_normalize_b_vec);
  }
}

/// \brief A test function to check the
/// \tparam T
/// \tparam LayoutType
/// \tparam DIM
/// \param[in] norm Normalization type
template <typename T, typename LayoutType, std::size_t DIM>
void test_normalization(KokkosFFT::Normalization norm) {
  using view_data_type = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using ViewType = Kokkos::View<view_data_type, LayoutType, execution_space>;

  [[maybe_unused]] auto [extents, coef] =
      get_extents_and_coef<T, std::size_t, DIM>();
  auto layout = KokkosFFT::Impl::create_layout<LayoutType>(extents);
  ViewType x("x", layout), ref_x("ref_x", layout), ref_f("ref_f", layout),
      ref_b("ref_b", layout);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(/*seed=*/12345);
  Kokkos::fill_random(exec, x, random_pool, 1.0);
  exec.fence();

  Kokkos::deep_copy(ref_x, x);
  Kokkos::deep_copy(ref_f, x);
  Kokkos::deep_copy(ref_b, x);

  [[maybe_unused]] auto [coef_f, to_normalize_f] =
      KokkosFFT::Impl::get_coefficients<T>(KokkosFFT::Direction::forward, norm,
                                           extents);
  [[maybe_unused]] auto [coef_b, to_normalize_b] =
      KokkosFFT::Impl::get_coefficients<T>(KokkosFFT::Direction::backward, norm,
                                           extents);

  multiply(exec, ref_f, coef_f);
  multiply(exec, ref_b, coef_b);
  exec.fence();

  // Backward FFT with forward Normalization -> Do nothing
  KokkosFFT::Impl::normalize<T>(exec, x, KokkosFFT::Direction::backward, norm,
                                extents);
  EXPECT_TRUE(allclose(exec, x, ref_b, 1.e-5, 1.e-12));
  exec.fence();
  Kokkos::deep_copy(x, ref_x);

  // Forward FFT with forward Normalization -> 1/N normalization
  KokkosFFT::Impl::normalize<T>(exec, x, KokkosFFT::Direction::forward, norm,
                                extents);
  EXPECT_TRUE(allclose(exec, x, ref_f, 1.e-5, 1.e-12));
  exec.fence();
  Kokkos::deep_copy(x, ref_x);

  auto extents_vec = KokkosFFT::Impl::to_vector(extents);
  KokkosFFT::Impl::normalize<T>(exec, x, KokkosFFT::Direction::backward, norm,
                                extents_vec);
  EXPECT_TRUE(allclose(exec, x, ref_b, 1.e-5, 1.e-12));
  exec.fence();
  Kokkos::deep_copy(x, ref_x);

  // Forward FFT with forward Normalization -> 1/N normalization
  KokkosFFT::Impl::normalize<T>(exec, x, KokkosFFT::Direction::forward, norm,
                                extents_vec);
  EXPECT_TRUE(allclose(exec, x, ref_f, 1.e-5, 1.e-12));
  exec.fence();
}

void test_swap_direction(KokkosFFT::Normalization norm) {
  auto new_direction = KokkosFFT::Impl::swap_direction(norm);
  if (norm == KokkosFFT::Normalization::forward) {
    EXPECT_EQ(new_direction, KokkosFFT::Normalization::backward);
  } else if (norm == KokkosFFT::Normalization::backward) {
    EXPECT_EQ(new_direction, KokkosFFT::Normalization::forward);
  } else if (norm == KokkosFFT::Normalization::ortho) {
    EXPECT_EQ(new_direction, KokkosFFT::Normalization::ortho);
  } else {
    // KokkosFFT::Normalization::none
    EXPECT_EQ(new_direction, KokkosFFT::Normalization::none);
  }
}

}  // namespace

TYPED_TEST_SUITE(TestNormalizationCoefficient, real_and_int_types);
TYPED_TEST_SUITE(TestGetCoefficient, real_and_int_types);
TYPED_TEST_SUITE(TestNormalization, test_types);

// Tests for normalization coefficient
TYPED_TEST(TestNormalizationCoefficient, OneOverN) {
  using float_type = typename TestFixture::float_type;
  using index_type = typename TestFixture::index_type;
  test_one_over_N<float_type, index_type, 1>();
  test_one_over_N<float_type, index_type, 2>();
  test_one_over_N<float_type, index_type, 3>();
}

TYPED_TEST(TestNormalizationCoefficient, OneOverSqrtN) {
  using float_type = typename TestFixture::float_type;
  using index_type = typename TestFixture::index_type;
  test_one_over_sqrt_N<float_type, index_type, 1>();
  test_one_over_sqrt_N<float_type, index_type, 2>();
  test_one_over_sqrt_N<float_type, index_type, 3>();
}

TYPED_TEST(TestGetCoefficient, Forward) {
  using float_type = typename TestFixture::float_type;
  using index_type = typename TestFixture::index_type;
  auto norm        = KokkosFFT::Normalization::forward;
  test_get_coefficients<float_type, index_type, 1>(norm);
  test_get_coefficients<float_type, index_type, 2>(norm);
  test_get_coefficients<float_type, index_type, 3>(norm);
}

TYPED_TEST(TestGetCoefficient, Backward) {
  using float_type = typename TestFixture::float_type;
  using index_type = typename TestFixture::index_type;
  auto norm        = KokkosFFT::Normalization::backward;
  test_get_coefficients<float_type, index_type, 1>(norm);
  test_get_coefficients<float_type, index_type, 2>(norm);
  test_get_coefficients<float_type, index_type, 3>(norm);
}

TYPED_TEST(TestGetCoefficient, Ortho) {
  using float_type = typename TestFixture::float_type;
  using index_type = typename TestFixture::index_type;
  auto norm        = KokkosFFT::Normalization::ortho;
  test_get_coefficients<float_type, index_type, 1>(norm);
  test_get_coefficients<float_type, index_type, 2>(norm);
  test_get_coefficients<float_type, index_type, 3>(norm);
}

TYPED_TEST(TestGetCoefficient, None) {
  using float_type = typename TestFixture::float_type;
  using index_type = typename TestFixture::index_type;
  auto norm        = KokkosFFT::Normalization::none;
  test_get_coefficients<float_type, index_type, 1>(norm);
  test_get_coefficients<float_type, index_type, 2>(norm);
  test_get_coefficients<float_type, index_type, 3>(norm);
}

TYPED_TEST(TestNormalization, Forward) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  auto norm         = KokkosFFT::Normalization::forward;
  test_normalization<float_type, layout_type, 1>(norm);
  test_normalization<float_type, layout_type, 2>(norm);
  test_normalization<float_type, layout_type, 3>(norm);
}

TYPED_TEST(TestNormalization, Backward) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  auto norm         = KokkosFFT::Normalization::backward;
  test_normalization<float_type, layout_type, 1>(norm);
  test_normalization<float_type, layout_type, 2>(norm);
  test_normalization<float_type, layout_type, 3>(norm);
}

TYPED_TEST(TestNormalization, Ortho) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  auto norm         = KokkosFFT::Normalization::ortho;
  test_normalization<float_type, layout_type, 1>(norm);
  test_normalization<float_type, layout_type, 2>(norm);
  test_normalization<float_type, layout_type, 3>(norm);
}

TYPED_TEST(TestNormalization, None) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  auto norm         = KokkosFFT::Normalization::none;
  test_normalization<float_type, layout_type, 1>(norm);
  test_normalization<float_type, layout_type, 2>(norm);
  test_normalization<float_type, layout_type, 3>(norm);
}

// Parameterized tests
TEST_P(ParamTestsNormalization, SwapDirection) {
  KokkosFFT::Normalization norm = GetParam();
  test_swap_direction(norm);
}

INSTANTIATE_TEST_SUITE_P(FFTShift, ParamTestsNormalization,
                         ::testing::Values(KokkosFFT::Normalization::forward,
                                           KokkosFFT::Normalization::backward,
                                           KokkosFFT::Normalization::ortho,
                                           KokkosFFT::Normalization::none));
