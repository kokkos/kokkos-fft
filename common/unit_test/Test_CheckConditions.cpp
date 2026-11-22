// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "KokkosFFT_CheckConditions.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Int like types
using int_types = ::testing::Types<int, std::size_t>;

// value type combinations that are tested
using paired_scalar_types = ::testing::Types<
    std::pair<float, float>, std::pair<float, Kokkos::complex<float>>,
    std::pair<Kokkos::complex<float>, Kokkos::complex<float>>,
    std::pair<double, double>, std::pair<double, Kokkos::complex<double>>,
    std::pair<Kokkos::complex<double>, Kokkos::complex<double>>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestPointerTypes : public ::testing::Test {
  using value_type1 = typename T::first_type;
  using value_type2 = typename T::second_type;
};

template <typename T>
struct TestContainerTypes : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestAxes : public ::testing::Test {
  using value_type = T;
};

template <typename ValueType1, typename ValueType2>
void test_are_pointers_aliasing() {
  using View1 = Kokkos::View<ValueType1*, execution_space>;
  using View2 = Kokkos::View<ValueType2*, execution_space>;

  const int n1 = 10;
  // sizeof ValueType2 is larger or equal to ValueType1
  const int n2 = sizeof(ValueType1) == sizeof(ValueType2) ? n1 : n1 / 2 + 1;
  View1 view1("view1", n1);
  View2 view2("view2", n1);
  View2 uview2(reinterpret_cast<ValueType2*>(view1.data()), n2);

  EXPECT_TRUE(KokkosFFT::Impl::are_aliasing(view1.data(), uview2.data()));
  EXPECT_FALSE(KokkosFFT::Impl::are_aliasing(view1.data(), view2.data()));
}

template <typename ValueType1, typename ValueType2>
void test_are_overlapping() {
  using View1 = Kokkos::View<ValueType1*, execution_space>;
  using View2 = Kokkos::View<ValueType2*, execution_space>;
  using LeftView1 =
      Kokkos::View<ValueType1**, Kokkos::LayoutLeft, execution_space>;

  // sizeof ValueType2 is a multiple of sizeof ValueType1
  constexpr std::size_t ratio = sizeof(ValueType2) / sizeof(ValueType1);
  View1 buffer("buffer", 16), other("other", 16);
  ValueType1* ptr = buffer.data();

  View1 a(ptr, 8);               // the first half of buffer
  View1 b_partial(ptr + 4, 8);   // starts inside a
  View1 c_adjacent(ptr + 8, 8);  // starts right after a
  View2 a_same(reinterpret_cast<ValueType2*>(ptr), 1);  // same start
  View2 c_adjacent2(reinterpret_cast<ValueType2*>(ptr + 8),
                    8 / ratio);  // the bytes of c_adjacent
  View1 empty(ptr, 0);

  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(a, a));
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(a, a_same));
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(a_same, a));
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(a, b_partial));
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(b_partial, a));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(a, c_adjacent));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(c_adjacent, a));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(a, c_adjacent2));
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(b_partial, c_adjacent2));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(a, empty));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(empty, a));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(empty, empty));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(buffer, other));

  // Strided Views: the span counts, not the size. In a 4 x 4 LayoutLeft
  // View, element (i, j) is at i + 4 j: a row spans [i, i + 13), a column
  // [4 j, 4 j + 4)
  LeftView1 matrix("matrix", 4, 4);
  auto row0 = Kokkos::subview(matrix, 0, Kokkos::ALL);
  auto row3 = Kokkos::subview(matrix, 3, Kokkos::ALL);
  auto col0 = Kokkos::subview(matrix, Kokkos::ALL, 0);
  auto col3 = Kokkos::subview(matrix, Kokkos::ALL, 3);
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(row0, col0));
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(row0, col3));
  // Interleaved rows share no element, but their spans intersect
  EXPECT_TRUE(KokkosFFT::Impl::are_overlapping(row0, row3));
  EXPECT_FALSE(KokkosFFT::Impl::are_overlapping(col0, col3));
}

template <typename ValueType1, typename ValueType2>
void test_check_inplace() {
  using View1 = Kokkos::View<ValueType1*, execution_space>;
  using View2 = Kokkos::View<ValueType2*, execution_space>;

  // sizeof ValueType2 is a multiple of sizeof ValueType1
  constexpr std::size_t ratio = sizeof(ValueType2) / sizeof(ValueType1);
  View1 buffer("buffer", 16);
  View2 other("other", 8);
  ValueType1* ptr = buffer.data();

  View1 in(ptr, 8);
  View2 out_same(reinterpret_cast<ValueType2*>(ptr),
                 8 / ratio);  // the same data, possibly another value type
  View2 out_partial(reinterpret_cast<ValueType2*>(ptr + ratio),
                    8 / ratio);  // starts inside in
  View2 out_adjacent(reinterpret_cast<ValueType2*>(ptr + 8),
                     8 / ratio);  // starts right after in
  View1 empty_in;                 // no data

  // In-place: the same start address
  EXPECT_TRUE(KokkosFFT::Impl::check_inplace(in, out_same));
  EXPECT_TRUE(KokkosFFT::Impl::check_inplace(buffer, buffer));

  // Out-of-place: no overlap
  EXPECT_FALSE(KokkosFFT::Impl::check_inplace(in, other));
  EXPECT_FALSE(KokkosFFT::Impl::check_inplace(in, out_adjacent));
  EXPECT_FALSE(KokkosFFT::Impl::check_inplace(empty_in, other));

  // Partial overlap: neither in-place nor out-of-place
  EXPECT_THROW(KokkosFFT::Impl::check_inplace(in, out_partial),
               std::runtime_error);
  EXPECT_THROW(KokkosFFT::Impl::check_inplace(out_partial, in),
               std::runtime_error);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_has_duplicate_values() {
  ContainerType0 v0 = {0, 1, 1};
  ContainerType1 v1 = {0, 1, 1, 1};
  ContainerType1 v2 = {0, 1, 2, 3};
  ContainerType2 v3 = {0};

  EXPECT_TRUE(KokkosFFT::Impl::has_duplicate_values(v0));
  EXPECT_TRUE(KokkosFFT::Impl::has_duplicate_values(v1));
  EXPECT_FALSE(KokkosFFT::Impl::has_duplicate_values(v2));
  EXPECT_FALSE(KokkosFFT::Impl::has_duplicate_values(v3));
}

template <typename ContainerType>
void test_is_out_of_range_value_included() {
  using IntType = std::remove_cv_t<
      std::remove_reference_t<typename ContainerType::value_type>>;
  ContainerType v = {0, 1, 2, 3, 4}, v2 = {0, 4, 1};

  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(2)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(3)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(4)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(5)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v, static_cast<IntType>(6)));

  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(2)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(3)));
  EXPECT_TRUE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(4)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(5)));
  EXPECT_FALSE(KokkosFFT::Impl::is_out_of_range_value_included(
      v2, static_cast<IntType>(6)));

  if constexpr (std::is_signed_v<IntType>) {
    // Since non-negative value is included, it should be invalid
    ContainerType v3 = {0, 1, -1};
    EXPECT_THROW(
        {
          KokkosFFT::Impl::is_out_of_range_value_included(
              v3, static_cast<IntType>(2));
        },
        std::runtime_error);
  }
}

template <typename IntType>
void test_are_valid_axes() {
  using real_type  = double;
  using View1DType = Kokkos::View<real_type*>;
  using View2DType = Kokkos::View<real_type**>;
  using View3DType = Kokkos::View<real_type***>;
  using View4DType = Kokkos::View<real_type****>;

  std::array<IntType, 1> axes0 = {0};
  std::array<IntType, 2> axes1 = {0, 1};
  std::array<IntType, 3> axes2 = {0, 1, 2};
  std::array<IntType, 3> axes3 = {0, 1, 1};
  std::array<IntType, 3> axes4 = {0, 1, 3};

  View1DType view1;
  View2DType view2;
  View3DType view3;
  View4DType view4;

  // 1D axes on 1D+ Views
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view1, axes0));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view2, axes0));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view3, axes0));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes0));

  // 2D axes on 2D+ Views
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view2, axes1));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view3, axes1));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes1));

  // 3D axes on 3D+ Views
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view3, axes2));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes2));
  EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes4));

  // 3D axes on 3D Views with out of range -> should fail
  EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view3, axes4));

  // axes include overlap -> should fail
  EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view3, axes3));
  EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view4, axes3));

  if constexpr (std::is_signed_v<IntType>) {
    // {0, 1, -2} is converted to {0, 1, 1} for 3D View and {0, 1, 2}
    // for 4D View. Invalid for 3D View but valid for 4D View
    std::array<IntType, 3> axes5 = {0, 1, -2};

    // axes include overlap -> should fail
    EXPECT_FALSE(KokkosFFT::Impl::are_valid_axes(view3, axes5));

    // axes do not include overlap -> OK
    EXPECT_TRUE(KokkosFFT::Impl::are_valid_axes(view4, axes5));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestPointerTypes, paired_scalar_types);
TYPED_TEST_SUITE(TestContainerTypes, int_types);
TYPED_TEST_SUITE(TestAxes, int_types);

TYPED_TEST(TestPointerTypes, are_pointers_aliasing) {
  using value_type1 = typename TestFixture::value_type1;
  using value_type2 = typename TestFixture::value_type2;
  test_are_pointers_aliasing<value_type1, value_type2>();
}

TYPED_TEST(TestPointerTypes, are_overlapping) {
  using value_type1 = typename TestFixture::value_type1;
  using value_type2 = typename TestFixture::value_type2;
  test_are_overlapping<value_type1, value_type2>();
}

TYPED_TEST(TestPointerTypes, check_inplace) {
  using value_type1 = typename TestFixture::value_type1;
  using value_type2 = typename TestFixture::value_type2;
  test_check_inplace<value_type1, value_type2>();
}

TYPED_TEST(TestContainerTypes, has_duplicate_values_in_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  test_has_duplicate_values<vector_type, vector_type, vector_type>();
}

TYPED_TEST(TestContainerTypes, has_duplicate_values_in_array) {
  using value_type  = typename TestFixture::value_type;
  using array_type0 = std::array<value_type, 3>;
  using array_type1 = std::array<value_type, 4>;
  using array_type2 = std::array<value_type, 1>;
  test_has_duplicate_values<array_type0, array_type1, array_type2>();
}

TYPED_TEST(TestContainerTypes, is_out_of_range_value_included_in_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  test_is_out_of_range_value_included<vector_type>();
}

TYPED_TEST(TestContainerTypes, is_out_of_range_value_included_in_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 5>;
  test_is_out_of_range_value_included<array_type>();
}

TYPED_TEST(TestAxes, are_valid_axes) {
  using value_type = typename TestFixture::value_type;
  test_are_valid_axes<value_type>();
}
