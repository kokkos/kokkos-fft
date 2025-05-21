// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <utility>
#include <gtest/gtest.h>
#include "KokkosFFT_TypeCartesianProduct.hpp"

// All the tests in this file are compile time tests, so we skip all the tests
// by GTEST_SKIP(). gtest is used for type parameterization.

namespace {
using multiple_types = ::testing::Types<bool, int, std::size_t, float, double>;
using integer_types  = ::testing::Types<int, std::size_t>;

// Define the types to combine
using base_real_types = std::tuple<float, double, long double>;
using base_int_types  = std::tuple<bool, int, std::size_t>;

using multiple_tuple_types = ::testing::Types<base_real_types, base_int_types>;

// Define fixtures
template <typename T>
struct CompileTestManipulateTuples : public ::testing::Test {
  using value_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestCartesianProduct : public ::testing::Test {
  using tuple_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestValueCartesianProduct : public ::testing::Test {
  using int_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestMakeCartesianProduct : public ::testing::Test {
  using tuple_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

// Tests for appending ValueType or std::tuple<ValueType> to
// Input tuple type
template <typename ValueType>
void test_concat_single_tuple() {
  using input_tuple_type = std::tuple<std::tuple<double, double>>;
  using tuple_and_tuple_type =
      KokkosFFT::Testing::Impl::for_each_tuple_cat_t<input_tuple_type,
                                                     std::tuple<ValueType>>;
  using tuple_and_value_type =
      KokkosFFT::Testing::Impl::for_each_tuple_cat_t<input_tuple_type,
                                                     ValueType>;
  using reference_tuple_type =
      std::tuple<std::tuple<double, double, ValueType>>;

  testing::StaticAssertTypeEq<tuple_and_value_type, reference_tuple_type>();
  testing::StaticAssertTypeEq<tuple_and_tuple_type, reference_tuple_type>();
}

// Tests for appending ValueType or std::tuple<ValueType> to
// Input 2D tuple type
template <typename ValueType>
void test_concat_two_tuples() {
  using input_tuple_type =
      std::tuple<std::tuple<double, double>, std::tuple<int, double>>;
  using tuple_and_tuple_type =
      KokkosFFT::Testing::Impl::for_each_tuple_cat_t<input_tuple_type,
                                                     std::tuple<ValueType>>;
  using tuple_and_value_type =
      KokkosFFT::Testing::Impl::for_each_tuple_cat_t<input_tuple_type,
                                                     ValueType>;
  using reference_tuple_type = std::tuple<std::tuple<double, double, ValueType>,
                                          std::tuple<int, double, ValueType>>;

  testing::StaticAssertTypeEq<tuple_and_value_type, reference_tuple_type>();
  testing::StaticAssertTypeEq<tuple_and_tuple_type, reference_tuple_type>();
}

// Tests for transforming a std::tuple<Args...> to a testing::Types<Args...>,
// identity otherwise
template <typename ValueType>
void test_tuple_to_testing_types() {
  using from_value_type = KokkosFFT::Testing::Impl::tuple_to_types_t<ValueType>;
  using from_tuple_type =
      KokkosFFT::Testing::Impl::tuple_to_types_t<std::tuple<ValueType>>;
  using from_multi_tuple_type =
      KokkosFFT::Testing::Impl::tuple_to_types_t<std::tuple<int, ValueType>>;
  using from_nested_tuple_type = KokkosFFT::Testing::Impl::tuple_to_types_t<
      std::tuple<std::tuple<int>, std::tuple<ValueType>>>;
  using reference_value_type       = ValueType;  // Do nothing on non-tuple type
  using reference_tuple_type       = ::testing::Types<ValueType>;
  using reference_multi_tuple_type = ::testing::Types<int, ValueType>;
  using reference_nested_tuple_type =
      ::testing::Types<std::tuple<int>, std::tuple<ValueType>>;

  testing::StaticAssertTypeEq<from_value_type, reference_value_type>();
  testing::StaticAssertTypeEq<from_tuple_type, reference_tuple_type>();
  testing::StaticAssertTypeEq<from_multi_tuple_type,
                              reference_multi_tuple_type>();
  testing::StaticAssertTypeEq<from_nested_tuple_type,
                              reference_nested_tuple_type>();
}

// Tests for getting a cartesian product of types
// E.g.
// std::tuple<float, double, long double> would be converted to
// std::tuple< std::tuple<float>, std::tuple<double>, std::tuple<long double> >
template <typename TupleType>
void test_cartesian_product_of_tuple1D() {
  using cartesian_product_type =
      KokkosFFT::Testing::Impl::cartesian_product_t<TupleType>;
  using T0 = typename std::tuple_element<0, TupleType>::type;
  using T1 = typename std::tuple_element<1, TupleType>::type;
  using T2 = typename std::tuple_element<2, TupleType>::type;

  using reference_tuple_type =
      std::tuple<std::tuple<T0>, std::tuple<T1>, std::tuple<T2>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

// Tests for getting a cartesian product of types
// E.g.
// std::tuple<float, double, long double> && std::tuple<float, double, long
// double> would be converted to std::tuple< std::tuple<float, float>,
// std::tuple<double, float>, std::tuple<long double, float>,
//             std::tuple<float, double>, std::tuple<double, double>,
//             std::tuple<long double, double>, std::tuple<float, long double>,
//             std::tuple<double, long double>, std::tuple<long double, long
//             double>>
template <typename TupleType1, typename TupleType2>
void test_cartesian_product_of_tuple2D() {
  using cartesian_product_type =
      KokkosFFT::Testing::Impl::cartesian_product_t<TupleType1, TupleType2>;

  // Analyze TupleType1
  using T0_1 = typename std::tuple_element<0, TupleType1>::type;
  using T1_1 = typename std::tuple_element<1, TupleType1>::type;
  using T2_1 = typename std::tuple_element<2, TupleType1>::type;

  // Analyze TupleType2
  using T0_2 = typename std::tuple_element<0, TupleType2>::type;
  using T1_2 = typename std::tuple_element<1, TupleType2>::type;
  using T2_2 = typename std::tuple_element<2, TupleType2>::type;

  using reference_tuple_type = std::tuple<
      std::tuple<T0_1, T0_2>, std::tuple<T1_1, T0_2>, std::tuple<T2_1, T0_2>,
      std::tuple<T0_1, T1_2>, std::tuple<T1_1, T1_2>, std::tuple<T2_1, T1_2>,
      std::tuple<T0_1, T2_2>, std::tuple<T1_1, T2_2>, std::tuple<T2_1, T2_2>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

// Tests for getting a cartesian product of types
// E.g.
// std::tuple<float, double, long double> && std::tuple<float, double, long
// double> would be converted to std::tuple< std::tuple<float, float, float>,
// std::tuple<double, float, float>, std::tuple<long double, float, float>,
//             ...
template <typename TupleType1, typename TupleType2, typename TupleType3>
void test_cartesian_product_of_tuple3D() {
  using cartesian_product_type =
      KokkosFFT::Testing::Impl::cartesian_product_t<TupleType1, TupleType2,
                                                    TupleType3>;

  // Analyze TupleType1
  using T0_1 = typename std::tuple_element<0, TupleType1>::type;
  using T1_1 = typename std::tuple_element<1, TupleType1>::type;
  using T2_1 = typename std::tuple_element<2, TupleType1>::type;

  // Analyze TupleType2
  using T0_2 = typename std::tuple_element<0, TupleType2>::type;
  using T1_2 = typename std::tuple_element<1, TupleType2>::type;
  using T2_2 = typename std::tuple_element<2, TupleType2>::type;

  // Analyze TupleType3
  using T0_3 = typename std::tuple_element<0, TupleType3>::type;
  using T1_3 = typename std::tuple_element<1, TupleType3>::type;
  using T2_3 = typename std::tuple_element<2, TupleType3>::type;

  using reference_tuple_type =
      std::tuple<std::tuple<T0_1, T0_2, T0_3>, std::tuple<T1_1, T0_2, T0_3>,
                 std::tuple<T2_1, T0_2, T0_3>, std::tuple<T0_1, T1_2, T0_3>,
                 std::tuple<T1_1, T1_2, T0_3>, std::tuple<T2_1, T1_2, T0_3>,
                 std::tuple<T0_1, T2_2, T0_3>, std::tuple<T1_1, T2_2, T0_3>,
                 std::tuple<T2_1, T2_2, T0_3>, std::tuple<T0_1, T0_2, T1_3>,
                 std::tuple<T1_1, T0_2, T1_3>, std::tuple<T2_1, T0_2, T1_3>,
                 std::tuple<T0_1, T1_2, T1_3>, std::tuple<T1_1, T1_2, T1_3>,
                 std::tuple<T2_1, T1_2, T1_3>, std::tuple<T0_1, T2_2, T1_3>,
                 std::tuple<T1_1, T2_2, T1_3>, std::tuple<T2_1, T2_2, T1_3>,
                 std::tuple<T0_1, T0_2, T2_3>, std::tuple<T1_1, T0_2, T2_3>,
                 std::tuple<T2_1, T0_2, T2_3>, std::tuple<T0_1, T1_2, T2_3>,
                 std::tuple<T1_1, T1_2, T2_3>, std::tuple<T2_1, T1_2, T2_3>,
                 std::tuple<T0_1, T2_2, T2_3>, std::tuple<T1_1, T2_2, T2_3>,
                 std::tuple<T2_1, T2_2, T2_3>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

// Tests for getting a cartesian product of sequences
// E.g.
// std::integer_sequence<std::size_t, 1, 2, 3> && std::integer_sequence<bool,
// true, false> would be converted to
// std::tuple<std::tuple<std::integral_constant<std::size_t, 1>,
// std::integral_constant<bool, true>>,
//             std::tuple<std::integral_constant<std::size_t, 2>,
//             std::integral_constant<bool, true>>,
//             std::tuple<std::integral_constant<std::size_t, 3>,
//             std::integral_constant<bool, true>>,
//             std::tuple<std::integral_constant<std::size_t, 1>,
//             std::integral_constant<bool, false>>,
//             std::tuple<std::integral_constant<std::size_t, 2>,
//             std::integral_constant<bool, false>>,
//             std::tuple<std::integral_constant<std::size_t, 3>,
//             std::integral_constant<bool, false>>>
template <typename IntType>
void test_cartesian_product_of_values() {
  using integers = std::integer_sequence<IntType, 1, 2, 3>;
  using booleans = std::integer_sequence<bool, true, false>;

  using cartesian_product_type =
      KokkosFFT::Testing::Impl::cartesian_product_t<integers, booleans>;

  using reference_tuple_type =
      std::tuple<std::tuple<std::integral_constant<IntType, 1>,
                            std::integral_constant<bool, true>>,
                 std::tuple<std::integral_constant<IntType, 2>,
                            std::integral_constant<bool, true>>,
                 std::tuple<std::integral_constant<IntType, 3>,
                            std::integral_constant<bool, true>>,
                 std::tuple<std::integral_constant<IntType, 1>,
                            std::integral_constant<bool, false>>,
                 std::tuple<std::integral_constant<IntType, 2>,
                            std::integral_constant<bool, false>>,
                 std::tuple<std::integral_constant<IntType, 3>,
                            std::integral_constant<bool, false>>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

// Tests for getting a cartesian product of types
// E.g.
// std::tuple<float, double, long double> would be converted to
// std::tuple< std::tuple<float>, std::tuple<double>, std::tuple<long double> >
template <typename TupleType>
void test_make_cartesian_product_of_tuple1D() {
  using cartesian_product_type =
      KokkosFFT::Testing::make_cartesian_types<TupleType>;
  using T0 = typename std::tuple_element<0, TupleType>::type;
  using T1 = typename std::tuple_element<1, TupleType>::type;
  using T2 = typename std::tuple_element<2, TupleType>::type;

  using reference_tuple_type =
      testing::Types<std::tuple<T0>, std::tuple<T1>, std::tuple<T2>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

// Tests for getting a cartesian product of types
// E.g.
// std::tuple<float, double, long double> && std::tuple<float, double, long
// double> would be converted to std::tuple< std::tuple<float, float>,
// std::tuple<double, float>, std::tuple<long double, float>,
//             std::tuple<float, double>, std::tuple<double, double>,
//             std::tuple<long double, double>, std::tuple<float, long double>,
//             std::tuple<double, long double>, std::tuple<long double, long
//             double>>
template <typename TupleType1, typename TupleType2>
void test_make_cartesian_product_of_tuple2D() {
  using cartesian_product_type =
      KokkosFFT::Testing::make_cartesian_types<TupleType1, TupleType2>;

  // Analyze TupleType1
  using T0_1 = typename std::tuple_element<0, TupleType1>::type;
  using T1_1 = typename std::tuple_element<1, TupleType1>::type;
  using T2_1 = typename std::tuple_element<2, TupleType1>::type;

  // Analyze TupleType2
  using T0_2 = typename std::tuple_element<0, TupleType2>::type;
  using T1_2 = typename std::tuple_element<1, TupleType2>::type;
  using T2_2 = typename std::tuple_element<2, TupleType2>::type;

  using reference_tuple_type = testing::Types<
      std::tuple<T0_1, T0_2>, std::tuple<T1_1, T0_2>, std::tuple<T2_1, T0_2>,
      std::tuple<T0_1, T1_2>, std::tuple<T1_1, T1_2>, std::tuple<T2_1, T1_2>,
      std::tuple<T0_1, T2_2>, std::tuple<T1_1, T2_2>, std::tuple<T2_1, T2_2>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

// Tests for getting a cartesian product of types
// E.g.
// std::tuple<float, double, long double> && std::tuple<float, double, long
// double> would be converted to std::tuple< std::tuple<float, float, float>,
// std::tuple<double, float, float>, std::tuple<long double, float, float>,
//             ...
template <typename TupleType1, typename TupleType2, typename TupleType3>
void test_make_cartesian_product_of_tuple3D() {
  using cartesian_product_type =
      KokkosFFT::Testing::make_cartesian_types<TupleType1, TupleType2,
                                               TupleType3>;

  // Analyze TupleType1
  using T0_1 = typename std::tuple_element<0, TupleType1>::type;
  using T1_1 = typename std::tuple_element<1, TupleType1>::type;
  using T2_1 = typename std::tuple_element<2, TupleType1>::type;

  // Analyze TupleType2
  using T0_2 = typename std::tuple_element<0, TupleType2>::type;
  using T1_2 = typename std::tuple_element<1, TupleType2>::type;
  using T2_2 = typename std::tuple_element<2, TupleType2>::type;

  // Analyze TupleType3
  using T0_3 = typename std::tuple_element<0, TupleType3>::type;
  using T1_3 = typename std::tuple_element<1, TupleType3>::type;
  using T2_3 = typename std::tuple_element<2, TupleType3>::type;

  using reference_tuple_type =
      testing::Types<std::tuple<T0_1, T0_2, T0_3>, std::tuple<T1_1, T0_2, T0_3>,
                     std::tuple<T2_1, T0_2, T0_3>, std::tuple<T0_1, T1_2, T0_3>,
                     std::tuple<T1_1, T1_2, T0_3>, std::tuple<T2_1, T1_2, T0_3>,
                     std::tuple<T0_1, T2_2, T0_3>, std::tuple<T1_1, T2_2, T0_3>,
                     std::tuple<T2_1, T2_2, T0_3>, std::tuple<T0_1, T0_2, T1_3>,
                     std::tuple<T1_1, T0_2, T1_3>, std::tuple<T2_1, T0_2, T1_3>,
                     std::tuple<T0_1, T1_2, T1_3>, std::tuple<T1_1, T1_2, T1_3>,
                     std::tuple<T2_1, T1_2, T1_3>, std::tuple<T0_1, T2_2, T1_3>,
                     std::tuple<T1_1, T2_2, T1_3>, std::tuple<T2_1, T2_2, T1_3>,
                     std::tuple<T0_1, T0_2, T2_3>, std::tuple<T1_1, T0_2, T2_3>,
                     std::tuple<T2_1, T0_2, T2_3>, std::tuple<T0_1, T1_2, T2_3>,
                     std::tuple<T1_1, T1_2, T2_3>, std::tuple<T2_1, T1_2, T2_3>,
                     std::tuple<T0_1, T2_2, T2_3>, std::tuple<T1_1, T2_2, T2_3>,
                     std::tuple<T2_1, T2_2, T2_3>>;

  testing::StaticAssertTypeEq<cartesian_product_type, reference_tuple_type>();
}

}  // namespace

TYPED_TEST_SUITE(CompileTestManipulateTuples, multiple_types);
TYPED_TEST_SUITE(CompileTestCartesianProduct, multiple_tuple_types);
TYPED_TEST_SUITE(CompileTestValueCartesianProduct, integer_types);
TYPED_TEST_SUITE(CompileTestMakeCartesianProduct, multiple_tuple_types);

TYPED_TEST(CompileTestManipulateTuples, ConcatSingleTuple) {
  using value_type = typename TestFixture::value_type;
  test_concat_single_tuple<value_type>();
}

TYPED_TEST(CompileTestManipulateTuples, ConcatTwoTuples) {
  using value_type = typename TestFixture::value_type;
  test_concat_two_tuples<value_type>();
}

TYPED_TEST(CompileTestManipulateTuples, TupleToTestingTypes) {
  using value_type = typename TestFixture::value_type;
  test_tuple_to_testing_types<value_type>();
}

TYPED_TEST(CompileTestCartesianProduct, Tuple1D) {
  using tuple_type = typename TestFixture::tuple_type;
  test_cartesian_product_of_tuple1D<tuple_type>();
}

TYPED_TEST(CompileTestCartesianProduct, Tuple2D) {
  using tuple_type = typename TestFixture::tuple_type;
  test_cartesian_product_of_tuple2D<tuple_type, tuple_type>();
}

TYPED_TEST(CompileTestCartesianProduct, Tuple3D) {
  using tuple_type = typename TestFixture::tuple_type;
  test_cartesian_product_of_tuple3D<tuple_type, tuple_type, tuple_type>();
}

TYPED_TEST(CompileTestValueCartesianProduct, IntegerSequences) {
  using int_type = typename TestFixture::int_type;
  test_cartesian_product_of_values<int_type>();
}

TYPED_TEST(CompileTestMakeCartesianProduct, Tuple1D) {
  using tuple_type = typename TestFixture::tuple_type;
  test_make_cartesian_product_of_tuple1D<tuple_type>();
}

TYPED_TEST(CompileTestMakeCartesianProduct, Tuple2D) {
  using tuple_type = typename TestFixture::tuple_type;
  test_make_cartesian_product_of_tuple2D<tuple_type, tuple_type>();
}

TYPED_TEST(CompileTestMakeCartesianProduct, Tuple3D) {
  using tuple_type = typename TestFixture::tuple_type;
  test_make_cartesian_product_of_tuple3D<tuple_type, tuple_type, tuple_type>();
}
