// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <array>
#include <type_traits>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "KokkosFFT_Container_Helpers.hpp"

namespace {
// Int like types
using int_types = ::testing::Types<int, std::size_t>;

// Int or float types
using data_types = ::testing::Types<int, std::size_t, float, double>;

// value type combinations that are tested
using paired_int_types =
    ::testing::Types<std::pair<int, int>, std::pair<int, std::size_t>,
                     std::pair<std::size_t, int>,
                     std::pair<std::size_t, std::size_t>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestIndexSequence : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestArange : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestTotalSize : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestConvertBaseTypes : public ::testing::Test {
  using value_type1 = typename T::first_type;
  using value_type2 = typename T::second_type;
};

template <typename T>
struct TestReversed : public ::testing::Test {
  using value_type = T;
};

template <typename IntType>
void test_index_sequence() {
  // Rank of the index sequence
  constexpr std::size_t DIM0 = 3;
  constexpr std::size_t DIM1 = 4;
  constexpr std::size_t DIM2 = 5;
  if constexpr (std::is_signed_v<IntType>) {
    constexpr IntType start0 = -static_cast<IntType>(DIM0);
    constexpr IntType start1 = -static_cast<IntType>(DIM1);
    constexpr IntType start2 = -static_cast<IntType>(DIM2);

    constexpr auto default_axes0 =
        KokkosFFT::Impl::index_sequence<IntType, DIM0, start0>();
    constexpr auto default_axes1 =
        KokkosFFT::Impl::index_sequence<IntType, DIM1, start1>();
    constexpr auto default_axes2 =
        KokkosFFT::Impl::index_sequence<IntType, DIM2, start2>();

    std::array<IntType, DIM0> ref_axes0 = {-3, -2, -1};
    std::array<IntType, DIM1> ref_axes1 = {-4, -3, -2, -1};
    std::array<IntType, DIM2> ref_axes2 = {-5, -4, -3, -2, -1};

    EXPECT_EQ(default_axes0, ref_axes0);
    EXPECT_EQ(default_axes1, ref_axes1);
    EXPECT_EQ(default_axes2, ref_axes2);
  } else {
    constexpr auto range0 = KokkosFFT::Impl::index_sequence<IntType, DIM0, 0>();
    constexpr auto range1 = KokkosFFT::Impl::index_sequence<IntType, DIM1, 0>();
    constexpr auto range2 = KokkosFFT::Impl::index_sequence<IntType, DIM2, 0>();
    std::array<IntType, DIM0> ref_range0 = {0, 1, 2};
    std::array<IntType, DIM1> ref_range1 = {0, 1, 2, 3};
    std::array<IntType, DIM2> ref_range2 = {0, 1, 2, 3, 4};

    EXPECT_EQ(range0, ref_range0);
    EXPECT_EQ(range1, ref_range1);
    EXPECT_EQ(range2, ref_range2);
  }
}

template <typename ValueType>
void test_arange() {
  ValueType seq_start = 0, seq_stop = 5;
  auto seq_result = KokkosFFT::Impl::arange(seq_start, seq_stop);
  std::vector<ValueType> ref_result = {0, 1, 2, 3, 4};
  EXPECT_EQ(seq_result, ref_result);

  auto empty_result = KokkosFFT::Impl::arange(seq_stop, seq_start);
  std::vector<ValueType> ref_empty_result{};
  EXPECT_EQ(empty_result, ref_empty_result);

  if constexpr (std::is_integral_v<ValueType>) {
    auto pos_result = KokkosFFT::Impl::arange(static_cast<ValueType>(0),
                                              static_cast<ValueType>(10),
                                              static_cast<ValueType>(2));
    std::vector<ValueType> ref_pos_result = {0, 2, 4, 6, 8};
    EXPECT_EQ(pos_result, ref_pos_result);

    if constexpr (std::is_signed_v<ValueType>) {
      ValueType start = -3, stop = 4, step = 2;
      auto neg_result = KokkosFFT::Impl::arange(start, stop, step);
      std::vector<ValueType> ref_neg_result = {-3, -1, 1, 3};
      EXPECT_EQ(neg_result, ref_neg_result);

      ValueType neg_step   = -2;
      auto neg_step_result = KokkosFFT::Impl::arange(stop, start, neg_step);
      std::vector<ValueType> ref_neg_step_result = {4, 2, 0, -2};
      EXPECT_EQ(neg_step_result, ref_neg_step_result);

      auto neg_empty_result = KokkosFFT::Impl::arange(start, stop, neg_step);
      std::vector<ValueType> ref_neg_empty_result{};
      EXPECT_EQ(neg_empty_result, ref_neg_empty_result);
    }
  } else {
    // For non-integral type, we test an explicit non-unit step size
    ValueType start = -1.0, stop = 1.1, step = 0.5;
    auto result = KokkosFFT::Impl::arange(start, stop, step);
    std::vector<ValueType> ref_result = {-1.0, -0.5, 0.0, 0.5, 1.0};
    EXPECT_EQ(result, ref_result);
  }

  // Failure test for zero step size
  EXPECT_THROW(
      {
        ValueType zero_step = 0;
        [[maybe_unused]] auto result_zero_step =
            KokkosFFT::Impl::arange(seq_start, seq_stop, zero_step);
      },
      std::runtime_error);

  // Failure test for overflow in the number of elements
  if constexpr (std::is_floating_point_v<ValueType>) {
    const ValueType overflow_start = 0;
    const ValueType overflow_stop  = std::numeric_limits<ValueType>::max();
    const ValueType overflow_step  = std::numeric_limits<ValueType>::min();

    EXPECT_THROW(
        {
          [[maybe_unused]] auto overflow_result = KokkosFFT::Impl::arange(
              overflow_start, overflow_stop, overflow_step);
        },
        std::runtime_error);
  }
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_total_size() {
  using IntType = KokkosFFT::Impl::base_container_value_type<ContainerType0>;

  ContainerType0 v0 = {0, 1, 4, 2, 3};
  ContainerType1 v1 = {2, 3, 5};
  ContainerType2 v2 = {1};
  auto total_size0  = KokkosFFT::Impl::total_size(v0);
  auto total_size1  = KokkosFFT::Impl::total_size(v1);
  auto total_size2  = KokkosFFT::Impl::total_size(v2);

  IntType ref_total_size0 = 0, ref_total_size1 = 30, ref_total_size2 = 1;

  EXPECT_EQ(total_size0, ref_total_size0);
  EXPECT_EQ(total_size1, ref_total_size1);
  EXPECT_EQ(total_size2, ref_total_size2);

  // Failure test with overflow
  ContainerType1 v3 = {2, 3, std::numeric_limits<IntType>::max()},
                 v4 = {1, std::numeric_limits<IntType>::max(),
                       std::numeric_limits<IntType>::max()},
                 v5 = {1, std::numeric_limits<IntType>::min(),
                       std::numeric_limits<IntType>::max()},
                 v6 = {1, std::numeric_limits<IntType>::max(),
                       std::numeric_limits<IntType>::min()},
                 v7 = {1, std::numeric_limits<IntType>::min(),
                       std::numeric_limits<IntType>::min()};

  EXPECT_THROW(
      { [[maybe_unused]] auto total_size3 = KokkosFFT::Impl::total_size(v3); },
      std::overflow_error);

  EXPECT_THROW(
      { [[maybe_unused]] auto total_size4 = KokkosFFT::Impl::total_size(v4); },
      std::overflow_error);

  // We expect overflow
  if constexpr (std::is_signed_v<IntType>) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size5 = KokkosFFT::Impl::total_size(v5);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size6 = KokkosFFT::Impl::total_size(v6);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size7 = KokkosFFT::Impl::total_size(v7);
        },
        std::overflow_error);
  } else {
    // This should be just zero for unsigned case
    auto total_size5 = KokkosFFT::Impl::total_size(v5);
    EXPECT_EQ(total_size5, 0);

    // This should be just zero for unsigned case
    auto total_size6 = KokkosFFT::Impl::total_size(v6);
    EXPECT_EQ(total_size6, 0);

    // This should be just zero for unsigned case
    auto total_size7 = KokkosFFT::Impl::total_size(v7);
    EXPECT_EQ(total_size7, 0);
  }

  // Including max still OK
  ContainerType1 v8       = {1, 2, std::numeric_limits<IntType>::min() / 2};
  auto total_size8        = KokkosFFT::Impl::total_size(v8);
  IntType ref_total_size8 = std::numeric_limits<IntType>::min();
  EXPECT_EQ(total_size8, ref_total_size8);

  if constexpr (std::is_signed_v<IntType>) {
    // Failure test with overflow
    ContainerType1 iv0 = {-1, 2, std::numeric_limits<IntType>::max()},
                   iv1 = {-2, 3, std::numeric_limits<IntType>::min()},
                   iv2 = {1, std::numeric_limits<IntType>::min(), -2},
                   iv3 = {1, std::numeric_limits<IntType>::min(), -1},
                   iv4 = {1, std::numeric_limits<IntType>::max(), -1};
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i0 =
              KokkosFFT::Impl::total_size(iv0);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i1 =
              KokkosFFT::Impl::total_size(iv1);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i2 =
              KokkosFFT::Impl::total_size(iv2);
        },
        std::overflow_error);
    EXPECT_THROW(
        {
          [[maybe_unused]] auto total_size_i3 =
              KokkosFFT::Impl::total_size(iv3);
        },
        std::overflow_error);

    auto total_size_i4        = KokkosFFT::Impl::total_size(iv4);
    IntType ref_total_size_i4 = std::numeric_limits<IntType>::min() + 1;
    EXPECT_EQ(total_size_i4, ref_total_size_i4);
  }
}

template <typename ToContainerType, typename FromContainerType>
void test_convert_base_int_type() {
  using From = KokkosFFT::Impl::base_container_value_type<FromContainerType>;
  using To   = KokkosFFT::Impl::base_container_value_type<ToContainerType>;

  FromContainerType v   = {2, 3, 5};
  ToContainerType v_ref = {2, 3, 5};

  auto v_converted = KokkosFFT::Impl::convert_base_int_type<To>(v);
  EXPECT_EQ(v_converted, v_ref);

  if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
    // Same signedness, should be OK
    FromContainerType v2   = {-2, -3, -5};
    ToContainerType v2_ref = {-2, -3, -5};

    auto v2_converted = KokkosFFT::Impl::convert_base_int_type<To>(v2);
    EXPECT_EQ(v2_converted, v2_ref);
  } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
    // From signed to unsigned, negative value should raise error
    FromContainerType v2 = {-2, 3, 5};
    EXPECT_THROW(
        {
          [[maybe_unused]] auto v2_converted =
              KokkosFFT::Impl::convert_base_int_type<To>(v2);
        },
        std::overflow_error);
  } else if constexpr (std::is_unsigned_v<From>) {
    // From std::size_t to int, overflow should raise error
    if (std::numeric_limits<From>::max() > std::numeric_limits<To>::max()) {
      FromContainerType v2 = {2, 3, std::numeric_limits<From>::max()};
      EXPECT_THROW(
          {
            [[maybe_unused]] auto v2_converted =
                KokkosFFT::Impl::convert_base_int_type<To>(v2);
          },
          std::overflow_error);
    }
  }
}

template <typename ContainerType>
void test_reversed() {
  ContainerType v = {2, 3, 5}, v_fixed = {2, 3, 5};
  ContainerType ref_reversed = {5, 3, 2};

  // Lvalue test
  auto out = KokkosFFT::Impl::reversed(v);
  EXPECT_EQ(out, ref_reversed);
  EXPECT_EQ(v, v_fixed) << "Input container modified in lvalue test";

  // Rvalue test
  auto out2 = KokkosFFT::Impl::reversed(ContainerType{2, 3, 5});
  auto out3 = KokkosFFT::Impl::reversed(std::move(v));
  EXPECT_EQ(out2, ref_reversed);
  EXPECT_EQ(out3, ref_reversed);

  // Check behavior of moved-from container
  if constexpr (KokkosFFT::Impl::is_std_vector_v<ContainerType>) {
    // The standard only guarantees that v is valid but unspecified.
    // We can safely check that we can call methods on it:
    EXPECT_NO_THROW({
      [[maybe_unused]] auto sz = v.size();
      v.clear();
    }) << "Moved-from vector should be in a valid state";
  } else {
    // For std::array with fundamental types, move is equivalent to copy
    // The original array should retain its values since elements are
    // fundamental types
    EXPECT_EQ(v, v_fixed)
        << "Array with fundamental types should be unchanged after move";
  }
}

}  // namespace

TYPED_TEST_SUITE(TestIndexSequence, int_types);
TYPED_TEST_SUITE(TestArange, data_types);
TYPED_TEST_SUITE(TestTotalSize, int_types);
TYPED_TEST_SUITE(TestConvertBaseTypes, paired_int_types);
TYPED_TEST_SUITE(TestReversed, int_types);

TYPED_TEST(TestIndexSequence, make_sequence_3Dto5D) {
  using value_type = typename TestFixture::value_type;
  test_index_sequence<value_type>();
}

TYPED_TEST(TestArange, arange) {
  using value_type = typename TestFixture::value_type;
  test_arange<value_type>();
}

TYPED_TEST(TestTotalSize, total_size_of_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  test_total_size<vector_type, vector_type, vector_type>();
}

TYPED_TEST(TestTotalSize, total_size_of_array) {
  using value_type  = typename TestFixture::value_type;
  using array_type0 = std::array<value_type, 5>;
  using array_type1 = std::array<value_type, 3>;
  using array_type2 = std::array<value_type, 1>;
  test_total_size<array_type0, array_type1, array_type2>();
}

TYPED_TEST(TestConvertBaseTypes, convert_arrays) {
  using value_type1 = typename TestFixture::value_type1;
  using value_type2 = typename TestFixture::value_type2;
  using array_type1 = std::array<value_type1, 3>;
  using array_type2 = std::array<value_type2, 3>;
  test_convert_base_int_type<array_type1, array_type2>();
}

TYPED_TEST(TestConvertBaseTypes, convert_vectors) {
  using value_type1  = typename TestFixture::value_type1;
  using value_type2  = typename TestFixture::value_type2;
  using vector_type1 = std::vector<value_type1>;
  using vector_type2 = std::vector<value_type2>;
  test_convert_base_int_type<vector_type1, vector_type2>();
}

TYPED_TEST(TestReversed, reversed_of_arrays) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  test_reversed<array_type>();
}

TYPED_TEST(TestReversed, reversed_of_vectors) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  test_reversed<vector_type>();
}
