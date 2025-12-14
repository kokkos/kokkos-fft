// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Layout.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

template <std::size_t DIM>
using axes_type    = std::array<int, DIM>;
using layout_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight,
                                      Kokkos::LayoutStride>;

using test_types =
    ::testing::Types<std::pair<int, Kokkos::LayoutLeft>,
                     std::pair<int, Kokkos::LayoutRight>,
                     std::pair<std::size_t, Kokkos::LayoutLeft>,
                     std::pair<std::size_t, Kokkos::LayoutRight> >;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct CompileTestLayoutIterate : public ::testing::Test {
  using layout_type = T;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct TestCreateLayout : public ::testing::Test {
  using index_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename LayoutType>
void test_layout_iterate_type_selector() {
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ASSERT_EQ(KokkosFFT::Impl::layout_iterate_type_selector<
                  LayoutType>::inner_iteration_pattern,
              Kokkos::Iterate::Left);
    ASSERT_EQ(KokkosFFT::Impl::layout_iterate_type_selector<
                  LayoutType>::outer_iteration_pattern,
              Kokkos::Iterate::Left);
  } else if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    ASSERT_EQ(KokkosFFT::Impl::layout_iterate_type_selector<
                  LayoutType>::inner_iteration_pattern,
              Kokkos::Iterate::Right);
    ASSERT_EQ(KokkosFFT::Impl::layout_iterate_type_selector<
                  LayoutType>::outer_iteration_pattern,
              Kokkos::Iterate::Right);
  } else {
    ASSERT_EQ(KokkosFFT::Impl::layout_iterate_type_selector<
                  LayoutType>::inner_iteration_pattern,
              Kokkos::Iterate::Default);
    ASSERT_EQ(KokkosFFT::Impl::layout_iterate_type_selector<
                  LayoutType>::outer_iteration_pattern,
              Kokkos::Iterate::Default);
  }
}

template <typename IndexType, typename LayoutType>
void test_create_layout() {
  constexpr std::size_t max_rank = 8;
  for (std::size_t rank = 1; rank <= max_rank; ++rank) {
    std::array<IndexType, max_rank> extents;
    for (std::size_t i = 0; i < max_rank; ++i) {
      extents.at(i) = static_cast<IndexType>(i + 1);
    }
    auto layout = KokkosFFT::Impl::create_layout<LayoutType>(extents);
    for (std::size_t i = 0; i < max_rank; ++i) {
      EXPECT_EQ(layout.dimension[i], extents.at(i));
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(CompileTestLayoutIterate, layout_types);
TYPED_TEST_SUITE(TestCreateLayout, test_types);

// Tests for layout_iterate_type_selector
TYPED_TEST(CompileTestLayoutIterate, LayoutIterateTypeSelector) {
  using layout_type = typename TestFixture::layout_type;
  test_layout_iterate_type_selector<layout_type>();
}

// Test create layout for 1D to 8D
TYPED_TEST(TestCreateLayout, 1Dto8D) {
  using index_type  = typename TestFixture::index_type;
  using layout_type = typename TestFixture::layout_type;
  test_create_layout<index_type, layout_type>();
}
