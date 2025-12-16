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
struct TestLayoutIterate : public ::testing::Test {
  using layout_type = T;
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
  std::size_t n1 = 1, n2 = 1, n3 = 2, n4 = 3, n5 = 5, n6 = 8, n7 = 13, n8 = 21;

  std::array<std::size_t, 1> ref_extents1D = {n1};
  std::array<std::size_t, 2> ref_extents2D = {n1, n2};
  std::array<std::size_t, 3> ref_extents3D = {n1, n2, n3};
  std::array<std::size_t, 4> ref_extents4D = {n1, n2, n3, n4};
  std::array<std::size_t, 5> ref_extents5D = {n1, n2, n3, n4, n5};
  std::array<std::size_t, 6> ref_extents6D = {n1, n2, n3, n4, n5, n6};
  std::array<std::size_t, 7> ref_extents7D = {n1, n2, n3, n4, n5, n6, n7};
  std::array<std::size_t, 8> ref_extents8D = {n1, n2, n3, n4, n5, n6, n7, n8};

  auto layout1 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents1D);
  auto layout2 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents2D);
  auto layout3 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents3D);
  auto layout4 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents4D);
  auto layout5 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents5D);
  auto layout6 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents6D);
  auto layout7 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents7D);
  auto layout8 = KokkosFFT::Impl::create_layout<LayoutType>(ref_extents8D);

  using layout_type = decltype(layout1);
  testing::StaticAssertTypeEq<typename layout_type::array_layout, LayoutType>();

  for (std::size_t i = 0; i < ref_extents1D.size(); ++i) {
    EXPECT_EQ(layout1.dimension[i], ref_extents1D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents2D.size(); ++i) {
    EXPECT_EQ(layout2.dimension[i], ref_extents2D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents3D.size(); ++i) {
    EXPECT_EQ(layout3.dimension[i], ref_extents3D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents4D.size(); ++i) {
    EXPECT_EQ(layout4.dimension[i], ref_extents4D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents5D.size(); ++i) {
    EXPECT_EQ(layout5.dimension[i], ref_extents5D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents6D.size(); ++i) {
    EXPECT_EQ(layout6.dimension[i], ref_extents6D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents7D.size(); ++i) {
    EXPECT_EQ(layout7.dimension[i], ref_extents7D.at(i));
  }

  for (std::size_t i = 0; i < ref_extents8D.size(); ++i) {
    EXPECT_EQ(layout8.dimension[i], ref_extents8D.at(i));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestLayoutIterate, layout_types);
TYPED_TEST_SUITE(TestCreateLayout, test_types);

// Tests for layout_iterate_type_selector
TYPED_TEST(TestLayoutIterate, LayoutIterateTypeSelector) {
  using layout_type = typename TestFixture::layout_type;
  test_layout_iterate_type_selector<layout_type>();
}

// Test create layout for 1D to 8D
TYPED_TEST(TestCreateLayout, 1Dto8D) {
  using index_type  = typename TestFixture::index_type;
  using layout_type = typename TestFixture::layout_type;
  test_create_layout<index_type, layout_type>();
}
