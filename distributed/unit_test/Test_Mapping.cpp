// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Mapping.hpp"

namespace {
using test_types =
    ::testing::Types<std::pair<int, Kokkos::LayoutLeft>,
                     std::pair<int, Kokkos::LayoutRight>,
                     std::pair<std::size_t, Kokkos::LayoutLeft>,
                     std::pair<std::size_t, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestMapping : public ::testing::Test {
  using value_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename iType, typename LayoutType>
void test_permute_map_by_axes2D_View2D() {
  using map_type = std::array<iType, 2>;

  map_type src_map_01 = {0, 1};
  map_type src_map_10 = {1, 0};
  iType axis0 = 0, axis1 = 1;
  auto dst_map_01_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_01,
                                                                    axis0);
  auto dst_map_10_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_10,
                                                                    axis0);
  auto dst_map_01_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_01,
                                                                    axis1);
  auto dst_map_10_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_10,
                                                                    axis1);

  map_type ref_dst_map_01_axis0, ref_dst_map_10_axis0;
  map_type ref_dst_map_01_axis1, ref_dst_map_10_axis1;
  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_dst_map_01_axis0 = {0, 1};
    ref_dst_map_10_axis0 = {0, 1};
    ref_dst_map_01_axis1 = {1, 0};
    ref_dst_map_10_axis1 = {1, 0};
  } else {
    ref_dst_map_01_axis0 = {1, 0};
    ref_dst_map_10_axis0 = {1, 0};
    ref_dst_map_01_axis1 = {0, 1};
    ref_dst_map_10_axis1 = {0, 1};
  }

  EXPECT_TRUE(dst_map_01_axis0 == ref_dst_map_01_axis0);
  EXPECT_TRUE(dst_map_10_axis0 == ref_dst_map_10_axis0);
  EXPECT_TRUE(dst_map_01_axis1 == ref_dst_map_01_axis1);
  EXPECT_TRUE(dst_map_10_axis1 == ref_dst_map_10_axis1);
}

template <typename iType, typename LayoutType>
void test_permute_map_by_axes3D_View3D() {
  using map_type = std::array<iType, 3>;

  map_type src_map_012 = {0, 1, 2};
  map_type src_map_021 = {0, 2, 1};
  map_type src_map_102 = {1, 0, 2};
  map_type src_map_120 = {1, 2, 0};
  map_type src_map_201 = {2, 0, 1};
  map_type src_map_210 = {2, 1, 0};

  iType axis0 = 0, axis1 = 1, axis2 = 2;

  auto dst_map_012_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_012,
                                                                    axis0);
  auto dst_map_021_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_021,
                                                                    axis0);
  auto dst_map_102_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_102,
                                                                    axis0);
  auto dst_map_120_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_120,
                                                                    axis0);
  auto dst_map_201_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_201,
                                                                    axis0);
  auto dst_map_210_axis0 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_210,
                                                                    axis0);

  auto dst_map_012_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_012,
                                                                    axis1);
  auto dst_map_021_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_021,
                                                                    axis1);
  auto dst_map_102_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_102,
                                                                    axis1);
  auto dst_map_120_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_120,
                                                                    axis1);
  auto dst_map_201_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_201,
                                                                    axis1);
  auto dst_map_210_axis1 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_210,
                                                                    axis1);

  auto dst_map_012_axis2 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_012,
                                                                    axis2);
  auto dst_map_021_axis2 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_021,
                                                                    axis2);
  auto dst_map_102_axis2 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_102,
                                                                    axis2);
  auto dst_map_120_axis2 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_120,
                                                                    axis2);
  auto dst_map_201_axis2 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_201,
                                                                    axis2);
  auto dst_map_210_axis2 =
      KokkosFFT::Distributed::Impl::permute_map_by_axes<LayoutType>(src_map_210,
                                                                    axis2);

  map_type ref_dst_map_012_axis0, ref_dst_map_021_axis0, ref_dst_map_102_axis0,
      ref_dst_map_120_axis0, ref_dst_map_201_axis0, ref_dst_map_210_axis0;
  map_type ref_dst_map_012_axis1, ref_dst_map_021_axis1, ref_dst_map_102_axis1,
      ref_dst_map_120_axis1, ref_dst_map_201_axis1, ref_dst_map_210_axis1;
  map_type ref_dst_map_012_axis2, ref_dst_map_021_axis2, ref_dst_map_102_axis2,
      ref_dst_map_120_axis2, ref_dst_map_201_axis2, ref_dst_map_210_axis2;

  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_dst_map_012_axis0 = {0, 1, 2};
    ref_dst_map_021_axis0 = {0, 2, 1};
    ref_dst_map_102_axis0 = {0, 1, 2};
    ref_dst_map_120_axis0 = {0, 1, 2};
    ref_dst_map_201_axis0 = {0, 2, 1};
    ref_dst_map_210_axis0 = {0, 2, 1};

    ref_dst_map_012_axis1 = {1, 0, 2};
    ref_dst_map_021_axis1 = {1, 0, 2};
    ref_dst_map_102_axis1 = {1, 0, 2};
    ref_dst_map_120_axis1 = {1, 2, 0};
    ref_dst_map_201_axis1 = {1, 2, 0};
    ref_dst_map_210_axis1 = {1, 2, 0};

    ref_dst_map_012_axis2 = {2, 0, 1};
    ref_dst_map_021_axis2 = {2, 0, 1};
    ref_dst_map_102_axis2 = {2, 1, 0};
    ref_dst_map_120_axis2 = {2, 1, 0};
    ref_dst_map_201_axis2 = {2, 0, 1};
    ref_dst_map_210_axis2 = {2, 1, 0};

  } else {
    ref_dst_map_012_axis0 = {1, 2, 0};
    ref_dst_map_021_axis0 = {2, 1, 0};
    ref_dst_map_102_axis0 = {1, 2, 0};
    ref_dst_map_120_axis0 = {1, 2, 0};
    ref_dst_map_201_axis0 = {2, 1, 0};
    ref_dst_map_210_axis0 = {2, 1, 0};

    ref_dst_map_012_axis1 = {0, 2, 1};
    ref_dst_map_021_axis1 = {0, 2, 1};
    ref_dst_map_102_axis1 = {0, 2, 1};
    ref_dst_map_120_axis1 = {2, 0, 1};
    ref_dst_map_201_axis1 = {2, 0, 1};
    ref_dst_map_210_axis1 = {2, 0, 1};

    ref_dst_map_012_axis2 = {0, 1, 2};
    ref_dst_map_021_axis2 = {0, 1, 2};
    ref_dst_map_102_axis2 = {1, 0, 2};
    ref_dst_map_120_axis2 = {1, 0, 2};
    ref_dst_map_201_axis2 = {0, 1, 2};
    ref_dst_map_210_axis2 = {1, 0, 2};
  }

  EXPECT_TRUE(dst_map_012_axis0 == ref_dst_map_012_axis0);
  EXPECT_TRUE(dst_map_021_axis0 == ref_dst_map_021_axis0);
  EXPECT_TRUE(dst_map_102_axis0 == ref_dst_map_102_axis0);
  EXPECT_TRUE(dst_map_120_axis0 == ref_dst_map_120_axis0);
  EXPECT_TRUE(dst_map_201_axis0 == ref_dst_map_201_axis0);
  EXPECT_TRUE(dst_map_210_axis0 == ref_dst_map_210_axis0);

  EXPECT_TRUE(dst_map_012_axis1 == ref_dst_map_012_axis1);
  EXPECT_TRUE(dst_map_021_axis1 == ref_dst_map_021_axis1);
  EXPECT_TRUE(dst_map_102_axis1 == ref_dst_map_102_axis1);
  EXPECT_TRUE(dst_map_120_axis1 == ref_dst_map_120_axis1);
  EXPECT_TRUE(dst_map_201_axis1 == ref_dst_map_201_axis1);
  EXPECT_TRUE(dst_map_210_axis1 == ref_dst_map_210_axis1);

  EXPECT_TRUE(dst_map_012_axis2 == ref_dst_map_012_axis2);
  EXPECT_TRUE(dst_map_021_axis2 == ref_dst_map_021_axis2);
  EXPECT_TRUE(dst_map_102_axis2 == ref_dst_map_102_axis2);
  EXPECT_TRUE(dst_map_120_axis2 == ref_dst_map_120_axis2);
  EXPECT_TRUE(dst_map_201_axis2 == ref_dst_map_201_axis2);
  EXPECT_TRUE(dst_map_210_axis2 == ref_dst_map_210_axis2);
}
}  // namespace

TYPED_TEST_SUITE(TestMapping, test_types);

TYPED_TEST(TestMapping, View2D) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;

  test_permute_map_by_axes2D_View2D<value_type, layout_type>();
}

TYPED_TEST(TestMapping, View3D) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;

  test_permute_map_by_axes3D_View3D<value_type, layout_type>();
}
