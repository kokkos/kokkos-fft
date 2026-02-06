// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_Extents.hpp"

namespace {
using test_types =
    ::testing::Types<std::pair<int, Kokkos::LayoutLeft>,
                     std::pair<int, Kokkos::LayoutRight>,
                     std::pair<std::size_t, Kokkos::LayoutLeft>,
                     std::pair<std::size_t, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestExtents : public ::testing::Test {
  using value_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

template <typename IntType, typename LayoutType>
void test_buffer_extents() {
  using extents_type        = std::array<IntType, 4>;
  using buffer_extents_type = std::array<IntType, 5>;
  using topology_type       = std::array<IntType, 4>;
  const IntType n0 = 13, n1 = 8, n2 = 17, n3 = 5;
  const IntType p0 = 2, p1 = 3;

  // Global View
  extents_type extents{n0, n1, n2, n3};

  // X-pencil
  topology_type topology0 = {1, p0, p1, 1};

  // Y-pencil
  topology_type topology1 = {p0, 1, p1, 1};

  // Z-pencil
  topology_type topology2 = {p0, p1, 1, 1};

  buffer_extents_type ref_buffer_01, ref_buffer_12;
  if (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_buffer_01 = {(n0 - 1) / p0 + 1, (n1 - 1) / p0 + 1, (n2 - 1) / p1 + 1,
                     n3, p0};
    ref_buffer_12 = {(n0 - 1) / p0 + 1, (n1 - 1) / p1 + 1, (n2 - 1) / p1 + 1,
                     n3, p1};
  } else {
    ref_buffer_01 = {p0, (n0 - 1) / p0 + 1, (n1 - 1) / p0 + 1,
                     (n2 - 1) / p1 + 1, n3};
    ref_buffer_12 = {p1, (n0 - 1) / p0 + 1, (n1 - 1) / p1 + 1,
                     (n2 - 1) / p1 + 1, n3};
  }

  buffer_extents_type buffer_01 =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          extents, topology0, topology1);
  buffer_extents_type buffer_12 =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          extents, topology1, topology2);

  EXPECT_TRUE(buffer_01 == ref_buffer_01);
  EXPECT_TRUE(buffer_12 == ref_buffer_12);

  // In valid, because you cannot go from X to Z in one exchange
  EXPECT_THROW(
      {
        [[maybe_unused]] buffer_extents_type buffer_02 =
            KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
                extents, topology0, topology2);
      },
      std::runtime_error);
}

template <typename ContainerType, typename iType>
void test_compute_mapped_extents(iType nprocs) {
  using extents_type   = std::array<iType, 3>;
  extents_type extents = {nprocs, 3, 8};
  ContainerType map012 = {0, 1, 2}, map021 = {0, 2, 1}, map102 = {1, 0, 2},
                map120 = {1, 2, 0}, map201 = {2, 0, 1}, map210 = {2, 1, 0};
  auto mapped_extents012 =
      KokkosFFT::Distributed::Impl::compute_mapped_extents(extents, map012);
  auto mapped_extents021 =
      KokkosFFT::Distributed::Impl::compute_mapped_extents(extents, map021);
  auto mapped_extents102 =
      KokkosFFT::Distributed::Impl::compute_mapped_extents(extents, map102);
  auto mapped_extents120 =
      KokkosFFT::Distributed::Impl::compute_mapped_extents(extents, map120);
  auto mapped_extents201 =
      KokkosFFT::Distributed::Impl::compute_mapped_extents(extents, map201);
  auto mapped_extents210 =
      KokkosFFT::Distributed::Impl::compute_mapped_extents(extents, map210);

  extents_type ref_mapped_extents012 = {nprocs, 3, 8},
               ref_mapped_extents021 = {nprocs, 8, 3},
               ref_mapped_extents102 = {3, nprocs, 8},
               ref_mapped_extents120 = {3, 8, nprocs},
               ref_mapped_extents201 = {8, nprocs, 3},
               ref_mapped_extents210 = {8, 3, nprocs};

  EXPECT_EQ(mapped_extents012, ref_mapped_extents012);
  EXPECT_EQ(mapped_extents021, ref_mapped_extents021);
  EXPECT_EQ(mapped_extents102, ref_mapped_extents102);
  EXPECT_EQ(mapped_extents120, ref_mapped_extents120);
  EXPECT_EQ(mapped_extents201, ref_mapped_extents201);
  EXPECT_EQ(mapped_extents210, ref_mapped_extents210);
}

template <typename ContainerType, typename iType>
void test_compute_fft_extents(iType nprocs) {
  using extents_type      = std::array<iType, 3>;
  extents_type in_extents = {nprocs, 3, 8}, out_extents = {nprocs, 3, 5};
  ContainerType map012 = {0, 1, 2}, map021 = {0, 2, 1}, map102 = {1, 0, 2},
                map120 = {1, 2, 0}, map201 = {2, 0, 1}, map210 = {2, 1, 0};
  auto fft_extents012 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map012);
  auto fft_extents021 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map021);
  auto fft_extents102 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map102);
  auto fft_extents120 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map120);
  auto fft_extents201 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map201);
  auto fft_extents210 = KokkosFFT::Distributed::Impl::compute_fft_extents(
      in_extents, out_extents, map210);

  extents_type ref_fft_extents012 = {nprocs, 3, 8},
               ref_fft_extents021 = {nprocs, 8, 3},
               ref_fft_extents102 = {3, nprocs, 8},
               ref_fft_extents120 = {3, 8, nprocs},
               ref_fft_extents201 = {8, nprocs, 3},
               ref_fft_extents210 = {8, 3, nprocs};

  EXPECT_EQ(fft_extents012, ref_fft_extents012);
  EXPECT_EQ(fft_extents021, ref_fft_extents021);
  EXPECT_EQ(fft_extents102, ref_fft_extents102);
  EXPECT_EQ(fft_extents120, ref_fft_extents120);
  EXPECT_EQ(fft_extents201, ref_fft_extents201);
  EXPECT_EQ(fft_extents210, ref_fft_extents210);
}

}  // namespace

TYPED_TEST_SUITE(TestExtents, test_types);

TYPED_TEST(TestExtents, BufferExtents) {
  using value_type  = typename TestFixture::value_type;
  using layout_type = typename TestFixture::layout_type;

  test_buffer_extents<value_type, layout_type>();
}

TYPED_TEST(TestExtents, mapped_extents_of_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = std::vector<value_type>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_compute_mapped_extents<vector_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestExtents, mapped_extents_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_compute_mapped_extents<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestExtents, fft_extents_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_compute_fft_extents<array_type, value_type>(nprocs);
  }
}
