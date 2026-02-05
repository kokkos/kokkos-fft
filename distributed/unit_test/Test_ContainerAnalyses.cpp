// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_ContainerAnalyses.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using base_int_types  = ::testing::Types<int, std::size_t>;

template <typename T>
struct TestContainerTypes : public ::testing::Test {
  static constexpr std::size_t rank = 5;
  using value_type                  = T;
  using vector_type                 = std::vector<T>;
  using array_type                  = std::array<T, rank>;
};

template <typename ContainerType, typename iType>
void test_extract_different_indices(iType nprocs) {
  ContainerType topology0 = {nprocs, 1, 8};
  ContainerType topology1 = {nprocs, 8, 1};
  ContainerType topology2 = {8, nprocs, 1};
  ContainerType topology3 = {nprocs, 1, 1};

  if (nprocs == 1) {
    auto diff00 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology0);
    auto diff01 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology1);
    auto diff02 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology2);
    auto diff03 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology3);
    auto diff10 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology0);
    auto diff12 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology2);
    auto diff20 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology0);
    auto diff21 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology1);

    std::vector<std::size_t> ref_diff00 = {};
    std::vector<std::size_t> ref_diff01 = {1, 2};
    std::vector<std::size_t> ref_diff02 = {0, 2};
    std::vector<std::size_t> ref_diff03 = {2};
    std::vector<std::size_t> ref_diff10 = {1, 2};
    std::vector<std::size_t> ref_diff12 = {0, 1};
    std::vector<std::size_t> ref_diff20 = {0, 2};
    std::vector<std::size_t> ref_diff21 = {0, 1};

    EXPECT_EQ(diff00, ref_diff00);
    EXPECT_EQ(diff01, ref_diff01);
    EXPECT_EQ(diff02, ref_diff02);
    EXPECT_EQ(diff03, ref_diff03);
    EXPECT_EQ(diff10, ref_diff10);
    EXPECT_EQ(diff12, ref_diff12);
    EXPECT_EQ(diff20, ref_diff20);
    EXPECT_EQ(diff21, ref_diff21);
  } else {
    auto diff00 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology0);
    auto diff01 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology1);
    auto diff02 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology2);
    auto diff03 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology0, topology3);
    auto diff10 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology0);
    auto diff12 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology1, topology2);
    auto diff20 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology0);
    auto diff21 = KokkosFFT::Distributed::Impl::extract_different_indices(
        topology2, topology1);

    std::vector<std::size_t> ref_diff00 = {};
    std::vector<std::size_t> ref_diff01 = {1, 2};
    std::vector<std::size_t> ref_diff02 = {0, 1, 2};
    std::vector<std::size_t> ref_diff03 = {2};
    std::vector<std::size_t> ref_diff10 = {1, 2};
    std::vector<std::size_t> ref_diff12 = {0, 1};
    std::vector<std::size_t> ref_diff20 = {0, 1, 2};
    std::vector<std::size_t> ref_diff21 = {0, 1};

    EXPECT_EQ(diff00, ref_diff00);
    EXPECT_EQ(diff01, ref_diff01);
    EXPECT_EQ(diff02, ref_diff02);
    EXPECT_EQ(diff03, ref_diff03);
    EXPECT_EQ(diff10, ref_diff10);
    EXPECT_EQ(diff12, ref_diff12);
    EXPECT_EQ(diff20, ref_diff20);
    EXPECT_EQ(diff21, ref_diff21);
  }
}

template <typename ContainerType, typename iType>
void test_extract_different_value_set(iType nprocs) {
  ContainerType topology0 = {nprocs, 1, 8};
  ContainerType topology1 = {nprocs, 8, 1};
  ContainerType topology2 = {8, nprocs, 1};

  auto diff01 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology0, topology1);
  auto diff02 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology0, topology2);
  auto diff10 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology1, topology0);
  auto diff12 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology1, topology2);
  auto diff20 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology2, topology0);
  auto diff21 = KokkosFFT::Distributed::Impl::extract_different_value_set(
      topology2, topology1);

  std::set<iType> ref_diff =
      (nprocs == 1) ? std::set<iType>{1, 8} : std::set<iType>{1, nprocs, 8};
  EXPECT_EQ(diff01, ref_diff);
  EXPECT_EQ(diff02, ref_diff);
  EXPECT_EQ(diff10, ref_diff);
  EXPECT_EQ(diff12, ref_diff);
  EXPECT_EQ(diff20, ref_diff);
  EXPECT_EQ(diff21, ref_diff);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_count_non_ones() {
  ContainerType0 v0 = {0, 1, 4, 2, 3};
  ContainerType1 v1 = {2, 3, 5};
  ContainerType2 v2 = {1};

  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_non_ones(v0), 4);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_non_ones(v1), 3);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::count_non_ones(v2), 0);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2>
void test_extract_non_one_indices() {
  ContainerType0 a0 = {1, 1, 4, 2, 3}, b0 = {1, 4, 2, 3, 1};
  ContainerType1 a1 = {2, 3, 5}, b1 = {5, 3, 2};
  ContainerType2 a2 = {1}, b2 = {1};

  std::vector<std::size_t> ref_diff_00 = {1, 2, 3, 4};
  std::vector<std::size_t> ref_diff_11 = {0, 1, 2};
  std::vector<std::size_t> ref_diff_22 = {};

  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_indices(a0, b0),
            ref_diff_00);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_indices(a1, b1),
            ref_diff_11);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_indices(a2, b2),
            ref_diff_22);
}

template <typename ContainerType0, typename ContainerType1,
          typename ContainerType2, typename iType>
void test_extract_non_one_values() {
  ContainerType0 a0 = {1, 1, 4, 2, 3};
  ContainerType1 a1 = {2, 3, 5};
  ContainerType2 a2 = {1};

  std::vector<iType> ref_diff_0 = {4, 2, 3};
  std::vector<iType> ref_diff_1 = {2, 3, 5};
  std::vector<iType> ref_diff_2 = {};

  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_values(a0),
            ref_diff_0);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_values(a1),
            ref_diff_1);
  EXPECT_EQ(KokkosFFT::Distributed::Impl::extract_non_one_values(a2),
            ref_diff_2);
}

template <typename ContainerType0, typename ContainerType1, typename iType>
void test_has_identical_non_ones(iType nprocs) {
  iType np0                = 4;
  ContainerType0 non_ones0 = {np0, nprocs, 2};  // The length 3 input is invalid
  ContainerType1 non_ones1 = {np0, nprocs}, ones = {1, 1};

  auto has_identical_non_ones0 =
      KokkosFFT::Distributed::Impl::has_identical_non_ones(non_ones0);
  auto has_identical_non_ones1 =
      KokkosFFT::Distributed::Impl::has_identical_non_ones(non_ones1);
  auto has_identical_non_ones2 =
      KokkosFFT::Distributed::Impl::has_identical_non_ones(ones);

  EXPECT_FALSE(has_identical_non_ones0);
  EXPECT_FALSE(has_identical_non_ones2);
  if (nprocs == 1) {
    EXPECT_FALSE(has_identical_non_ones1);
  } else {
    if (nprocs == np0) {
      EXPECT_TRUE(has_identical_non_ones1);
    } else {
      EXPECT_FALSE(has_identical_non_ones1);
    }
  }
}

template <typename ContainerType, typename iType>
void test_swap_elements(iType nprocs) {
  ContainerType topology = {nprocs, 1, 8};

  auto swapped_01 = KokkosFFT::Distributed::Impl::swap_elements(topology, 0, 1);
  auto swapped_02 = KokkosFFT::Distributed::Impl::swap_elements(topology, 0, 2);
  auto swapped_10 = KokkosFFT::Distributed::Impl::swap_elements(topology, 1, 0);
  auto swapped_12 = KokkosFFT::Distributed::Impl::swap_elements(topology, 1, 2);
  auto swapped_20 = KokkosFFT::Distributed::Impl::swap_elements(topology, 2, 0);
  auto swapped_21 = KokkosFFT::Distributed::Impl::swap_elements(topology, 2, 1);

  ContainerType ref_swapped_01 = {1, nprocs, 8};
  ContainerType ref_swapped_02 = {8, 1, nprocs};
  ContainerType ref_swapped_10 = {1, nprocs, 8};
  ContainerType ref_swapped_12 = {nprocs, 8, 1};
  ContainerType ref_swapped_20 = {8, 1, nprocs};
  ContainerType ref_swapped_21 = {nprocs, 8, 1};

  EXPECT_EQ(swapped_01, ref_swapped_01);
  EXPECT_EQ(swapped_02, ref_swapped_02);
  EXPECT_EQ(swapped_10, ref_swapped_10);
  EXPECT_EQ(swapped_12, ref_swapped_12);
  EXPECT_EQ(swapped_20, ref_swapped_20);
  EXPECT_EQ(swapped_21, ref_swapped_21);
}

template <typename ContainerType, typename iType>
void test_merge_topology(iType nprocs) {
  ContainerType topology0 = {1, 1, nprocs};
  ContainerType topology1 = {1, nprocs, 1};
  ContainerType topology2 = {nprocs, 1, 1};
  ContainerType topology3 = {nprocs, 1, 8};
  ContainerType topology4 = {nprocs, 8, 1};

  auto merged01 =
      KokkosFFT::Distributed::Impl::merge_topology(topology0, topology1);
  auto merged02 =
      KokkosFFT::Distributed::Impl::merge_topology(topology0, topology2);
  auto merged12 =
      KokkosFFT::Distributed::Impl::merge_topology(topology1, topology2);
  ContainerType ref_merged01 = {1, nprocs, nprocs};
  ContainerType ref_merged02 = {nprocs, 1, nprocs};
  ContainerType ref_merged12 = {nprocs, nprocs, 1};
  EXPECT_EQ(merged01, ref_merged01);
  EXPECT_EQ(merged02, ref_merged02);
  EXPECT_EQ(merged12, ref_merged12);

  // Failure tests because these do not have same size
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged03 =
            KokkosFFT::Distributed::Impl::merge_topology(topology0, topology3);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged04 =
            KokkosFFT::Distributed::Impl::merge_topology(topology0, topology4);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged13 =
            KokkosFFT::Distributed::Impl::merge_topology(topology1, topology3);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged14 =
            KokkosFFT::Distributed::Impl::merge_topology(topology1, topology4);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged23 =
            KokkosFFT::Distributed::Impl::merge_topology(topology2, topology3);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto merged24 =
            KokkosFFT::Distributed::Impl::merge_topology(topology2, topology4);
      },
      std::runtime_error);

  // This case is valid
  ContainerType ref_merged34 = {nprocs, 8, 8};
  auto merged34 =
      KokkosFFT::Distributed::Impl::merge_topology(topology3, topology4);
  EXPECT_EQ(merged34, ref_merged34);
}

template <typename ContainerType, typename iType>
void test_diff_topology(iType nprocs) {
  ContainerType topology0  = {1, 1, nprocs};
  ContainerType topology1  = {1, nprocs, 1};
  ContainerType topology2  = {nprocs, 1, 1};
  ContainerType topology3  = {nprocs, 1, 8};
  ContainerType topology01 = {1, nprocs, nprocs};
  ContainerType topology02 = {nprocs, 1, nprocs};
  ContainerType topology12 = {nprocs, nprocs, 1};

  iType diff0_01 =
      KokkosFFT::Distributed::Impl::diff_topology(topology0, topology01);
  iType diff0_02 =
      KokkosFFT::Distributed::Impl::diff_topology(topology0, topology02);
  iType diff1_12 =
      KokkosFFT::Distributed::Impl::diff_topology(topology1, topology12);
  iType diff2_12 =
      KokkosFFT::Distributed::Impl::diff_topology(topology2, topology12);

  iType ref_diff = nprocs;

  EXPECT_EQ(diff0_01, ref_diff);
  EXPECT_EQ(diff0_02, ref_diff);
  EXPECT_EQ(diff1_12, ref_diff);
  EXPECT_EQ(diff2_12, ref_diff);

  if (nprocs == 1) {
    iType diff03 =
        KokkosFFT::Distributed::Impl::diff_topology(topology0, topology3);
    iType ref_diff03 = topology3.at(2);
    EXPECT_EQ(diff03, ref_diff03);
  } else {
    // Failure tests because more than two elements are different
    EXPECT_THROW(
        {
          [[maybe_unused]] iType diff03 =
              KokkosFFT::Distributed::Impl::diff_topology(topology0, topology3);
        },
        std::runtime_error);
  }
}
}  // namespace

TYPED_TEST_SUITE(TestContainerTypes, base_int_types);

TYPED_TEST(TestContainerTypes, find_different_indices_of_vector) {
  using value_type     = typename TestFixture::value_type;
  using container_type = typename TestFixture::vector_type;

  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_indices<container_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, find_different_indices_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_indices<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, find_different_value_set_of_vector) {
  using value_type  = typename TestFixture::value_type;
  using vector_type = typename TestFixture::vector_type;

  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_value_set<vector_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, find_different_value_set_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_extract_different_value_set<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_count_non_ones_of_vector) {
  using container_type = typename TestFixture::vector_type;
  test_count_non_ones<container_type, container_type, container_type>();
}

TYPED_TEST(TestContainerTypes, test_count_non_ones_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_count_non_ones<container_type0, container_type1, container_type2>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_indices_of_vector) {
  using container_type = typename TestFixture::vector_type;
  test_extract_non_one_indices<container_type, container_type,
                               container_type>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_indices_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_extract_non_one_indices<container_type0, container_type1,
                               container_type2>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_values_of_vector) {
  using container_type = typename TestFixture::vector_type;
  using value_type     = typename TestFixture::value_type;
  test_extract_non_one_values<container_type, container_type, container_type,
                              value_type>();
}

TYPED_TEST(TestContainerTypes, test_extract_non_one_values_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 5>;
  using container_type1 = std::array<value_type, 3>;
  using container_type2 = std::array<value_type, 1>;
  test_extract_non_one_values<container_type0, container_type1, container_type2,
                              value_type>();
}

TYPED_TEST(TestContainerTypes, test_has_identical_non_ones_of_vector) {
  using container_type = typename TestFixture::vector_type;
  using value_type     = typename TestFixture::value_type;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_has_identical_non_ones<container_type, container_type, value_type>(
        nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_has_identical_non_ones_of_array) {
  using value_type      = typename TestFixture::value_type;
  using container_type0 = std::array<value_type, 3>;
  using container_type1 = std::array<value_type, 2>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_has_identical_non_ones<container_type0, container_type1, value_type>(
        nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_swap_elements_of_vector) {
  using container_type = typename TestFixture::vector_type;
  using value_type     = typename TestFixture::value_type;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_swap_elements<container_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_swap_elements_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_swap_elements<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_merge_topology_of_vector) {
  using container_type = typename TestFixture::vector_type;
  using value_type     = typename TestFixture::value_type;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_merge_topology<container_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_merge_topology_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_merge_topology<array_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_diff_topology_of_vector) {
  using container_type = typename TestFixture::vector_type;
  using value_type     = typename TestFixture::value_type;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_diff_topology<container_type, value_type>(nprocs);
  }
}

TYPED_TEST(TestContainerTypes, test_diff_topology_of_array) {
  using value_type = typename TestFixture::value_type;
  using array_type = std::array<value_type, 3>;
  for (value_type nprocs = 1; nprocs <= 6; ++nprocs) {
    test_diff_topology<array_type, value_type>(nprocs);
  }
}
