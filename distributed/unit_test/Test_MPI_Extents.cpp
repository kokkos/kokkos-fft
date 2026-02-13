// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <mpi.h>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_MPI_Extents.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestMPIExtents : public ::testing::Test {
  using layout_type = T;

  std::size_t m_rank   = 0;
  std::size_t m_nprocs = 1;
  std::size_t m_npx    = 1;

  virtual void SetUp() {
    int rank, nprocs;
    ::MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ::MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    m_rank   = rank;
    m_nprocs = nprocs;
    m_npx    = std::sqrt(m_nprocs);

    if (this->m_nprocs > 4) {
      GTEST_SKIP() << "The number of MPI processes should be smaller or equal "
                      "to 4 for MPI extents tests";
    }
  }
};

template <typename LayoutType>
void test_compute_global_extents2D(std::size_t rank, std::size_t nprocs) {
  using extents_type = std::array<std::size_t, 2>;
  using topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, 2, LayoutType>;
  using ViewType = Kokkos::View<double**, execution_space>;

  extents_type array0{1, nprocs}, array1{nprocs, 1};
  topology_type topology0{1, nprocs}, topology1{nprocs, 1};

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0           = gn0;
  const std::size_t n1_t0_quotient  = (gn1 - 1) / nprocs + 1;
  const std::size_t n1_t0_remainder = gn1 - n1_t0_quotient * (nprocs - 1);
  const std::size_t n1_t0 =
      rank != (nprocs - 1) ? n1_t0_quotient : n1_t0_remainder;

  const std::size_t n0_t1_quotient  = (gn0 - 1) / nprocs + 1;
  const std::size_t n0_t1_remainder = gn0 - n0_t1_quotient * (nprocs - 1);
  const std::size_t n0_t1 =
      rank != (nprocs - 1) ? n0_t1_quotient : n0_t1_remainder;
  const std::size_t n1_t1 = gn1;

  ViewType v0("v0", n0_t0, n1_t0);
  ViewType v1("v1", n0_t1, n1_t1);

  extents_type ref_global_shape{gn0, gn1};

  // With array
  {
    auto global_shape_t0 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v0, array0, MPI_COMM_WORLD);
    auto global_shape_t1 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v1, array1, MPI_COMM_WORLD);
    EXPECT_EQ(global_shape_t0, ref_global_shape);
    EXPECT_EQ(global_shape_t1, ref_global_shape);
  }

  // With topology
  {
    auto global_shape_t0 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v0, topology0, MPI_COMM_WORLD);
    auto global_shape_t1 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v1, topology1, MPI_COMM_WORLD);
    EXPECT_EQ(global_shape_t0, ref_global_shape);
    EXPECT_EQ(global_shape_t1, ref_global_shape);
  }
}

template <typename LayoutType>
void test_compute_global_extents3D(std::size_t rank, std::size_t npx,
                                   std::size_t npy) {
  using extents_type = std::array<std::size_t, 3>;
  using topology_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, LayoutType>;
  using ViewType = Kokkos::View<double***, execution_space>;

  extents_type array0{1, npx, npy}, array1{npx, 1, npy};
  topology_type topology0{1, npx, npy}, topology1{npx, 1, npy};

  std::size_t rx = rank / npy, ry = rank % npy;
  std::size_t rx_topo = rx, ry_topo = ry;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    rx_topo = rank % npx;
    ry_topo = rank / npx;
  }

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32, gn2 = 25;
  extents_type ref_global_shape{gn0, gn1, gn2};

  // With array
  {
    const std::size_t n0_t0 = gn0;
    const std::size_t n1_t0 = distribute_extents(gn1, rx, npx);
    const std::size_t n2_t0 = distribute_extents(gn2, ry, npy);

    const std::size_t n0_t1 = distribute_extents(gn0, rx, npx);
    const std::size_t n1_t1 = gn1;
    const std::size_t n2_t1 = distribute_extents(gn2, ry, npy);

    ViewType v0("v0", n0_t0, n1_t0, n2_t0);
    ViewType v1("v1", n0_t1, n1_t1, n2_t1);

    auto global_shape_t0 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v0, array0, MPI_COMM_WORLD);
    auto global_shape_t1 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v1, array1, MPI_COMM_WORLD);

    EXPECT_EQ(global_shape_t0, ref_global_shape);
    EXPECT_EQ(global_shape_t1, ref_global_shape);
  }

  // With topology
  {
    const std::size_t n0_t0 = gn0;
    const std::size_t n1_t0 = distribute_extents(gn1, rx_topo, npx);
    const std::size_t n2_t0 = distribute_extents(gn2, ry_topo, npy);

    const std::size_t n0_t1 = distribute_extents(gn0, rx_topo, npx);
    const std::size_t n1_t1 = gn1;
    const std::size_t n2_t1 = distribute_extents(gn2, ry_topo, npy);

    ViewType v0("v0", n0_t0, n1_t0, n2_t0);
    ViewType v1("v1", n0_t1, n1_t1, n2_t1);
    auto global_shape_t0 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v0, topology0, MPI_COMM_WORLD);
    auto global_shape_t1 = KokkosFFT::Distributed::Impl::compute_global_extents(
        v1, topology1, MPI_COMM_WORLD);

    EXPECT_EQ(global_shape_t0, ref_global_shape);
    EXPECT_EQ(global_shape_t1, ref_global_shape);
  }
}

template <typename LayoutType>
void test_rank_to_coord() {
  using extents_1D_type = std::array<std::size_t, 1>;
  using extents_2D_type = std::array<std::size_t, 2>;
  using extents_3D_type = std::array<std::size_t, 3>;
  using extents_4D_type = std::array<std::size_t, 4>;

  using topology_1D_type =
      KokkosFFT::Distributed::Topology<std::size_t, 1, LayoutType>;
  using topology_2D_type =
      KokkosFFT::Distributed::Topology<std::size_t, 2, LayoutType>;
  using topology_3D_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, LayoutType>;
  using topology_4D_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, LayoutType>;

  topology_1D_type topology1{2};
  topology_2D_type topology2{1, 2}, topology2_2{4, 2};
  topology_3D_type topology3{1, 2, 1}, topology3_2{4, 2, 1};
  topology_4D_type topology4{1, 4, 1, 1}, topology4_2{1, 1, 4, 2};

  extents_1D_type ref_coord1_rank0{0}, ref_coord1_rank1{1};
  extents_2D_type ref_coord2_rank0{0, 0}, ref_coord2_rank1{0, 1};
  extents_2D_type ref_coord2_2_rank0{0, 0}, ref_coord2_2_rank1{0, 1},
      ref_coord2_2_rank2{1, 0}, ref_coord2_2_rank3{1, 1},
      ref_coord2_2_rank4{2, 0}, ref_coord2_2_rank5{2, 1},
      ref_coord2_2_rank6{3, 0}, ref_coord2_2_rank7{3, 1};
  extents_3D_type ref_coord3_rank0{0, 0, 0}, ref_coord3_rank1{0, 1, 0};
  extents_3D_type ref_coord3_2_rank0{0, 0, 0}, ref_coord3_2_rank1{0, 1, 0},
      ref_coord3_2_rank2{1, 0, 0}, ref_coord3_2_rank3{1, 1, 0},
      ref_coord3_2_rank4{2, 0, 0}, ref_coord3_2_rank5{2, 1, 0},
      ref_coord3_2_rank6{3, 0, 0}, ref_coord3_2_rank7{3, 1, 0};
  extents_4D_type ref_coord4_rank0{0, 0, 0, 0}, ref_coord4_rank1{0, 1, 0, 0},
      ref_coord4_rank2{0, 2, 0, 0}, ref_coord4_rank3{0, 3, 0, 0};
  extents_4D_type ref_coord4_2_rank0{0, 0, 0, 0},
      ref_coord4_2_rank1{0, 0, 0, 1}, ref_coord4_2_rank2{0, 0, 1, 0},
      ref_coord4_2_rank3{0, 0, 1, 1}, ref_coord4_2_rank4{0, 0, 2, 0},
      ref_coord4_2_rank5{0, 0, 2, 1}, ref_coord4_2_rank6{0, 0, 3, 0},
      ref_coord4_2_rank7{0, 0, 3, 1};

  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
    ref_coord2_2_rank0 = {0, 0}, ref_coord2_2_rank1 = {1, 0};
    ref_coord2_2_rank2 = {2, 0}, ref_coord2_2_rank3 = {3, 0};
    ref_coord2_2_rank4 = {0, 1}, ref_coord2_2_rank5 = {1, 1};
    ref_coord2_2_rank6 = {2, 1}, ref_coord2_2_rank7 = {3, 1};

    ref_coord3_2_rank0 = {0, 0, 0}, ref_coord3_2_rank1 = {1, 0, 0};
    ref_coord3_2_rank2 = {2, 0, 0}, ref_coord3_2_rank3 = {3, 0, 0};
    ref_coord3_2_rank4 = {0, 1, 0}, ref_coord3_2_rank5 = {1, 1, 0};
    ref_coord3_2_rank6 = {2, 1, 0}, ref_coord3_2_rank7 = {3, 1, 0};

    ref_coord4_2_rank0 = {0, 0, 0, 0}, ref_coord4_2_rank1 = {0, 0, 1, 0};
    ref_coord4_2_rank2 = {0, 0, 2, 0}, ref_coord4_2_rank3 = {0, 0, 3, 0};
    ref_coord4_2_rank4 = {0, 0, 0, 1}, ref_coord4_2_rank5 = {0, 0, 1, 1};
    ref_coord4_2_rank6 = {0, 0, 2, 1}, ref_coord4_2_rank7 = {0, 0, 3, 1};
  }

  auto coord1_rank0   = KokkosFFT::Distributed::rank_to_coord(topology1, 0);
  auto coord1_rank1   = KokkosFFT::Distributed::rank_to_coord(topology1, 1);
  auto coord2_rank0   = KokkosFFT::Distributed::rank_to_coord(topology2, 0);
  auto coord2_rank1   = KokkosFFT::Distributed::rank_to_coord(topology2, 1);
  auto coord2_2_rank0 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 0);
  auto coord2_2_rank1 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 1);
  auto coord2_2_rank2 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 2);
  auto coord2_2_rank3 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 3);
  auto coord2_2_rank4 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 4);
  auto coord2_2_rank5 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 5);
  auto coord2_2_rank6 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 6);
  auto coord2_2_rank7 = KokkosFFT::Distributed::rank_to_coord(topology2_2, 7);

  auto coord3_rank0   = KokkosFFT::Distributed::rank_to_coord(topology3, 0);
  auto coord3_rank1   = KokkosFFT::Distributed::rank_to_coord(topology3, 1);
  auto coord3_2_rank0 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 0);
  auto coord3_2_rank1 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 1);
  auto coord3_2_rank2 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 2);
  auto coord3_2_rank3 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 3);
  auto coord3_2_rank4 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 4);
  auto coord3_2_rank5 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 5);
  auto coord3_2_rank6 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 6);
  auto coord3_2_rank7 = KokkosFFT::Distributed::rank_to_coord(topology3_2, 7);

  auto coord4_rank0   = KokkosFFT::Distributed::rank_to_coord(topology4, 0);
  auto coord4_rank1   = KokkosFFT::Distributed::rank_to_coord(topology4, 1);
  auto coord4_rank2   = KokkosFFT::Distributed::rank_to_coord(topology4, 2);
  auto coord4_rank3   = KokkosFFT::Distributed::rank_to_coord(topology4, 3);
  auto coord4_2_rank0 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 0);
  auto coord4_2_rank1 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 1);
  auto coord4_2_rank2 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 2);
  auto coord4_2_rank3 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 3);
  auto coord4_2_rank4 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 4);
  auto coord4_2_rank5 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 5);
  auto coord4_2_rank6 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 6);
  auto coord4_2_rank7 = KokkosFFT::Distributed::rank_to_coord(topology4_2, 7);

  EXPECT_EQ(coord1_rank0, ref_coord1_rank0);
  EXPECT_EQ(coord1_rank1, ref_coord1_rank1);
  EXPECT_EQ(coord2_rank0, ref_coord2_rank0);
  EXPECT_EQ(coord2_rank1, ref_coord2_rank1);
  EXPECT_EQ(coord2_2_rank0, ref_coord2_2_rank0);
  EXPECT_EQ(coord2_2_rank1, ref_coord2_2_rank1);
  EXPECT_EQ(coord2_2_rank2, ref_coord2_2_rank2);
  EXPECT_EQ(coord2_2_rank3, ref_coord2_2_rank3);
  EXPECT_EQ(coord2_2_rank4, ref_coord2_2_rank4);
  EXPECT_EQ(coord2_2_rank5, ref_coord2_2_rank5);
  EXPECT_EQ(coord2_2_rank6, ref_coord2_2_rank6);
  EXPECT_EQ(coord2_2_rank7, ref_coord2_2_rank7);
  EXPECT_EQ(coord3_rank0, ref_coord3_rank0);
  EXPECT_EQ(coord3_rank1, ref_coord3_rank1);
  EXPECT_EQ(coord3_2_rank0, ref_coord3_2_rank0);
  EXPECT_EQ(coord3_2_rank1, ref_coord3_2_rank1);
  EXPECT_EQ(coord3_2_rank2, ref_coord3_2_rank2);
  EXPECT_EQ(coord3_2_rank3, ref_coord3_2_rank3);
  EXPECT_EQ(coord3_2_rank4, ref_coord3_2_rank4);
  EXPECT_EQ(coord3_2_rank5, ref_coord3_2_rank5);
  EXPECT_EQ(coord3_2_rank6, ref_coord3_2_rank6);
  EXPECT_EQ(coord3_2_rank7, ref_coord3_2_rank7);
  EXPECT_EQ(coord4_rank0, ref_coord4_rank0);
  EXPECT_EQ(coord4_rank1, ref_coord4_rank1);
  EXPECT_EQ(coord4_rank2, ref_coord4_rank2);
  EXPECT_EQ(coord4_rank3, ref_coord4_rank3);
  EXPECT_EQ(coord4_2_rank0, ref_coord4_2_rank0);
  EXPECT_EQ(coord4_2_rank1, ref_coord4_2_rank1);
  EXPECT_EQ(coord4_2_rank2, ref_coord4_2_rank2);
  EXPECT_EQ(coord4_2_rank3, ref_coord4_2_rank3);
  EXPECT_EQ(coord4_2_rank4, ref_coord4_2_rank4);
  EXPECT_EQ(coord4_2_rank5, ref_coord4_2_rank5);
  EXPECT_EQ(coord4_2_rank6, ref_coord4_2_rank6);
  EXPECT_EQ(coord4_2_rank7, ref_coord4_2_rank7);

  // Failure tests: rank out of range
  EXPECT_THROW(
      {
        [[maybe_unused]] auto coord =
            KokkosFFT::Distributed::rank_to_coord(topology1, 2);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto coord =
            KokkosFFT::Distributed::rank_to_coord(topology2, 2);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto coord =
            KokkosFFT::Distributed::rank_to_coord(topology3, 2);
      },
      std::runtime_error);
  EXPECT_THROW(
      {
        [[maybe_unused]] auto coord =
            KokkosFFT::Distributed::rank_to_coord(topology4, 4);
      },
      std::runtime_error);
}

template <typename ContainerType>
void test_compute_global_max(std::size_t rank, std::size_t nprocs) {
  ContainerType local_values{rank + 1, rank + 2, rank + 3};
  const std::size_t global_max =
      KokkosFFT::Distributed::Impl::compute_global_max(local_values,
                                                       MPI_COMM_WORLD);
  EXPECT_EQ(global_max, nprocs + 2);
}

template <typename ContainerType>
void test_compute_global_min(std::size_t rank) {
  ContainerType local_values{rank + 1, rank + 2, rank + 3};
  const std::size_t global_min =
      KokkosFFT::Distributed::Impl::compute_global_min(local_values,
                                                       MPI_COMM_WORLD);
  EXPECT_EQ(global_min, 1);
}

template <typename LayoutType>
void test_compute_local_extent2D(std::size_t rank, std::size_t nprocs) {
  using extents_type  = std::array<std::size_t, 2>;
  using topology_type = std::array<std::size_t, 2>;

  topology_type topology0{1, nprocs};
  topology_type topology1{nprocs, 1};

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, rank, nprocs);
  const std::size_t n0_t1 = distribute_extents(gn0, rank, nprocs);
  const std::size_t n1_t1 = gn1;

  extents_type global_shape{gn0, gn1};
  extents_type ref_local_shape_t0{n0_t0, n1_t0},
      ref_local_shape_t1{n0_t1, n1_t1};
  extents_type ref_local_starts_t0{}, ref_local_starts_t1{};
  for (std::size_t r = 0; r < rank; r++) {
    ref_local_starts_t0.at(1) += distribute_extents(gn1, r, nprocs);
    ref_local_starts_t1.at(0) += distribute_extents(gn0, r, nprocs);
  }

  auto [local_shape_t0, local_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents(global_shape, topology0,
                                                    MPI_COMM_WORLD);
  auto [local_shape_t1, local_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents(global_shape, topology1,
                                                    MPI_COMM_WORLD);

  EXPECT_EQ(local_shape_t0, ref_local_shape_t0);
  EXPECT_EQ(local_shape_t1, ref_local_shape_t1);

  EXPECT_EQ(local_starts_t0, ref_local_starts_t0);
  EXPECT_EQ(local_starts_t1, ref_local_starts_t1);
}

template <typename LayoutType>
void test_compute_local_extents3D(std::size_t rank, std::size_t npx,
                                  std::size_t npy) {
  using extents_type    = std::array<std::size_t, 3>;
  using topology_r_type = KokkosFFT::Distributed::Topology<std::size_t, 3>;

  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology2{npx, npy, 1};

  std::size_t rx = rank / npy, ry = rank % npy;

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 8, gn1 = 7, gn2 = 5;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t0 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t1 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t1 = gn1;
  const std::size_t n2_t1 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t2 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t2 = distribute_extents(gn1, ry, npy);
  const std::size_t n2_t2 = gn2;

  extents_type global_shape{gn0, gn1, gn2};
  extents_type ref_local_shape_t0{n0_t0, n1_t0, n2_t0},
      ref_local_shape_t1{n0_t1, n1_t1, n2_t1},
      ref_local_shape_t2{n0_t2, n1_t2, n2_t2};
  extents_type ref_local_starts_t0{}, ref_local_starts_t1{},
      ref_local_starts_t2{};
  for (std::size_t r = 0; r < rx; r++) {
    ref_local_starts_t0.at(1) += distribute_extents(gn1, r, npx);
    ref_local_starts_t1.at(0) += distribute_extents(gn0, r, npx);
    ref_local_starts_t2.at(0) += distribute_extents(gn0, r, npx);
  }

  for (std::size_t r = 0; r < ry; r++) {
    ref_local_starts_t0.at(2) += distribute_extents(gn2, r, npy);
    ref_local_starts_t1.at(2) += distribute_extents(gn2, r, npy);
    ref_local_starts_t2.at(1) += distribute_extents(gn1, r, npy);
  }

  auto [local_shape_t0, local_starts_t0] =
      KokkosFFT::Distributed::compute_local_extents(global_shape, topology0,
                                                    MPI_COMM_WORLD);
  auto [local_shape_t1, local_starts_t1] =
      KokkosFFT::Distributed::compute_local_extents(global_shape, topology1,
                                                    MPI_COMM_WORLD);
  auto [local_shape_t2, local_starts_t2] =
      KokkosFFT::Distributed::compute_local_extents(global_shape, topology2,
                                                    MPI_COMM_WORLD);

  EXPECT_EQ(local_shape_t0, ref_local_shape_t0);
  EXPECT_EQ(local_shape_t1, ref_local_shape_t1);
  EXPECT_EQ(local_shape_t2, ref_local_shape_t2);

  EXPECT_EQ(local_starts_t0, ref_local_starts_t0);
  EXPECT_EQ(local_starts_t1, ref_local_starts_t1);
  EXPECT_EQ(local_starts_t2, ref_local_starts_t2);
}

template <typename LayoutType>
void test_compute_next_extents2D(std::size_t rank, std::size_t nprocs) {
  using extents_type    = std::array<std::size_t, 2>;
  using topology_r_type = KokkosFFT::Distributed::Topology<std::size_t, 2>;
  using map_type        = std::array<std::size_t, 2>;

  topology_r_type topology0{1, nprocs}, topology1{nprocs, 1};
  map_type map0{0, 1}, map1{1, 0};

  auto distribute_extents = [&](std::size_t n, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return rank < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 19, gn1 = 32;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, nprocs);

  const std::size_t n0_t1 = distribute_extents(gn0, nprocs);
  const std::size_t n1_t1 = gn1;

  extents_type global_shape{gn0, gn1};
  extents_type ref_next_shape_t0_map0{n0_t0, n1_t0},
      ref_next_shape_t0_map1{n1_t0, n0_t0},
      ref_next_shape_t1_map0{n0_t1, n1_t1},
      ref_next_shape_t1_map1{n1_t1, n0_t1};

  auto next_shape_t0_map0 = KokkosFFT::Distributed::Impl::compute_next_extents(
      global_shape, topology0, map0, rank);
  auto next_shape_t0_map1 = KokkosFFT::Distributed::Impl::compute_next_extents(
      global_shape, topology0, map1, rank);
  auto next_shape_t1_map0 = KokkosFFT::Distributed::Impl::compute_next_extents(
      global_shape, topology1, map0, rank);
  auto next_shape_t1_map1 = KokkosFFT::Distributed::Impl::compute_next_extents(
      global_shape, topology1, map1, rank);

  EXPECT_EQ(next_shape_t0_map0, ref_next_shape_t0_map0);
  EXPECT_EQ(next_shape_t0_map1, ref_next_shape_t0_map1);
  EXPECT_EQ(next_shape_t1_map0, ref_next_shape_t1_map0);
  EXPECT_EQ(next_shape_t1_map1, ref_next_shape_t1_map1);
}

template <typename LayoutType>
void test_compute_next_extents3D(std::size_t rank, std::size_t npx,
                                 std::size_t npy) {
  using extents_type = std::array<std::size_t, 3>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topology_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using map_type = std::array<std::size_t, 3>;

  topology_r_type topology0{1, npx, npy}, topology1{npx, 1, npy},
      topology2{npx, npy, 1};
  topology_l_type topology3{npy, npx, 1};
  map_type map012{0, 1, 2}, map021{0, 2, 1}, map102{1, 0, 2}, map120{1, 2, 0},
      map201{2, 0, 1}, map210{2, 1, 0};

  std::size_t rx = rank / npy, ry = rank % npy;

  auto distribute_extents = [&](std::size_t n, std::size_t r, std::size_t t) {
    std::size_t quotient  = n / t;
    std::size_t remainder = n % t;
    return r < remainder ? (quotient + 1) : quotient;
  };

  const std::size_t gn0 = 8, gn1 = 7, gn2 = 5;
  const std::size_t n0_t0 = gn0;
  const std::size_t n1_t0 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t0 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t1 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t1 = gn1;
  const std::size_t n2_t1 = distribute_extents(gn2, ry, npy);

  const std::size_t n0_t2 = distribute_extents(gn0, rx, npx);
  const std::size_t n1_t2 = distribute_extents(gn1, ry, npy);
  const std::size_t n2_t2 = gn2;

  const std::size_t n0_t3 = distribute_extents(gn0, ry, npy);
  const std::size_t n1_t3 = distribute_extents(gn1, rx, npx);
  const std::size_t n2_t3 = gn2;

  extents_type global_shape{gn0, gn1, gn2};
  extents_type ref_next_shape_t0_map012{n0_t0, n1_t0, n2_t0},
      ref_next_shape_t0_map021{n0_t0, n2_t0, n1_t0},
      ref_next_shape_t0_map102{n1_t0, n0_t0, n2_t0},
      ref_next_shape_t0_map120{n1_t0, n2_t0, n0_t0},
      ref_next_shape_t0_map201{n2_t0, n0_t0, n1_t0},
      ref_next_shape_t0_map210{n2_t0, n1_t0, n0_t0},
      ref_next_shape_t1_map012{n0_t1, n1_t1, n2_t1},
      ref_next_shape_t1_map021{n0_t1, n2_t1, n1_t1},
      ref_next_shape_t1_map102{n1_t1, n0_t1, n2_t1},
      ref_next_shape_t1_map120{n1_t1, n2_t1, n0_t1},
      ref_next_shape_t1_map201{n2_t1, n0_t1, n1_t1},
      ref_next_shape_t1_map210{n2_t1, n1_t1, n0_t1},
      ref_next_shape_t2_map012{n0_t2, n1_t2, n2_t2},
      ref_next_shape_t2_map021{n0_t2, n2_t2, n1_t2},
      ref_next_shape_t2_map102{n1_t2, n0_t2, n2_t2},
      ref_next_shape_t2_map120{n1_t2, n2_t2, n0_t2},
      ref_next_shape_t2_map201{n2_t2, n0_t2, n1_t2},
      ref_next_shape_t2_map210{n2_t2, n1_t2, n0_t2},
      ref_next_shape_t3_map012{n0_t3, n1_t3, n2_t3},
      ref_next_shape_t3_map021{n0_t3, n2_t3, n1_t3},
      ref_next_shape_t3_map102{n1_t3, n0_t3, n2_t3},
      ref_next_shape_t3_map120{n1_t3, n2_t3, n0_t3},
      ref_next_shape_t3_map201{n2_t3, n0_t3, n1_t3},
      ref_next_shape_t3_map210{n2_t3, n1_t3, n0_t3};

  auto next_shape_t0_map012 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology0, map012, rank);
  auto next_shape_t0_map021 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology0, map021, rank);
  auto next_shape_t0_map102 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology0, map102, rank);
  auto next_shape_t0_map120 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology0, map120, rank);
  auto next_shape_t0_map201 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology0, map201, rank);
  auto next_shape_t0_map210 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology0, map210, rank);

  auto next_shape_t1_map012 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology1, map012, rank);
  auto next_shape_t1_map021 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology1, map021, rank);
  auto next_shape_t1_map102 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology1, map102, rank);
  auto next_shape_t1_map120 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology1, map120, rank);
  auto next_shape_t1_map201 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology1, map201, rank);
  auto next_shape_t1_map210 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology1, map210, rank);

  auto next_shape_t2_map012 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology2, map012, rank);
  auto next_shape_t2_map021 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology2, map021, rank);
  auto next_shape_t2_map102 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology2, map102, rank);
  auto next_shape_t2_map120 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology2, map120, rank);
  auto next_shape_t2_map201 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology2, map201, rank);
  auto next_shape_t2_map210 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology2, map210, rank);

  auto next_shape_t3_map012 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology3, map012, rank);
  auto next_shape_t3_map021 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology3, map021, rank);
  auto next_shape_t3_map102 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology3, map102, rank);
  auto next_shape_t3_map120 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology3, map120, rank);
  auto next_shape_t3_map201 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology3, map201, rank);
  auto next_shape_t3_map210 =
      KokkosFFT::Distributed::Impl::compute_next_extents(
          global_shape, topology3, map210, rank);

  EXPECT_EQ(next_shape_t0_map012, ref_next_shape_t0_map012);
  EXPECT_EQ(next_shape_t0_map021, ref_next_shape_t0_map021);
  EXPECT_EQ(next_shape_t0_map102, ref_next_shape_t0_map102);
  EXPECT_EQ(next_shape_t0_map120, ref_next_shape_t0_map120);
  EXPECT_EQ(next_shape_t0_map201, ref_next_shape_t0_map201);
  EXPECT_EQ(next_shape_t0_map210, ref_next_shape_t0_map210);

  EXPECT_EQ(next_shape_t1_map012, ref_next_shape_t1_map012);
  EXPECT_EQ(next_shape_t1_map021, ref_next_shape_t1_map021);
  EXPECT_EQ(next_shape_t1_map102, ref_next_shape_t1_map102);
  EXPECT_EQ(next_shape_t1_map120, ref_next_shape_t1_map120);
  EXPECT_EQ(next_shape_t1_map201, ref_next_shape_t1_map201);
  EXPECT_EQ(next_shape_t1_map210, ref_next_shape_t1_map210);

  EXPECT_EQ(next_shape_t2_map012, ref_next_shape_t2_map012);
  EXPECT_EQ(next_shape_t2_map021, ref_next_shape_t2_map021);
  EXPECT_EQ(next_shape_t2_map102, ref_next_shape_t2_map102);
  EXPECT_EQ(next_shape_t2_map120, ref_next_shape_t2_map120);
  EXPECT_EQ(next_shape_t2_map201, ref_next_shape_t2_map201);
  EXPECT_EQ(next_shape_t2_map210, ref_next_shape_t2_map210);

  EXPECT_EQ(next_shape_t3_map012, ref_next_shape_t3_map012);
  EXPECT_EQ(next_shape_t3_map021, ref_next_shape_t3_map021);
  EXPECT_EQ(next_shape_t3_map102, ref_next_shape_t3_map102);
  EXPECT_EQ(next_shape_t3_map120, ref_next_shape_t3_map120);
  EXPECT_EQ(next_shape_t3_map201, ref_next_shape_t3_map201);
  EXPECT_EQ(next_shape_t3_map210, ref_next_shape_t3_map210);
}

}  // namespace

TYPED_TEST_SUITE(TestMPIExtents, test_types);

TYPED_TEST(TestMPIExtents, compute_global_extents2D) {
  using layout_type = typename TestFixture::layout_type;
  test_compute_global_extents2D<layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestMPIExtents, compute_global_extents3D) {
  using layout_type = typename TestFixture::layout_type;

  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_compute_global_extents3D<layout_type>(this->m_rank, this->m_npx,
                                             this->m_npx);
}

TYPED_TEST(TestMPIExtents, rank_to_coord) {
  using layout_type = typename TestFixture::layout_type;
  test_rank_to_coord<layout_type>();
}

TYPED_TEST(TestMPIExtents, compute_global_max_vector) {
  using container_type = std::vector<std::size_t>;
  test_compute_global_max<container_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestMPIExtents, compute_global_max_array) {
  using container_type = std::array<std::size_t, 3>;
  test_compute_global_max<container_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestMPIExtents, compute_global_min_vector) {
  using container_type = std::vector<std::size_t>;
  test_compute_global_min<container_type>(this->m_rank);
}

TYPED_TEST(TestMPIExtents, compute_global_min_array) {
  using container_type = std::array<std::size_t, 3>;
  test_compute_global_min<container_type>(this->m_rank);
}

TYPED_TEST(TestMPIExtents, compute_local_extents2D) {
  using layout_type = typename TestFixture::layout_type;
  test_compute_local_extent2D<layout_type>(this->m_rank, this->m_nprocs);
}

TYPED_TEST(TestMPIExtents, compute_local_extents3D) {
  using layout_type = typename TestFixture::layout_type;
  if (this->m_nprocs == 1 || this->m_npx * this->m_npx != this->m_nprocs) {
    GTEST_SKIP() << "The number of MPI processes should be a perfect square "
                    "for this test";
  }

  test_compute_local_extents3D<layout_type>(this->m_rank, this->m_npx,
                                            this->m_npx);
}

TYPED_TEST(TestMPIExtents, compute_next_extents2D) {
  using layout_type = typename TestFixture::layout_type;

  for (std::size_t nprocs = 1; nprocs <= 6; ++nprocs) {
    for (std::size_t rank = 0; rank < nprocs; ++rank) {
      test_compute_next_extents2D<layout_type>(rank, nprocs);
    }
  }
}

TYPED_TEST(TestMPIExtents, compute_next_extents3D) {
  using layout_type = typename TestFixture::layout_type;

  for (std::size_t npx = 1; npx <= 3; ++npx) {
    for (std::size_t npy = 1; npy <= 3; ++npy) {
      for (std::size_t rank = 0; rank < npx * npy; ++rank) {
        test_compute_next_extents3D<layout_type>(rank, npx, npy);
      }
    }
  }
}
