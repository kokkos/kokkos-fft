// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Distributed_PackUnpack.hpp"
#include "KokkosFFT_Distributed_Extents.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestUnpack : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_max_nprocs = 4;
};

/// \brief Test unpack function for 2D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] rank MPI rank
/// \param[in] nprocs Number of MPI ranks
/// \param[in] order Order of the dimensions (0: (n0, n1); 1: (n1, n0))
///
/// 0. map = {0, 1}: the source view is (n0, n1/np) for x-pencil and (n0/np, n1)
/// for y-pencil. 0.0 LayoutLeft: The sendbuffer is (n0/np, n1/np, np) for both
/// pencils. 0.1 LayoutRight: The sendbuffer is (np, n0/np, n1/np) for both
/// pencils.
///
/// 1. map = {1, 0}: the source view is (n1/np, n0) for x-pencil and (n1, n0/np)
/// for y-pencil. 1.0 LayoutLeft: The sendbuffer is (n0/np, n1/np, np) for both
/// pencils. 1.1 LayoutRight: The sendbuffer is (np, n0/np, n1/np) for both
/// pencils.
template <typename T, typename LayoutType>
void test_unpack_view2D(std::size_t rank, std::size_t nprocs, int order = 0) {
  using SrcView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using DstView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using map_type      = std::array<int, 2>;
  using shape_type    = std::array<std::size_t, 2>;

  const std::size_t gn0 = 8, gn1 = 7;

  const auto [n0_start, n0_length] = distribute_extents(gn0, rank, nprocs);
  const auto [n1_start, n1_length] = distribute_extents(gn1, rank, nprocs);

  shape_type global_extents   = {gn0, gn1},
             local_extents_t0 = shape_type{gn0, n1_length},
             local_extents_t1 = shape_type{n0_length, gn1};

  shape_type topology0 = {1, nprocs}, topology1 = {nprocs, 1};

  shape_type dst_map = (order == 0) ? shape_type({0, 1}) : shape_type({1, 0});
  map_type int_map   = (order == 0) ? map_type({0, 1}) : map_type({1, 0});

  // Create global and local views
  SrcView2DType gu("gu", gn0, gn1);

  // Data in Topology 0 (X-pencil): original and permuted data
  SrcView2DType u_t0(
      "u_t0", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t0)),
      u_p_t0("u_p_t0", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t0, dst_map))),
      u_p_t0_ref("u_p_t0_ref",
                 KokkosFFT::Impl::create_layout<LayoutType>(
                     KokkosFFT::Distributed::Impl::compute_mapped_extents(
                         local_extents_t0, dst_map)));

  // Data in Topology 1 (Y-pencil): original and permuted data
  SrcView2DType u_t1(
      "u_t1", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t1)),
      u_p_t1("u_p_t1", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t1, dst_map))),
      u_p_t1_ref("u_p_t1_ref",
                 KokkosFFT::Impl::create_layout<LayoutType>(
                     KokkosFFT::Distributed::Impl::compute_mapped_extents(
                         local_extents_t1, dst_map)));

  // Buffers
  auto buffer_extents =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          global_extents, topology0, topology1);
  DstView3DType recv_t0(
      "recv_t0", KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents)),
      recv_t1("recv_t1",
              KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents));

  // Initialize input views and references without considering permutation
  auto h_gu = Kokkos::create_mirror_view(gu);
  for (std::size_t i1 = 0; i1 < gu.extent(1); i1++) {
    for (std::size_t i0 = 0; i0 < gu.extent(0); i0++) {
      h_gu(i0, i1) = static_cast<T>(i1 + i0 * gn1);
    }
  }

  // Copy global source to local source and recv buffers
  Kokkos::pair<std::size_t, std::size_t> range_gu_t0(n1_start,
                                                     n1_start + n1_length),
      range_gu_t1(n0_start, n0_start + n0_length);

  auto h_u_t0 = Kokkos::create_mirror_view(u_t0);
  auto h_u_t1 = Kokkos::create_mirror_view(u_t1);

  auto sub_h_gu_t0 = Kokkos::subview(h_gu, Kokkos::ALL, range_gu_t0);
  auto sub_h_gu_t1 = Kokkos::subview(h_gu, range_gu_t1, Kokkos::ALL);
  Kokkos::deep_copy(h_u_t0, sub_h_gu_t0);
  Kokkos::deep_copy(h_u_t1, sub_h_gu_t1);

  auto h_recv_t0 = Kokkos::create_mirror_view(recv_t0);
  auto h_recv_t1 = Kokkos::create_mirror_view(recv_t1);

  for (std::size_t i2 = 0; i2 < recv_t0.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < recv_t0.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < recv_t0.extent(0); i0++) {
        std::size_t i0_tmp = i0, i1_tmp = i1;
        std::size_t p = 0;
        if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
          p = i2;
        } else {
          i0_tmp = i1, i1_tmp = i2;
          p = i0;
        }

        const auto [start0, length0] = distribute_extents(gn0, p, nprocs);
        const auto [start1, length1] = distribute_extents(gn1, p, nprocs);
        std::size_t gi0              = i0_tmp + start0;
        std::size_t gi1              = i1_tmp + start1;
        if (gi0 < gn0 && i0_tmp < length0 && i1_tmp < h_u_t0.extent(1)) {
          h_recv_t0(i0, i1, i2) = h_u_t0(gi0, i1_tmp);
        }
        if (gi1 < gn1 && i1_tmp < length1 && i0_tmp < h_u_t1.extent(0)) {
          h_recv_t1(i0, i1, i2) = h_u_t1(i0_tmp, gi1);
        }
      }
    }
  }
  Kokkos::deep_copy(u_t0, h_u_t0);
  Kokkos::deep_copy(u_t1, h_u_t1);
  Kokkos::deep_copy(recv_t0, h_recv_t0);
  Kokkos::deep_copy(recv_t1, h_recv_t1);

  // Make permuted local views with safe_transpose
  execution_space exec;
  KokkosFFT::Impl::transpose(exec, u_t0, u_p_t0_ref, int_map, true);
  KokkosFFT::Impl::transpose(exec, u_t1, u_p_t1_ref, int_map, true);
  exec.fence();

  // Apply unpack kernel
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t0, u_p_t0, dst_map, 0);
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t1, u_p_t1, dst_map, 1);

  auto h_u_p_t0 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t0);
  auto h_u_p_t1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t1);
  auto h_u_p_t0_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t0_ref);
  auto h_u_p_t1_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t1_ref);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check u_p_t0 is correct
  for (std::size_t i1 = 0; i1 < u_p_t0.extent(1); i1++) {
    for (std::size_t i0 = 0; i0 < u_p_t0.extent(0); i0++) {
      auto diff = Kokkos::abs(h_u_p_t0(i0, i1) - h_u_p_t0_ref(i0, i1));
      EXPECT_LE(diff, epsilon);
    }
  }

  // Check u_p_t1 is correct
  for (std::size_t i1 = 0; i1 < u_p_t1.extent(1); i1++) {
    for (std::size_t i0 = 0; i0 < u_p_t1.extent(0); i0++) {
      auto diff = Kokkos::abs(h_u_p_t1(i0, i1) - h_u_p_t1_ref(i0, i1));
      EXPECT_LE(diff, epsilon);
    }
  }
}

/// \brief Test unpack function for 3D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] rank MPI rank
/// \param[in] nprocs Number of MPI ranks
/// \param[in] order Order of the dimensions
/// 0: (n0, n1, n2), 1: (n0, n2, n1), 2: (n1, n0, n2)
/// 3: (n1, n2, n0), 4: (n2, n0, n1), 5: (n2, n1, n0)
///
template <typename T, typename LayoutType>
void test_unpack_view3D(std::size_t rank, std::size_t nprocs, int order = 0) {
  using SrcView3DType = Kokkos::View<T***, LayoutType, execution_space>;
  using DstView4DType = Kokkos::View<T****, LayoutType, execution_space>;
  using map_type      = std::array<int, 3>;
  using shape_type    = std::array<std::size_t, 3>;

  const std::size_t gn0 = 8, gn1 = 7, gn2 = 5;

  const auto [n0_start, n0_length] = distribute_extents(gn0, rank, nprocs);
  const auto [n1_start, n1_length] = distribute_extents(gn1, rank, nprocs);
  const auto [n2_start, n2_length] = distribute_extents(gn2, rank, nprocs);

  shape_type global_extents   = {gn0, gn1, gn2},
             local_extents_t0 = shape_type{gn0, gn1, n2_length},
             local_extents_t1 = shape_type{gn0, n1_length, gn2},
             local_extents_t2 = shape_type{n0_length, gn1, gn2};

  shape_type topology0 = {1, 1, nprocs}, topology1 = {1, nprocs, 1},
             topology2 = {nprocs, 1, 1};

  shape_type dst_map = (order == 0)   ? shape_type({0, 1, 2})
                       : (order == 1) ? shape_type({0, 2, 1})
                       : (order == 2) ? shape_type({1, 0, 2})
                       : (order == 3) ? shape_type({1, 2, 0})
                       : (order == 4) ? shape_type({2, 0, 1})
                                      : shape_type({2, 1, 0});

  map_type int_map = (order == 0)   ? map_type({0, 1, 2})
                     : (order == 1) ? map_type({0, 2, 1})
                     : (order == 2) ? map_type({1, 0, 2})
                     : (order == 3) ? map_type({1, 2, 0})
                     : (order == 4) ? map_type({2, 0, 1})
                                    : map_type({2, 1, 0});

  // Create global and local views
  SrcView3DType gu("gu", gn0, gn1, gn2);

  // Data in Topology 0 (Z-slab): original and permuted data
  SrcView3DType u_t0(
      "u_t0", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t0)),
      u_p_t0_01("u_p_t0_01",
                KokkosFFT::Impl::create_layout<LayoutType>(
                    KokkosFFT::Distributed::Impl::compute_mapped_extents(
                        local_extents_t0, dst_map))),
      u_p_t0_02("u_p_t0_02",
                KokkosFFT::Impl::create_layout<LayoutType>(
                    KokkosFFT::Distributed::Impl::compute_mapped_extents(
                        local_extents_t0, dst_map))),
      u_p_t0_ref("u_p_t0_ref",
                 KokkosFFT::Impl::create_layout<LayoutType>(
                     KokkosFFT::Distributed::Impl::compute_mapped_extents(
                         local_extents_t0, dst_map)));

  // Data in Topology 1 (Y-slab): original and permuted data
  SrcView3DType u_t1(
      "u_t1", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t1)),
      u_p_t1_10("u_p_t1_10",
                KokkosFFT::Impl::create_layout<LayoutType>(
                    KokkosFFT::Distributed::Impl::compute_mapped_extents(
                        local_extents_t1, dst_map))),
      u_p_t1_12("u_p_t1_12",
                KokkosFFT::Impl::create_layout<LayoutType>(
                    KokkosFFT::Distributed::Impl::compute_mapped_extents(
                        local_extents_t1, dst_map))),
      u_p_t1_ref("u_p_t1_ref",
                 KokkosFFT::Impl::create_layout<LayoutType>(
                     KokkosFFT::Distributed::Impl::compute_mapped_extents(
                         local_extents_t1, dst_map)));

  // Data in Topology 2 (X-slab): original and permuted data
  SrcView3DType u_t2(
      "u_t2", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t2)),
      u_p_t2_20("u_p_t2_20",
                KokkosFFT::Impl::create_layout<LayoutType>(
                    KokkosFFT::Distributed::Impl::compute_mapped_extents(
                        local_extents_t2, dst_map))),
      u_p_t2_21("u_p_t2_21",
                KokkosFFT::Impl::create_layout<LayoutType>(
                    KokkosFFT::Distributed::Impl::compute_mapped_extents(
                        local_extents_t2, dst_map))),
      u_p_t2_ref("u_p_t2_ref",
                 KokkosFFT::Impl::create_layout<LayoutType>(
                     KokkosFFT::Distributed::Impl::compute_mapped_extents(
                         local_extents_t2, dst_map)));

  // Buffers
  auto buffer_extents_t01 =
           KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
               global_extents, topology0, topology1),
       buffer_extents_t02 =
           KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
               global_extents, topology0, topology2),
       buffer_extents_t12 =
           KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
               global_extents, topology1, topology2);
  DstView4DType recv_t01("recv_t01", KokkosFFT::Impl::create_layout<LayoutType>(
                                         buffer_extents_t01)),
      recv_t02("recv_t02",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t02)),
      recv_t10("recv_t10",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t01)),
      recv_t12("recv_t12",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t12)),
      recv_t20("recv_t20",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t02)),
      recv_t21("recv_t21",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t12));

  // Initialize input views and references without considering permutation
  auto h_gu = Kokkos::create_mirror_view(gu);
  for (std::size_t i2 = 0; i2 < gu.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < gu.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < gu.extent(0); i0++) {
        h_gu(i0, i1, i2) = static_cast<T>(i2 + i1 * gn2 + i0 * gn2 * gn1);
      }
    }
  }

  // Copy global source to local source and send buffers
  Kokkos::pair<std::size_t, std::size_t> range_gu_t0(n2_start,
                                                     n2_start + n2_length),
      range_gu_t1(n1_start, n1_start + n1_length),
      range_gu_t2(n0_start, n0_start + n0_length);

  auto h_u_t0 = Kokkos::create_mirror_view(u_t0);
  auto h_u_t1 = Kokkos::create_mirror_view(u_t1);
  auto h_u_t2 = Kokkos::create_mirror_view(u_t2);

  auto sub_h_gu_t0 =
      Kokkos::subview(h_gu, Kokkos::ALL, Kokkos::ALL, range_gu_t0);
  auto sub_h_gu_t1 =
      Kokkos::subview(h_gu, Kokkos::ALL, range_gu_t1, Kokkos::ALL);
  auto sub_h_gu_t2 =
      Kokkos::subview(h_gu, range_gu_t2, Kokkos::ALL, Kokkos::ALL);
  Kokkos::deep_copy(h_u_t0, sub_h_gu_t0);
  Kokkos::deep_copy(h_u_t1, sub_h_gu_t1);
  Kokkos::deep_copy(h_u_t2, sub_h_gu_t2);

  auto h_recv_t01 = Kokkos::create_mirror_view(recv_t01);
  auto h_recv_t02 = Kokkos::create_mirror_view(recv_t02);
  auto h_recv_t10 = Kokkos::create_mirror_view(recv_t10);
  auto h_recv_t12 = Kokkos::create_mirror_view(recv_t12);
  auto h_recv_t20 = Kokkos::create_mirror_view(recv_t20);
  auto h_recv_t21 = Kokkos::create_mirror_view(recv_t21);

  // t0 (gn0, gn1, gn2/p) -> t1 (gn0, gn1/p, gn2)
  // t1 (gn0, gn1/p, gn2) -> t0 (gn0, gn1, gn2/p)
  for (std::size_t i3 = 0; i3 < recv_t01.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < recv_t01.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < recv_t01.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < recv_t01.extent(0); i0++) {
          std::size_t i0_tmp = i0, i1_tmp = i1, i2_tmp = i2;
          std::size_t p = 0;
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            p = i3;
          } else {
            i0_tmp = i1, i1_tmp = i2, i2_tmp = i3;
            p = i0;
          }

          const auto [start1, length1] = distribute_extents(gn1, p, nprocs);
          const auto [start2, length2] = distribute_extents(gn2, p, nprocs);
          std::size_t gi1              = i1_tmp + start1;
          std::size_t gi2              = i2_tmp + start2;
          if (gi1 < gn1 && i1_tmp < length1 && i2_tmp < h_u_t0.extent(2)) {
            h_recv_t01(i0, i1, i2, i3) = h_u_t0(i0_tmp, gi1, i2_tmp);
          }
          if (gi2 < gn2 && i2_tmp < length2 && i1_tmp < h_u_t1.extent(1)) {
            h_recv_t10(i0, i1, i2, i3) = h_u_t1(i0_tmp, i1_tmp, gi2);
          }
        }
      }
    }
  }

  // t0 (gn0, gn1, gn2/p) -> t2 (gn0/p, gn1, gn2)
  // t2 (gn0/p, gn1, gn2) -> t0 (gn0, gn1, gn2/p)
  for (std::size_t i3 = 0; i3 < recv_t02.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < recv_t02.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < recv_t02.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < recv_t02.extent(0); i0++) {
          std::size_t i0_tmp = i0, i1_tmp = i1, i2_tmp = i2;
          std::size_t p = 0;
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            p = i3;
          } else {
            i0_tmp = i1, i1_tmp = i2, i2_tmp = i3;
            p = i0;
          }

          const auto [start0, length0] = distribute_extents(gn0, p, nprocs);
          const auto [start2, length2] = distribute_extents(gn2, p, nprocs);
          std::size_t gi0              = i0_tmp + start0;
          std::size_t gi2              = i2_tmp + start2;
          if (gi0 < gn0 && i0_tmp < length0 && i2_tmp < h_u_t0.extent(2)) {
            h_recv_t02(i0, i1, i2, i3) = h_u_t0(gi0, i1_tmp, i2_tmp);
          }
          if (gi2 < gn2 && i2_tmp < length2 && i0_tmp < h_u_t2.extent(0)) {
            h_recv_t20(i0, i1, i2, i3) = h_u_t2(i0_tmp, i1_tmp, gi2);
          }
        }
      }
    }
  }

  // t1 (gn0, gn1/p, gn2) -> t2 (gn0/p, gn1, gn2)
  // t2 (gn0/p, gn1, gn2) -> t1 (gn0, gn1/p, gn2)
  for (std::size_t i3 = 0; i3 < recv_t12.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < recv_t12.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < recv_t12.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < recv_t12.extent(0); i0++) {
          std::size_t i0_tmp = i0, i1_tmp = i1, i2_tmp = i2;
          std::size_t p = 0;
          if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft>) {
            p = i3;
          } else {
            i0_tmp = i1, i1_tmp = i2, i2_tmp = i3;
            p = i0;
          }

          const auto [start0, length0] = distribute_extents(gn0, p, nprocs);
          const auto [start1, length1] = distribute_extents(gn1, p, nprocs);
          std::size_t gi0              = i0_tmp + start0;
          std::size_t gi1              = i1_tmp + start1;

          if (gi0 < gn0 && i0_tmp < length0 && i1_tmp < h_u_t1.extent(1)) {
            h_recv_t12(i0, i1, i2, i3) = h_u_t1(gi0, i1_tmp, i2_tmp);
          }
          if (gi1 < gn1 && i1_tmp < length1 && i0_tmp < h_u_t2.extent(0)) {
            h_recv_t21(i0, i1, i2, i3) = h_u_t2(i0_tmp, gi1, i2_tmp);
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u_t0, h_u_t0);
  Kokkos::deep_copy(u_t1, h_u_t1);
  Kokkos::deep_copy(u_t2, h_u_t2);

  Kokkos::deep_copy(recv_t01, h_recv_t01);
  Kokkos::deep_copy(recv_t02, h_recv_t02);
  Kokkos::deep_copy(recv_t10, h_recv_t10);
  Kokkos::deep_copy(recv_t12, h_recv_t12);
  Kokkos::deep_copy(recv_t20, h_recv_t20);
  Kokkos::deep_copy(recv_t21, h_recv_t21);

  // Make permuted local views with safe_transpose
  execution_space exec;
  KokkosFFT::Impl::transpose(exec, u_t0, u_p_t0_ref, int_map, true);
  KokkosFFT::Impl::transpose(exec, u_t1, u_p_t1_ref, int_map, true);
  KokkosFFT::Impl::transpose(exec, u_t2, u_p_t2_ref, int_map, true);
  exec.fence();

  // Apply pack kernel
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t01, u_p_t0_01, dst_map, 1);
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t02, u_p_t0_02, dst_map, 0);
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t10, u_p_t1_10, dst_map, 2);
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t12, u_p_t1_12, dst_map, 0);
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t20, u_p_t2_20, dst_map, 2);
  KokkosFFT::Distributed::Impl::unpack(exec, recv_t21, u_p_t2_21, dst_map, 1);

  auto h_u_p_t0_01 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t0_01);
  auto h_u_p_t0_02 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t0_02);
  auto h_u_p_t1_10 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t1_10);
  auto h_u_p_t1_12 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t1_12);
  auto h_u_p_t2_20 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t2_20);
  auto h_u_p_t2_21 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t2_21);
  auto h_u_p_t0_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t0_ref);
  auto h_u_p_t1_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t1_ref);
  auto h_u_p_t2_ref =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u_p_t2_ref);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check u_p_t0 is correct
  for (std::size_t i2 = 0; i2 < u_p_t0_ref.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < u_p_t0_ref.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < u_p_t0_ref.extent(0); i0++) {
        auto diff_01 =
            Kokkos::abs(h_u_p_t0_01(i0, i1, i2) - h_u_p_t0_ref(i0, i1, i2));
        auto diff_02 =
            Kokkos::abs(h_u_p_t0_02(i0, i1, i2) - h_u_p_t0_ref(i0, i1, i2));
        EXPECT_LE(diff_01, epsilon);
        EXPECT_LE(diff_02, epsilon);
      }
    }
  }

  // Check u_p_t1 is correct
  for (std::size_t i2 = 0; i2 < u_p_t1_ref.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < u_p_t1_ref.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < u_p_t1_ref.extent(0); i0++) {
        auto diff_10 =
            Kokkos::abs(h_u_p_t1_10(i0, i1, i2) - h_u_p_t1_ref(i0, i1, i2));
        auto diff_12 =
            Kokkos::abs(h_u_p_t1_12(i0, i1, i2) - h_u_p_t1_ref(i0, i1, i2));
        EXPECT_LE(diff_10, epsilon);
        EXPECT_LE(diff_12, epsilon);
      }
    }
  }

  // Check u_p_t2 is correct
  for (std::size_t i2 = 0; i2 < u_p_t2_ref.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < u_p_t2_ref.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < u_p_t2_ref.extent(0); i0++) {
        auto diff_20 =
            Kokkos::abs(h_u_p_t2_20(i0, i1, i2) - h_u_p_t2_ref(i0, i1, i2));
        auto diff_21 =
            Kokkos::abs(h_u_p_t2_21(i0, i1, i2) - h_u_p_t2_ref(i0, i1, i2));
        EXPECT_LE(diff_20, epsilon);
        EXPECT_LE(diff_21, epsilon);
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(TestUnpack, test_types);

TYPED_TEST(TestUnpack, View2D_01) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view2D<float_type, layout_type>(rank, nprocs, 0);
    }
  }
}

TYPED_TEST(TestUnpack, View2D_10) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view2D<float_type, layout_type>(rank, nprocs, 1);
    }
  }
}

TYPED_TEST(TestUnpack, View3D_012) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view3D<float_type, layout_type>(rank, nprocs, 0);
    }
  }
}

TYPED_TEST(TestUnpack, View3D_021) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view3D<float_type, layout_type>(rank, nprocs, 1);
    }
  }
}

TYPED_TEST(TestUnpack, View3D_102) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view3D<float_type, layout_type>(rank, nprocs, 2);
    }
  }
}

TYPED_TEST(TestUnpack, View3D_120) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view3D<float_type, layout_type>(rank, nprocs, 3);
    }
  }
}

TYPED_TEST(TestUnpack, View3D_201) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view3D<float_type, layout_type>(rank, nprocs, 4);
    }
  }
}

TYPED_TEST(TestUnpack, View3D_210) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_unpack_view3D<float_type, layout_type>(rank, nprocs, 5);
    }
  }
}
