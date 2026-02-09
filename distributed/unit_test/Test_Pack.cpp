// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <vector>
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

template <typename T>
struct TestPack : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;

  int m_max_nprocs = 4;
};

using param_type = std::tuple<int, int, std::vector<int>, std::vector<int>>;
struct ParamTestsPack : public ::testing::TestWithParam<param_type> {};

/// \brief Test bin_mapping function
/// \tparam iType Integer type for indices
/// \param[in] N Total size
/// \param[in] nbins Number of bins (e.g., number of MPI processes)
/// \param[in] ref_start Reference start indices for each bin
/// \param[in] ref_length Reference lengths for each bin
///
/// For example, if N=10 and nbins=3, the bins will be:
/// [0, 4), [4, 7), [7, 10)
/// which corresponds to (start=0, length=4), (start=4, length=3),
/// (start=7, length=3).
template <typename iType>
void test_bin_mapping(iType N, iType nbins, const std::vector<iType>& ref_start,
                      const std::vector<iType>& ref_length) {
  std::vector<iType> start, length;
  for (iType ibin = 0; ibin < nbins; ++ibin) {
    auto [s, l] = KokkosFFT::Distributed::Impl::bin_mapping(N, nbins, ibin);
    start.push_back(s);
    length.push_back(l);
  }
  EXPECT_EQ(start, ref_start);
  EXPECT_EQ(length, ref_length);
}

/// \brief Test merge_indices function
/// \tparam iType Integer type for indices
/// \param[in] size Size of the index range to test
/// \param[in] start Start indices for each bin
/// \param[in] length Lengths for each bin
template <typename iType>
void test_merge_indices(iType size, const std::vector<iType>& start,
                        const std::vector<iType>& length) {
  iType axis0 = 0, axis1 = 1;
  for (std::size_t i = 0; i < start.size(); ++i) {
    for (iType idx = 0; idx < size; idx++) {
      // if axis == merged_axis, idx is merged
      auto merged_idx_00 = KokkosFFT::Distributed::Impl::merge_indices(
          idx, start.at(i), length.at(i), axis0, axis0);

      // if axis != merged_axis, idx is unchanged
      auto merged_idx_01 = KokkosFFT::Distributed::Impl::merge_indices(
          idx, start.at(i), length.at(i), axis0, axis1);

      if (idx >= length.at(i)) {
        EXPECT_EQ(merged_idx_00, -1);
      } else {
        auto ref_merged_idx_00 = idx + start.at(i);
        EXPECT_EQ(merged_idx_00, ref_merged_idx_00);
      }

      EXPECT_EQ(merged_idx_01, idx);
    }
  }
}

/// \brief Test pack function for 2D View
/// \tparam T Type of the data (float or double)
/// \tparam LayoutType Layout of the data (LayoutLeft or LayoutRight)
///
/// \param[in] rank MPI rank
/// \param[in] nprocs Number of MPI ranks
/// \param[in] order Order of the dimensions (0: (n0, n1); 1: (n1, n0))
///
/// 0. map = {0, 1}: the source view is (n0, n1/np) for x-pencil and (n0/np, n1)
/// for y-pencil.
/// 0.0 LayoutLeft: The sendbuffer is (n0/np, n1/np, np) for both pencils.
/// 0.1 LayoutRight: The sendbuffer is (np, n0/np, n1/np) for both pencils.
///
/// 1. map = {1, 0}: the source view is (n1/np, n0) for x-pencil and (n1, n0/np)
/// for y-pencil.
/// 1.0 LayoutLeft: The sendbuffer is (n0/np, n1/np, np) for both pencils.
/// 1.1 LayoutRight: The sendbuffer is (np, n0/np, n1/np) for both pencils.
template <typename T, typename LayoutType>
void test_pack_view2D(std::size_t rank, std::size_t nprocs, int order = 0) {
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

  shape_type src_map = (order == 0) ? shape_type({0, 1}) : shape_type({1, 0});
  map_type int_map   = (order == 0) ? map_type({0, 1}) : map_type({1, 0});

  // Create global and local views
  SrcView2DType gu("gu", gn0, gn1);

  // Data in Topology 0 (X-pencil): original and permuted data
  SrcView2DType u_t0(
      "u_t0", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t0)),
      u_p_t0("u_p_t0", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t0, src_map)));

  // Data in Topology 1 (Y-pencil): original and permuted data
  SrcView2DType u_t1(
      "u_t1", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t1)),
      u_p_t1("u_p_t1", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t1, src_map)));

  // Buffers
  auto buffer_extents =
      KokkosFFT::Distributed::Impl::compute_buffer_extents<LayoutType>(
          global_extents, topology0, topology1);
  DstView3DType send_t0(
      "send_t0", KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents)),
      send_t0_ref("send_t0_ref",
                  KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents)),
      send_t1("send_t1",
              KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents)),
      send_t1_ref("send_t1_ref",
                  KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents));

  // Initialize input views and references without considering permutation
  auto h_gu = Kokkos::create_mirror_view(gu);
  for (std::size_t i1 = 0; i1 < gu.extent(1); i1++) {
    for (std::size_t i0 = 0; i0 < gu.extent(0); i0++) {
      h_gu(i0, i1) = static_cast<T>(i1 + i0 * gn1);
    }
  }

  // Copy global source to local source and send buffers
  Kokkos::pair<std::size_t, std::size_t> range_gu_t0(n1_start,
                                                     n1_start + n1_length),
      range_gu_t1(n0_start, n0_start + n0_length);

  auto h_u_t0 = Kokkos::create_mirror_view(u_t0);
  auto h_u_t1 = Kokkos::create_mirror_view(u_t1);

  auto sub_h_gu_t0 = Kokkos::subview(h_gu, Kokkos::ALL, range_gu_t0);
  auto sub_h_gu_t1 = Kokkos::subview(h_gu, range_gu_t1, Kokkos::ALL);
  Kokkos::deep_copy(h_u_t0, sub_h_gu_t0);
  Kokkos::deep_copy(h_u_t1, sub_h_gu_t1);

  auto h_send_t0_ref = Kokkos::create_mirror_view(send_t0_ref);
  auto h_send_t1_ref = Kokkos::create_mirror_view(send_t1_ref);

  for (std::size_t i2 = 0; i2 < send_t0_ref.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send_t0_ref.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send_t0_ref.extent(0); i0++) {
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
          h_send_t0_ref(i0, i1, i2) = h_u_t0(gi0, i1_tmp);
        }
        if (gi1 < gn1 && i1_tmp < length1 && i0_tmp < h_u_t1.extent(0)) {
          h_send_t1_ref(i0, i1, i2) = h_u_t1(i0_tmp, gi1);
        }
      }
    }
  }
  Kokkos::deep_copy(u_t0, h_u_t0);
  Kokkos::deep_copy(u_t1, h_u_t1);

  // Make permuted local views with safe_transpose
  execution_space exec;
  KokkosFFT::Impl::transpose(exec, u_t0, u_p_t0, int_map, true);
  KokkosFFT::Impl::transpose(exec, u_t1, u_p_t1, int_map, true);
  exec.fence();

  // Apply pack kernel
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t0, send_t0, src_map, 0);
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t1, send_t1, src_map, 1);

  auto h_send_t0 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t0);
  auto h_send_t1 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t1);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check send_t0 is correct
  for (std::size_t i2 = 0; i2 < send_t0.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send_t0.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send_t0.extent(0); i0++) {
        auto diff =
            Kokkos::abs(h_send_t0(i0, i1, i2) - h_send_t0_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }

  // Check send_t1 is correct
  for (std::size_t i2 = 0; i2 < send_t1.extent(2); i2++) {
    for (std::size_t i1 = 0; i1 < send_t1.extent(1); i1++) {
      for (std::size_t i0 = 0; i0 < send_t1.extent(0); i0++) {
        auto diff =
            Kokkos::abs(h_send_t1(i0, i1, i2) - h_send_t1_ref(i0, i1, i2));
        EXPECT_LE(diff, epsilon);
      }
    }
  }
}

/// \brief Test pack function for 3D View
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
void test_pack_view3D(std::size_t rank, std::size_t nprocs, int order = 0) {
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

  shape_type src_map = (order == 0)   ? shape_type({0, 1, 2})
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
      u_p_t0("u_p_t0", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t0, src_map)));

  // Data in Topology 1 (Y-slab): original and permuted data
  SrcView3DType u_t1(
      "u_t1", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t1)),
      u_p_t1("u_p_t1", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t1, src_map)));

  // Data in Topology 2 (X-slab): original and permuted data
  SrcView3DType u_t2(
      "u_t2", KokkosFFT::Impl::create_layout<LayoutType>(local_extents_t2)),
      u_p_t2("u_p_t2", KokkosFFT::Impl::create_layout<LayoutType>(
                           KokkosFFT::Distributed::Impl::compute_mapped_extents(
                               local_extents_t2, src_map)));

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
  DstView4DType send_t01("send_t01", KokkosFFT::Impl::create_layout<LayoutType>(
                                         buffer_extents_t01)),
      send_t01_ref("send_t01_ref", KokkosFFT::Impl::create_layout<LayoutType>(
                                       buffer_extents_t01)),
      send_t02("send_t02",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t02)),
      send_t02_ref("send_t02_ref", KokkosFFT::Impl::create_layout<LayoutType>(
                                       buffer_extents_t02)),
      send_t10("send_t10",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t01)),
      send_t10_ref("send_t10_ref", KokkosFFT::Impl::create_layout<LayoutType>(
                                       buffer_extents_t01)),
      send_t12("send_t12",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t12)),
      send_t12_ref("send_t12_ref", KokkosFFT::Impl::create_layout<LayoutType>(
                                       buffer_extents_t12)),
      send_t20("send_t20",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t02)),
      send_t20_ref("send_t20_ref", KokkosFFT::Impl::create_layout<LayoutType>(
                                       buffer_extents_t02)),
      send_t21("send_t21",
               KokkosFFT::Impl::create_layout<LayoutType>(buffer_extents_t12)),
      send_t21_ref("send_t21_ref", KokkosFFT::Impl::create_layout<LayoutType>(
                                       buffer_extents_t12));

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

  auto h_send_t01_ref = Kokkos::create_mirror_view(send_t01_ref);
  auto h_send_t02_ref = Kokkos::create_mirror_view(send_t02_ref);
  auto h_send_t10_ref = Kokkos::create_mirror_view(send_t10_ref);
  auto h_send_t12_ref = Kokkos::create_mirror_view(send_t12_ref);
  auto h_send_t20_ref = Kokkos::create_mirror_view(send_t20_ref);
  auto h_send_t21_ref = Kokkos::create_mirror_view(send_t21_ref);

  // t0 (gn0, gn1, gn2/p) -> t1 (gn0, gn1/p, gn2)
  // t1 (gn0, gn1/p, gn2) -> t0 (gn0, gn1, gn2/p)
  for (std::size_t i3 = 0; i3 < send_t01_ref.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t01_ref.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t01_ref.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t01_ref.extent(0); i0++) {
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
            h_send_t01_ref(i0, i1, i2, i3) = h_u_t0(i0_tmp, gi1, i2_tmp);
          }
          if (gi2 < gn2 && i2_tmp < length2 && i1_tmp < h_u_t1.extent(1)) {
            h_send_t10_ref(i0, i1, i2, i3) = h_u_t1(i0_tmp, i1_tmp, gi2);
          }
        }
      }
    }
  }

  // t0 (gn0, gn1, gn2/p) -> t2 (gn0/p, gn1, gn2)
  // t2 (gn0/p, gn1, gn2) -> t0 (gn0, gn1, gn2/p)
  for (std::size_t i3 = 0; i3 < send_t02_ref.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t02_ref.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t02_ref.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t02_ref.extent(0); i0++) {
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
            h_send_t02_ref(i0, i1, i2, i3) = h_u_t0(gi0, i1_tmp, i2_tmp);
          }
          if (gi2 < gn2 && i2_tmp < length2 && i0_tmp < h_u_t2.extent(0)) {
            h_send_t20_ref(i0, i1, i2, i3) = h_u_t2(i0_tmp, i1_tmp, gi2);
          }
        }
      }
    }
  }

  // t1 (gn0, gn1/p, gn2) -> t2 (gn0/p, gn1, gn2)
  // t2 (gn0/p, gn1, gn2) -> t1 (gn0, gn1/p, gn2)
  for (std::size_t i3 = 0; i3 < send_t12_ref.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t12_ref.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t12_ref.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t12_ref.extent(0); i0++) {
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
            h_send_t12_ref(i0, i1, i2, i3) = h_u_t1(gi0, i1_tmp, i2_tmp);
          }
          if (gi1 < gn1 && i1_tmp < length1 && i0_tmp < h_u_t2.extent(0)) {
            h_send_t21_ref(i0, i1, i2, i3) = h_u_t2(i0_tmp, gi1, i2_tmp);
          }
        }
      }
    }
  }

  Kokkos::deep_copy(u_t0, h_u_t0);
  Kokkos::deep_copy(u_t1, h_u_t1);
  Kokkos::deep_copy(u_t2, h_u_t2);

  // Make permuted local views with safe_transpose
  execution_space exec;
  KokkosFFT::Impl::transpose(exec, u_t0, u_p_t0, int_map, true);
  KokkosFFT::Impl::transpose(exec, u_t1, u_p_t1, int_map, true);
  KokkosFFT::Impl::transpose(exec, u_t2, u_p_t2, int_map, true);
  exec.fence();

  // Apply pack kernel
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t0, send_t01, src_map, 1);
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t0, send_t02, src_map, 0);
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t1, send_t10, src_map, 2);
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t1, send_t12, src_map, 0);
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t2, send_t20, src_map, 2);
  KokkosFFT::Distributed::Impl::pack(exec, u_p_t2, send_t21, src_map, 1);

  auto h_send_t01 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t01);
  auto h_send_t02 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t02);
  auto h_send_t10 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t10);
  auto h_send_t12 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t12);
  auto h_send_t20 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t20);
  auto h_send_t21 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), send_t21);

  T epsilon = std::numeric_limits<T>::epsilon() * 100;

  // Check send_t01 is correct
  for (std::size_t i3 = 0; i3 < send_t01.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t01.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t01.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t01.extent(0); i0++) {
          auto diff = Kokkos::abs(h_send_t01(i0, i1, i2, i3) -
                                  h_send_t01_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check send_t02 is correct
  for (std::size_t i3 = 0; i3 < send_t02.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t02.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t02.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t02.extent(0); i0++) {
          auto diff = Kokkos::abs(h_send_t02(i0, i1, i2, i3) -
                                  h_send_t02_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check send_t10 is correct
  for (std::size_t i3 = 0; i3 < send_t10.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t10.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t10.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t10.extent(0); i0++) {
          auto diff = Kokkos::abs(h_send_t10(i0, i1, i2, i3) -
                                  h_send_t10_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check send_t12 is correct
  for (std::size_t i3 = 0; i3 < send_t12.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t12.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t12.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t12.extent(0); i0++) {
          auto diff = Kokkos::abs(h_send_t12(i0, i1, i2, i3) -
                                  h_send_t12_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check send_t20 is correct
  for (std::size_t i3 = 0; i3 < send_t20.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t20.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t20.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t20.extent(0); i0++) {
          auto diff = Kokkos::abs(h_send_t20(i0, i1, i2, i3) -
                                  h_send_t20_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }

  // Check send_t21 is correct
  for (std::size_t i3 = 0; i3 < send_t21.extent(3); i3++) {
    for (std::size_t i2 = 0; i2 < send_t21.extent(2); i2++) {
      for (std::size_t i1 = 0; i1 < send_t21.extent(1); i1++) {
        for (std::size_t i0 = 0; i0 < send_t21.extent(0); i0++) {
          auto diff = Kokkos::abs(h_send_t21(i0, i1, i2, i3) -
                                  h_send_t21_ref(i0, i1, i2, i3));
          EXPECT_LE(diff, epsilon);
        }
      }
    }
  }
}

}  // namespace

TEST_P(ParamTestsPack, bin_mapping) {
  auto [N, nprocs, starts, lengths] = GetParam();
  test_bin_mapping(N, nprocs, starts, lengths);
}

TEST_P(ParamTestsPack, merge_indices) {
  auto [N, nprocs, starts, lengths] = GetParam();
  auto size                         = (N - 1) / nprocs + 1;
  test_merge_indices(size, starts, lengths);
}

// Parameterized over N, nprocs, starts, lengths
INSTANTIATE_TEST_SUITE_P(
    PTestPack, ParamTestsPack,
    ::testing::Values(
        param_type{10, 3, std::vector<int>{0, 4, 7}, std::vector<int>{4, 3, 3}},
        param_type{12, 3, std::vector<int>{0, 4, 8}, std::vector<int>{4, 4, 4}},
        param_type{128, 6, std::vector<int>{0, 22, 44, 65, 86, 107},
                   std::vector<int>{22, 22, 21, 21, 21, 21}}));

TYPED_TEST_SUITE(TestPack, test_types);

TYPED_TEST(TestPack, View2D_01) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view2D<float_type, layout_type>(rank, nprocs, 0);
    }
  }
}

TYPED_TEST(TestPack, View2D_10) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view2D<float_type, layout_type>(rank, nprocs, 1);
    }
  }
}

TYPED_TEST(TestPack, View3D_012) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view3D<float_type, layout_type>(rank, nprocs, 0);
    }
  }
}

TYPED_TEST(TestPack, View3D_021) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view3D<float_type, layout_type>(rank, nprocs, 1);
    }
  }
}

TYPED_TEST(TestPack, View3D_102) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view3D<float_type, layout_type>(rank, nprocs, 2);
    }
  }
}

TYPED_TEST(TestPack, View3D_120) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view3D<float_type, layout_type>(rank, nprocs, 3);
    }
  }
}

TYPED_TEST(TestPack, View3D_201) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view3D<float_type, layout_type>(rank, nprocs, 4);
    }
  }
}

TYPED_TEST(TestPack, View3D_210) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;

  for (int nprocs = 1; nprocs <= this->m_max_nprocs; ++nprocs) {
    for (int rank = 0; rank < nprocs; ++rank) {
      test_pack_view3D<float_type, layout_type>(rank, nprocs, 5);
    }
  }
}
