// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <mpi.h>
#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Distributed_Topologies.hpp"
#include "Test_Utils.hpp"

namespace {
class TopologyParamTests : public ::testing::TestWithParam<int> {};
class SlabParamTests : public ::testing::TestWithParam<int> {};
class PencilParamTests : public ::testing::TestWithParam<int> {};

/// \brief Convert topology type to string for better test failure messages.
/// \param[in] type The topology type to convert.
/// \return The string representation of the topology type.
inline std::string topology_type_to_string(
    KokkosFFT::Distributed::Impl::TopologyType type) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  switch (type) {
    case TopologyType::Empty: return "Empty";
    case TopologyType::Shared: return "Shared";
    case TopologyType::Slab: return "Slab";
    case TopologyType::Pencil: return "Pencil";
    case TopologyType::Brick: return "Brick";
    case TopologyType::Invalid: return "Invalid";
    default: return "Unknown";
  }
}

template <std::size_t DIM, typename iType>
auto to_slab_topology(std::size_t idx, iType value) {
  std::array<iType, DIM> topology{};
  topology.fill(1);
  topology.at(idx) = value;
  return topology;
}

template <typename LayoutType, std::size_t DIM, typename iType>
auto to_pencil_topology(std::size_t idx0, std::size_t idx1, iType np0,
                        iType np1) {
  using TopologyType = KokkosFFT::Distributed::Topology<iType, DIM, LayoutType>;
  TopologyType topology{};
  topology.fill(1);
  topology.at(idx0) = np0;
  topology.at(idx1) = np1;
  return topology;
}

template <std::size_t DIM, typename iType>
auto to_vec_topology(const std::vector<iType>& indices, iType value) {
  std::vector<std::array<iType, DIM>> topologies;
  for (auto index : indices) {
    topologies.push_back(to_slab_topology<DIM>(index, value));
  }
  return topologies;
}

template <std::size_t DIM, typename iType>
bool has_consecutive_duplicates(
    const std::vector<std::array<iType, DIM>>& vec) {
  return vec.empty() || std::adjacent_find(vec.begin(), vec.end()) != vec.end();
}

/// \brief Generate error message for topology type test failures.
/// \tparam TopologyContainerType The type of the topology input.
/// \param[in] topology The input topology that caused the failure.
/// \param[in] ref The expected topology type that should have been returned.
/// \return Error message including the input topology, expected topology type,
/// and actual topology type.
template <typename TopologyContainerType>
std::string error_to_topology_type(
    const TopologyContainerType& topology,
    KokkosFFT::Distributed::Impl::TopologyType ref) {
  std::string msg =
      KokkosFFT::Impl::container_to_string("Input topology: ", topology);
  msg += ", should be: " + topology_type_to_string(ref) + ", but got: " +
         topology_type_to_string(
             KokkosFFT::Distributed::Impl::to_topology_type(topology));
  return msg;
}

/// \brief Generate error message for get_common_topology_type test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] ref The expected common topology type that should have been
/// returned.
/// \return Error message including the input topologies, expected common
/// topology type, and actual common topology type.
template <typename Topology1Type, typename Topology2Type>
std::string error_get_common_topology_type(
    const Topology1Type& topo1, const Topology2Type& topo2,
    KokkosFFT::Distributed::Impl::TopologyType ref) {
  std::string msg =
      KokkosFFT::Impl::container_to_string("Input topologies: ", topo1);
  msg += " and " + KokkosFFT::Impl::container_to_string("", topo2);
  msg +=
      ", should be: " + topology_type_to_string(ref) + ", but got: " +
      topology_type_to_string(
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2));
  return msg;
}

/// \brief Generate error message for is_topology test failures.
/// \tparam TopologyType The type of the topology input.
/// \param[in] topology The input topology that caused the failure.
/// \param[in] specified The topology type that was expected to be identified or
/// not identified.
/// \param[in] expected Whether the input topology was expected to be identified
/// as the specified topology type.
/// \return Error message including the input topology, specified topology type,
/// and whether it was expected to be identified.
template <typename TopologyType>
std::string error_is_topology(
    const TopologyType& topology,
    KokkosFFT::Distributed::Impl::TopologyType specified, bool expected) {
  std::string msg =
      KokkosFFT::Impl::container_to_string("Input topology: ", topology);
  if (expected) {
    msg += ", should be identified as " + topology_type_to_string(specified) +
           ", but it is not.";
  } else {
    msg += ", should not be identified as " +
           topology_type_to_string(specified) + ", but it is.";
  }
  return msg;
}

/// \brief Generate error message for are_topologies test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] specified The topology type that was expected to be identified or
/// not identified.
/// \param[in] expected Whether the input topologies were expected to be
/// identified as the specified topology type.
/// \return Error message including the input topologies, specified topology
/// type, and whether they were expected to be identified.
template <typename Topology1Type, typename Topology2Type>
std::string error_are_topologies(
    const Topology1Type& topo1, const Topology2Type& topo2,
    KokkosFFT::Distributed::Impl::TopologyType specified, bool expected) {
  std::string msg =
      KokkosFFT::Impl::container_to_string("Input topologies: ", topo1);
  msg += " and " + KokkosFFT::Impl::container_to_string("", topo2);
  if (expected) {
    msg += ", should be identified as " + topology_type_to_string(specified) +
           ", but it is not.";
  } else {
    msg += ", should not be identified as " +
           topology_type_to_string(specified) + ", but it is.";
  }
  return msg;
}

/// \brief Generate error message for in_out_axes test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] actual The actual in/out axes.
/// \param[in] expected The expected in/out axes.
/// \return Error message including the input topologies and the expected in/out
/// axes.
template <typename Topology1Type, typename Topology2Type>
std::string error_in_out_axes(
    const Topology1Type& topo1, const Topology2Type& topo2,
    const std::tuple<std::size_t, std::size_t>& actual,
    const std::tuple<std::size_t, std::size_t>& expected) {
  std::string msg = "Input topologies: ";
  msg += KokkosFFT::Impl::container_to_string("topology1: ", topo1);
  msg += " and ";
  msg += KokkosFFT::Impl::container_to_string("topology2: ", topo2);
  msg += ", should have in/out axes: (";
  msg += std::to_string(std::get<0>(expected));
  msg += ", ";
  msg += std::to_string(std::get<1>(expected));
  msg += "), but got: (";
  msg += std::to_string(std::get<0>(actual));
  msg += ", ";
  msg += std::to_string(std::get<1>(actual));
  msg += ").";
  return msg;
}

/// \brief Generate error message for decompose_axes test failures.
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
/// \param[in] topologies The input topologies that caused the failure.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] expected The expected decomposed axes
/// \return Error message including the input topologies, FFT axes, and the
/// expected decomposition
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::string error_decompose_axes(
    const std::vector<std::array<std::size_t, DIM>>& topologies,
    const std::array<iType, FFT_DIM>& axes,
    const std::vector<std::vector<iType>>& expected) {
  auto actual = KokkosFFT::Distributed::Impl::decompose_axes(topologies, axes);
  std::string msg = "Input topologies: ";
  for (std::size_t i = 0; i < topologies.size(); ++i) {
    msg += KokkosFFT::Impl::container_to_string(
        "topology" + std::to_string(i) + ": ", topologies.at(i));
    if (i != topologies.size() - 1) {
      msg += " and ";
    }
  }
  msg += ", with FFT axes: ";
  msg += KokkosFFT::Impl::container_to_string("axes: ", axes);
  msg += ", should have decomposed axes: (";
  for (std::size_t i = 0; i < expected.size(); ++i) {
    msg += KokkosFFT::Impl::container_to_string("", expected.at(i));
    if (i != expected.size() - 1) {
      msg += " and ";
    }
  }
  msg += "), but got: (";
  for (std::size_t i = 0; i < actual.size(); ++i) {
    msg += KokkosFFT::Impl::container_to_string("", actual.at(i));
    if (i != actual.size() - 1) {
      msg += " and ";
    }
  }
  msg += ").";

  return msg;
}

/// \brief Generate error message for get_all_slab_topologies and
/// get_all_pencil_topologies test failures.
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
/// \param[in] gin_extents The input extents.
/// \param[in] gout_extents The output extents.
/// \param[in] in_topology The input topology.
/// \param[out] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] actual The computed vector of all possible topologies that can be
/// formed
/// \param[in] expected The expected vector of all possible topologies that can
/// be formed
/// \return Error message including the input topologies, FFT axes, and the
/// expected vector of topologies
template <typename iType, std::size_t DIM, std::size_t FFT_DIM>
std::string error_all_topologies(
    const std::array<std::size_t, DIM>& gin_extents,
    const std::array<std::size_t, DIM>& gout_extents,
    const std::array<std::size_t, DIM>& in_topology,
    const std::array<std::size_t, DIM>& out_topology,
    const std::array<iType, FFT_DIM>& axes,
    const std::vector<std::array<std::size_t, DIM>>& actual,
    const std::vector<std::array<std::size_t, DIM>>& expected) {
  std::string msg;
  msg += "Input global extents: ";
  msg += KokkosFFT::Impl::container_to_string("gin_extents: ", gin_extents);
  msg += " and ";
  msg += KokkosFFT::Impl::container_to_string("gout_extents: ", gout_extents);
  msg += ", with Input topologies: ";
  msg += KokkosFFT::Impl::container_to_string("in_topology: ", in_topology);
  msg += " and ";
  msg += KokkosFFT::Impl::container_to_string("out_topology: ", out_topology);
  msg += ", with FFT axes: ";
  msg += KokkosFFT::Impl::container_to_string("axes: ", axes);
  msg += ", should have topologies: (";
  for (std::size_t i = 0; i < expected.size(); ++i) {
    msg += KokkosFFT::Impl::container_to_string("", expected.at(i));
    if (i != expected.size() - 1) {
      msg += ", ";
    }
  }
  msg += "), but got: (";
  for (std::size_t i = 0; i < actual.size(); ++i) {
    msg += KokkosFFT::Impl::container_to_string("", actual.at(i));
    if (i != actual.size() - 1) {
      msg += ", ";
    }
  }
  msg += ").";

  return msg;
}

/// \brief Generate error message for get_all_pencil_topologies and
/// get_all_pencil_topologies test failures.
/// \tparam Topology1Type The type of the first topology input.
/// \tparam Topology2Type The type of the second topology input.
/// \tparam iType The index type used for the topology.
/// \tparam DIM The dimensionality of the topology.
/// \tparam FFT_DIM The dimensionality of the FFT axes.
/// \param[in] in_topology The input topology.
/// \param[out] out_topology The output topology.
/// \param[in] axes The axes along which the FFT is performed.
/// \param[in] actual The computed vector of all possible topologies that can be
/// formed
/// \param[in] expected The expected vector of all possible topologies that can
/// be formed
/// \return Error message including the input topologies, FFT axes, and the
/// expected vector of topologies
template <typename Topology1Type, typename Topology2Type,
          typename TopoAndAxesType, typename iType, std::size_t FFT_DIM>
std::string error_all_pencil_topologies(const Topology1Type& in_topology,
                                        const Topology2Type& out_topology,
                                        const std::array<iType, FFT_DIM>& axes,
                                        const TopoAndAxesType& actual,
                                        const TopoAndAxesType& expected) {
  auto print_topo_and_axes = [](const TopoAndAxesType& t) {
    auto [all_topologies, all_trans_axes, all_layouts, all_axes] = t;
    std::string msg;
    msg += "topologies: (";
    for (std::size_t i = 0; i < all_topologies.size(); ++i) {
      msg += "(";
      for (std::size_t j = 0; j < all_topologies.at(i).size(); ++j) {
        msg += std::to_string(all_topologies.at(i).at(j));
        if (j != all_topologies.at(i).size() - 1) {
          msg += ", ";
        }
      }
      msg += ")";
      if (i != all_topologies.size() - 1) {
        msg += " and ";
      }
    }
    msg += "), trans_axes: (";
    for (std::size_t i = 0; i < all_trans_axes.size(); ++i) {
      msg += std::to_string(all_trans_axes.at(i));
      if (i != all_trans_axes.size() - 1) {
        msg += ", ";
      }
    }
    msg += "), layouts: (";
    for (std::size_t i = 0; i < all_layouts.size(); ++i) {
      msg += std::to_string(all_layouts.at(i));
      if (i != all_layouts.size() - 1) {
        msg += ", ";
      }
    }
    msg += "), axes: (";
    for (std::size_t i = 0; i < all_axes.size(); ++i) {
      msg += KokkosFFT::Impl::container_to_string("", all_axes.at(i));
      if (i != all_axes.size() - 1) {
        msg += ", ";
      }
    }
    msg += ")";
    return msg;
  };

  std::string msg;
  msg += "Input topologies: ";
  msg += "(";
  for (std::size_t i = 0; i < in_topology.size(); ++i) {
    msg += std::to_string(in_topology.at(i));
    if (i != in_topology.size() - 1) {
      msg += ", ";
    }
  }
  msg += ") and (";
  for (std::size_t i = 0; i < out_topology.size(); ++i) {
    msg += std::to_string(out_topology.at(i));
    if (i != out_topology.size() - 1) {
      msg += ", ";
    }
  }
  msg += "), with FFT axes: (";
  msg += std::to_string(axes.at(0));
  for (std::size_t i = 1; i < axes.size(); ++i) {
    msg += ", " + std::to_string(axes.at(i));
  }

  msg += "), should have: " + print_topo_and_axes(expected);
  msg += ", but got: " + print_topo_and_axes(actual);

  return msg;
}

template <bool is_std_array>
void test_to_topology_type(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type = std::tuple<topology1D_type, TopologyType>;
  using topology2D_and_ref2D_type = std::tuple<topology2D_type, TopologyType>;
  using topology3D_and_ref3D_type = std::tuple<topology3D_type, TopologyType>;
  using topology4D_and_ref4D_type = std::tuple<topology4D_type, TopologyType>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, KokkosFFT::Distributed::Impl::TopologyType::Shared},
        {topology1D_type{0},
         KokkosFFT::Distributed::Impl::TopologyType::Empty}};
    for (const auto& [topo, ref] : topo1D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Slab},
        {topology2_2, TopologyType::Slab},
        {topology2D_type{0, nprocs}, TopologyType::Empty},
        {topology2D_type{nprocs, 0}, TopologyType::Empty}};
    for (const auto& [topo, ref] : topo2D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Pencil},
        {topology3_2, TopologyType::Pencil},
        {topology3_3, TopologyType::Pencil},
        {topology3_4, TopologyType::Slab},
        {topology3_5, TopologyType::Shared}};
    for (const auto& [topo, ref] : topo3D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Pencil},
        {topology4_2, TopologyType::Pencil},
        {topology4_3, TopologyType::Pencil},
        {topology4_4, TopologyType::Pencil},
        {topology4_5, TopologyType::Pencil},
        {topology4_6, TopologyType::Pencil},
        {topology4_7, TopologyType::Slab},
        {topology4_8, TopologyType::Shared}};
    for (const auto& [topo, ref] : topo4D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }
  } else {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, TopologyType::Slab}};
    for (const auto& [topo, ref] : topo1D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Pencil},
        {topology2_2, TopologyType::Pencil}};
    for (const auto& [topo, ref] : topo2D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Brick},
        {topology3_2, TopologyType::Brick},
        {topology3_3, TopologyType::Brick},
        {topology3_4, TopologyType::Brick},
        {topology3_5, TopologyType::Brick}};
    for (const auto& [topo, ref] : topo3D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Invalid},
        {topology4_2, TopologyType::Invalid},
        {topology4_3, TopologyType::Invalid},
        {topology4_4, TopologyType::Invalid},
        {topology4_5, TopologyType::Invalid},
        {topology4_6, TopologyType::Invalid},
        {topology4_7, TopologyType::Invalid},
        {topology4_8, TopologyType::Invalid}};
    for (const auto& [topo, ref] : topo4D_test_cases) {
      auto topo_type = KokkosFFT::Distributed::Impl::to_topology_type(topo);
      EXPECT_EQ(topo_type, ref) << error_to_topology_type(topo, ref);
    }
  }
}

template <bool is_std_array>
void test_get_common_topology_type(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type =
      std::tuple<topology1D_type, topology1D_type, TopologyType>;
  using topology2D_and_ref2D_type =
      std::tuple<topology2D_type, topology2D_type, TopologyType>;
  using topology3D_and_ref3D_type =
      std::tuple<topology3D_type, topology3D_type, TopologyType>;
  using topology4D_and_ref4D_type =
      std::tuple<topology4D_type, topology4D_type, TopologyType>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1,
         KokkosFFT::Distributed::Impl::TopologyType::Shared},
        {topology1D_type{0}, topology1,
         KokkosFFT::Distributed::Impl::TopologyType::Empty},
        {topology1, topology1D_type{0},
         KokkosFFT::Distributed::Impl::TopologyType::Empty}};
    for (const auto& [topo1, topo2, ref] : topo1D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1,
         KokkosFFT::Distributed::Impl::TopologyType::Slab},
        {topology2_2, topology2_2,
         KokkosFFT::Distributed::Impl::TopologyType::Slab},
        {topology2_1, topology2_2,
         KokkosFFT::Distributed::Impl::TopologyType::Slab},
        {topology2D_type{0, nprocs}, topology2_1,
         KokkosFFT::Distributed::Impl::TopologyType::Empty},
        {topology2D_type{nprocs, 0}, topology2_2,
         KokkosFFT::Distributed::Impl::TopologyType::Empty}};

    for (const auto& [topo1, topo2, ref] : topo2D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, topology3_1, TopologyType::Pencil},
        {topology3_2, topology3_2, TopologyType::Pencil},
        {topology3_3, topology3_3, TopologyType::Pencil},
        {topology3_4, topology3_4, TopologyType::Slab},
        {topology3_5, topology3_5, TopologyType::Shared},
        {topology3_1, topology3_2, TopologyType::Pencil},
        {topology3_1, topology3_3, TopologyType::Pencil},
        {topology3_1, topology3_4, TopologyType::Invalid},
        {topology3_1, topology3_5, TopologyType::Invalid}};
    for (const auto& [topo1, topo2, ref] : topo3D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Pencil},
        {topology4_2, topology4_2, TopologyType::Pencil},
        {topology4_3, topology4_3, TopologyType::Pencil},
        {topology4_4, topology4_4, TopologyType::Pencil},
        {topology4_5, topology4_5, TopologyType::Pencil},
        {topology4_6, topology4_6, TopologyType::Pencil},
        {topology4_7, topology4_7, TopologyType::Slab},
        {topology4_8, topology4_8, TopologyType::Shared},
        {topology4_1, topology4_2, TopologyType::Pencil},
        {topology4_1, topology4_3, TopologyType::Pencil},
        {topology4_1, topology4_4, TopologyType::Pencil},
        {topology4_1, topology4_5, TopologyType::Pencil},
        {topology4_1, topology4_6, TopologyType::Pencil},
        {topology4_2, topology4_3, TopologyType::Pencil},
        {topology4_2, topology4_4, TopologyType::Pencil},
        {topology4_2, topology4_5, TopologyType::Pencil},
        {topology4_2, topology4_6, TopologyType::Pencil},
        {topology4_3, topology4_4, TopologyType::Pencil},
        {topology4_3, topology4_5, TopologyType::Pencil},
        {topology4_3, topology4_6, TopologyType::Pencil},
        {topology4_4, topology4_5, TopologyType::Pencil},
        {topology4_4, topology4_6, TopologyType::Pencil},
        {topology4_5, topology4_6, TopologyType::Pencil},
        {topology4_7, topology4_7, TopologyType::Slab},
        {topology4_8, topology4_8, TopologyType::Shared}};

    for (const auto& [topo1, topo2, ref] : topo4D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }
  } else {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1, TopologyType::Slab}};
    for (const auto& [topo1, topo2, ref] : topo1D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1, TopologyType::Pencil},
        {topology2_2, topology2_2, TopologyType::Pencil},
        {topology2_1, topology2_2, TopologyType::Pencil}};
    for (const auto& [topo1, topo2, ref] : topo2D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, topology3_1, TopologyType::Brick},
        {topology3_2, topology3_2, TopologyType::Brick},
        {topology3_3, topology3_3, TopologyType::Brick},
        {topology3_4, topology3_4, TopologyType::Brick},
        {topology3_5, topology3_5, TopologyType::Brick},
        {topology3_1, topology3_2, TopologyType::Brick},
        {topology3_1, topology3_3, TopologyType::Brick},
        {topology3_1, topology3_4, TopologyType::Brick},
        {topology3_1, topology3_5, TopologyType::Brick}};
    for (const auto& [topo1, topo2, ref] : topo3D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Invalid},
        {topology4_2, topology4_2, TopologyType::Invalid},
        {topology4_3, topology4_3, TopologyType::Invalid},
        {topology4_4, topology4_4, TopologyType::Invalid},
        {topology4_5, topology4_5, TopologyType::Invalid},
        {topology4_6, topology4_6, TopologyType::Invalid},
        {topology4_7, topology4_7, TopologyType::Invalid},
        {topology4_8, topology4_8, TopologyType::Invalid},
        {topology4_1, topology4_2, TopologyType::Invalid},
        {topology4_1, topology4_3, TopologyType::Invalid},
        {topology4_1, topology4_4, TopologyType::Invalid},
        {topology4_1, topology4_5, TopologyType::Invalid},
        {topology4_1, topology4_6, TopologyType::Invalid},
        {topology4_2, topology4_3, TopologyType::Invalid},
        {topology4_2, topology4_4, TopologyType::Invalid},
        {topology4_2, topology4_5, TopologyType::Invalid}};

    for (const auto& [topo1, topo2, ref] : topo4D_test_cases) {
      auto topo_type =
          KokkosFFT::Distributed::Impl::get_common_topology_type(topo1, topo2);
      EXPECT_EQ(topo_type, ref)
          << error_get_common_topology_type(topo1, topo2, ref);
    }
  }
}

template <bool is_std_array>
void test_is_topology(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type = std::tuple<topology1D_type, TopologyType>;
  using topology2D_and_ref2D_type = std::tuple<topology2D_type, TopologyType>;
  using topology3D_and_ref3D_type = std::tuple<topology3D_type, TopologyType>;
  using topology4D_and_ref4D_type = std::tuple<topology4D_type, TopologyType>;
  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology is shared
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, TopologyType::Shared},
        {topology1D_type{0}, TopologyType::Empty}};
    for (const auto& [topo, ref_topo_type] : topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 2D topology is slab
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Slab},
        {topology2_2, TopologyType::Slab},
        {topology2D_type{0, nprocs}, TopologyType::Empty},
        {topology2D_type{nprocs, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 3D case
    // Pencil topologies
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Pencil},
        {topology3_2, TopologyType::Pencil},
        {topology3_3, TopologyType::Pencil},
        {topology3_4, TopologyType::Slab},
        {topology3_5, TopologyType::Shared},
        {topology3D_type{0, p1, p2}, TopologyType::Empty},
        {topology3D_type{p0, 0, p2}, TopologyType::Empty},
        {topology3D_type{p0, p1, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo3D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 4D case
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Pencil},
        {topology4_2, TopologyType::Pencil},
        {topology4_3, TopologyType::Pencil},
        {topology4_4, TopologyType::Pencil},
        {topology4_5, TopologyType::Pencil},
        {topology4_6, TopologyType::Pencil},
        {topology4_7, TopologyType::Slab},
        {topology4_8, TopologyType::Shared}};

    for (const auto& [topo, ref_topo_type] : topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }
  } else {
    // 1D topology is slab
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, TopologyType::Slab},
        {topology1D_type{0}, TopologyType::Empty}};
    for (const auto& [topo, ref_topo_type] : topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 2D topology is pencil
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, TopologyType::Pencil},
        {topology2_2, TopologyType::Pencil},
        {topology2D_type{0, nprocs}, TopologyType::Empty},
        {topology2D_type{nprocs, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 3D topology
    std::vector<topology3D_and_ref3D_type> topo3D_test_cases = {
        {topology3_1, TopologyType::Brick},
        {topology3_2, TopologyType::Brick},
        {topology3_3, TopologyType::Brick},
        {topology3_4, TopologyType::Brick},
        {topology3_5, TopologyType::Brick},
        {topology3D_type{0, p1, p2}, TopologyType::Empty},
        {topology3D_type{p0, 0, p2}, TopologyType::Empty},
        {topology3D_type{p0, p1, 0}, TopologyType::Empty}};

    for (const auto& [topo, ref_topo_type] : topo3D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, TopologyType::Invalid},
        {topology4_2, TopologyType::Invalid},
        {topology4_3, TopologyType::Invalid},
        {topology4_4, TopologyType::Invalid},
        {topology4_5, TopologyType::Invalid},
        {topology4_6, TopologyType::Invalid},
        {topology4_7, TopologyType::Invalid},
        {topology4_8, TopologyType::Invalid}};

    for (const auto& [topo, ref_topo_type] : topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo))
              << error_is_topology(topo, topo_type, false);
        }
      }
    }
  }
}

template <bool is_std_array>
void test_are_topologies(std::size_t nprocs) {
  using KokkosFFT::Distributed::Impl::TopologyType;
  using topology1D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 1>,
      KokkosFFT::Distributed::Topology<std::size_t, 1, Kokkos::LayoutRight>>;
  using topology2D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 2>,
      KokkosFFT::Distributed::Topology<std::size_t, 2, Kokkos::LayoutLeft>>;
  using topology3D_r_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>>;
  using topology3D_l_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 3>,
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>>;
  using topology4D_type = std::conditional_t<
      is_std_array, std::array<std::size_t, 4>,
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>>;

  using topology1D_and_ref1D_type =
      std::tuple<topology1D_type, topology1D_type, TopologyType, bool>;
  using topology2D_and_ref2D_type =
      std::tuple<topology2D_type, topology2D_type, TopologyType, bool>;
  using topology3D_rr_and_ref3D_type =
      std::tuple<topology3D_r_type, topology3D_r_type, TopologyType, bool>;
  using topology3D_rl_and_ref3D_type =
      std::tuple<topology3D_r_type, topology3D_l_type, TopologyType, bool>;
  using topology4D_and_ref4D_type =
      std::tuple<topology4D_type, topology4D_type, TopologyType, bool>;

  const std::size_t p0 = 2, p1 = 3, p2 = 4, p3 = 5;

  topology1D_type topology1{nprocs};
  topology2D_type topology2_1{p0, nprocs}, topology2_2{nprocs, p1};
  topology3D_r_type topology3_1{p0, p1, nprocs}, topology3_2{p0, nprocs, p2},
      topology3_3{nprocs, p1, p2}, topology3_4{nprocs, nprocs, p2},
      topology3_5{nprocs, nprocs, nprocs};
  topology3D_l_type topology3_6{p0, p1, nprocs}, topology3_7{p0, nprocs, p2},
      topology3_8{nprocs, p1, p2}, topology3_9{nprocs, nprocs, p2},
      topology3_10{nprocs, nprocs, nprocs};
  topology4D_type topology4_1{p0, p1, nprocs, nprocs},
      topology4_2{p0, nprocs, p2, nprocs}, topology4_3{p0, nprocs, nprocs, p3},
      topology4_4{nprocs, p1, p2, nprocs}, topology4_5{nprocs, p1, nprocs, p3},
      topology4_6{nprocs, nprocs, p2, p3},
      topology4_7{p0, nprocs, nprocs, nprocs},
      topology4_8{nprocs, nprocs, nprocs, nprocs};

  if (nprocs == 1) {
    // 1D topology is shared
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1, TopologyType::Shared, true},
        {topology1D_type{0}, topology1D_type{0}, TopologyType::Empty, true},
        {topology1, topology1D_type{0}, TopologyType::Invalid, false}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 2D topology is slab
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1, TopologyType::Slab, true},
        {topology2_2, topology2_2, TopologyType::Slab, true},
        {topology2_1, topology2_2, TopologyType::Slab, true},
        {topology2_1, topology2D_type{0, nprocs}, TopologyType::Invalid, false},
        {topology2_1, topology2D_type{nprocs, 0}, TopologyType::Invalid,
         false}};

    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 3D case Pencil topologies
    std::vector<topology3D_rr_and_ref3D_type> topo3D_rr_test_cases = {
        {topology3_1, topology3_1, TopologyType::Pencil, true},
        {topology3_2, topology3_2, TopologyType::Pencil, true},
        {topology3_3, topology3_3, TopologyType::Pencil, true},
        {topology3_4, topology3_4, TopologyType::Slab, true},
        {topology3_5, topology3_5, TopologyType::Shared, true},
        {topology3_1, topology3_2, TopologyType::Pencil, true},
        {topology3_2, topology3_3, TopologyType::Pencil, true},
        {topology3_3, topology3_1, TopologyType::Pencil, true},
        {topology3D_r_type{0, p1, p2}, topology3D_r_type{0, p1, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, 0, p2}, topology3D_r_type{p0, 0, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, p1, 0}, topology3D_r_type{p0, p1, 0},
         TopologyType::Empty, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rr_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }
    std::vector<topology3D_rl_and_ref3D_type> topo3D_rl_test_cases = {
        {topology3_1, topology3_6, TopologyType::Pencil, true},
        {topology3_2, topology3_7, TopologyType::Pencil, true},
        {topology3_3, topology3_8, TopologyType::Pencil, true},
        {topology3_4, topology3_9, TopologyType::Slab, true},
        {topology3_5, topology3_10, TopologyType::Shared, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rl_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if (topo_type == ref_topo_type && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 4D case
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Pencil, true},
        {topology4_2, topology4_2, TopologyType::Pencil, true},
        {topology4_3, topology4_3, TopologyType::Pencil, true},
        {topology4_4, topology4_4, TopologyType::Pencil, true},
        {topology4_5, topology4_5, TopologyType::Pencil, true},
        {topology4_6, topology4_6, TopologyType::Pencil, true},
        {topology4_7, topology4_7, TopologyType::Slab, true},
        {topology4_8, topology4_8, TopologyType::Shared, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }
  } else {
    // 1D topology
    std::vector<topology1D_and_ref1D_type> topo1D_test_cases = {
        {topology1, topology1, TopologyType::Slab, true},
        {topology1D_type{0}, topology1D_type{0}, TopologyType::Empty, true},
        {topology1, topology1D_type{0}, TopologyType::Invalid, false}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo1D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 2D topology
    std::vector<topology2D_and_ref2D_type> topo2D_test_cases = {
        {topology2_1, topology2_1, TopologyType::Pencil, true},
        {topology2_2, topology2_2, TopologyType::Pencil, true},
        {topology2_1, topology2_2, TopologyType::Pencil, true},
        {topology2_1, topology2D_type{0, nprocs}, TopologyType::Invalid, false},
        {topology2_1, topology2D_type{nprocs, 0}, TopologyType::Invalid,
         false}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo2D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 3D topology
    std::vector<topology3D_rr_and_ref3D_type> topo3D_rr_test_cases = {
        {topology3_1, topology3_1, TopologyType::Brick, true},
        {topology3_2, topology3_2, TopologyType::Brick, true},
        {topology3_3, topology3_3, TopologyType::Brick, true},
        {topology3_4, topology3_4, TopologyType::Brick, true},
        {topology3_5, topology3_5, TopologyType::Brick, true},
        {topology3_1, topology3_2, TopologyType::Brick, true},
        {topology3_2, topology3_3, TopologyType::Brick, true},
        {topology3_3, topology3_1, TopologyType::Brick, true},
        {topology3D_r_type{0, p1, p2}, topology3D_r_type{0, p1, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, 0, p2}, topology3D_r_type{p0, 0, p2},
         TopologyType::Empty, true},
        {topology3D_r_type{p0, p1, 0}, topology3D_r_type{p0, p1, 0},
         TopologyType::Empty, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rr_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    std::vector<topology3D_rl_and_ref3D_type> topo3D_rl_test_cases = {
        {topology3_1, topology3_6, TopologyType::Brick, true},
        {topology3_2, topology3_7, TopologyType::Brick, true},
        {topology3_3, topology3_8, TopologyType::Brick, true},
        {topology3_4, topology3_9, TopologyType::Brick, true},
        {topology3_5, topology3_10, TopologyType::Brick, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo3D_rl_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }

    // 4D topology
    std::vector<topology4D_and_ref4D_type> topo4D_test_cases = {
        {topology4_1, topology4_1, TopologyType::Invalid, true},
        {topology4_2, topology4_2, TopologyType::Invalid, true},
        {topology4_3, topology4_3, TopologyType::Invalid, true},
        {topology4_4, topology4_4, TopologyType::Invalid, true},
        {topology4_5, topology4_5, TopologyType::Invalid, true},
        {topology4_6, topology4_6, TopologyType::Invalid, true},
        {topology4_7, topology4_7, TopologyType::Invalid, true},
        {topology4_8, topology4_8, TopologyType::Invalid, true}};
    for (const auto& [topo1, topo2, ref_topo_type, is_same_type] :
         topo4D_test_cases) {
      for (int i = 0; i < static_cast<int>(TopologyType::Count); ++i) {
        auto topo_type = static_cast<TopologyType>(i);
        if ((topo_type == ref_topo_type) && is_same_type) {
          EXPECT_TRUE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, true);
        } else {
          EXPECT_FALSE(KokkosFFT::Distributed::Impl::are_specified_topologies(
              topo_type, topo1, topo2))
              << error_are_topologies(topo1, topo2, topo_type, false);
        }
      }
    }
  }
}

void test_slab_in_out_axes_2D(std::size_t nprocs) {
  using topo_type = std::array<std::size_t, 2>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, std::tuple<std::size_t, std::size_t>>;
  topo_type topo0{1, nprocs}, topo1{nprocs, 1}, topo2{nprocs, 7}, topo3{1, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    std::vector<topo_and_ref_type> topo_test_cases = {{topo0, topo1, {0, 1}},
                                                      {topo1, topo0, {1, 0}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_inout_axes] :
         topo_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto inout_axes =
                KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in,
                                                               topo_out);
          },
          std::runtime_error);
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {{topo0, topo1, {0, 1}},
                                                      {topo1, topo0, {1, 0}}};

    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_test_cases) {
      auto inout_axes =
          KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
      EXPECT_EQ(inout_axes, ref_inout_axes)
          << error_in_out_axes(topo_in, topo_out, inout_axes, ref_inout_axes);
    }
  }

  // Failure tests because of shape mismatch (or size 1 case)
  std::vector<topo_and_ref_type> topo_failure_test_cases = {
      {topo0, topo2, {0, 1}}, {topo0, topo3, {0, 1}}};
  for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_inout_axes] :
       topo_failure_test_cases) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axes =
              KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
        },
        std::runtime_error);
  }
}

void test_slab_in_out_axes_3D(std::size_t nprocs) {
  using topo_type = std::array<std::size_t, 3>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, std::tuple<std::size_t, std::size_t>>;
  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1},
      topo3{1, nprocs, 7}, topo4{1, 1, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo1, {0, 1}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {1, 0}},
        {topo1, topo2, {1, 2}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {2, 1}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_inout_axes] :
         topo_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto inout_axes =
                KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in,
                                                               topo_out);
          },
          std::runtime_error);
    }
  } else {
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo1, {1, 2}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {2, 1}},
        {topo1, topo2, {0, 1}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {1, 0}}};
    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_test_cases) {
      auto inout_axes =
          KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
      EXPECT_EQ(inout_axes, ref_inout_axes)
          << error_in_out_axes(topo_in, topo_out, inout_axes, ref_inout_axes);
    }
  }

  // Failure tests because of shape mismatch (or size 1 case)
  std::vector<topo_and_ref_type> topo_failure_test_cases = {
      {topo0, topo3, {0, 1}}, {topo0, topo4, {0, 1}}};
  for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_inout_axes] :
       topo_failure_test_cases) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axes =
              KokkosFFT::Distributed::Impl::slab_in_out_axes(topo_in, topo_out);
        },
        std::runtime_error);
  }
}

void test_decompose_axes_slab(std::size_t nprocs) {
  using topo3D_type       = std::array<std::size_t, 3>;
  using topo4D_type       = std::array<std::size_t, 4>;
  using axes_type         = std::array<std::size_t, 3>;
  using vec_topo3D_type   = std::vector<topo3D_type>;
  using vec_topo4D_type   = std::vector<topo4D_type>;
  using vec_axes_type     = std::vector<std::size_t>;
  using vec_vec_axes_type = std::vector<vec_axes_type>;
  using topo3D_and_ref_type =
      std::tuple<vec_topo3D_type, axes_type, vec_vec_axes_type>;
  using topo4D_and_ref_type =
      std::tuple<vec_topo4D_type, axes_type, vec_vec_axes_type>;

  // 3D topologies
  topo3D_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1};

  // 4D topologies
  topo4D_type topo3{1, 1, 1, nprocs}, topo4{1, 1, nprocs, 1};

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};

  std::vector<axes_type> all_axes{axes012, axes021, axes102,
                                  axes120, axes201, axes210};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      vec_vec_axes_type ref_all_axes2{KokkosFFT::Impl::to_vector(axes), {}},
          ref_all_axes3{KokkosFFT::Impl::to_vector(axes), {}, {}};
      // 3D case
      std::vector<topo3D_and_ref_type> topo3D_test_cases = {
          {vec_topo3D_type{topo0, topo1}, axes, ref_all_axes2},
          {vec_topo3D_type{topo0, topo2}, axes, ref_all_axes2},
          {vec_topo3D_type{topo1, topo2}, axes, ref_all_axes2},
          {vec_topo3D_type{topo2, topo0, topo2}, axes, ref_all_axes3}};
      for (const auto& [topos3D, axes3D, ref_axes3D] : topo3D_test_cases) {
        auto all_axes_3D =
            KokkosFFT::Distributed::Impl::decompose_axes(topos3D, axes3D);
        EXPECT_EQ(all_axes_3D, ref_axes3D)
            << error_decompose_axes(topos3D, axes3D, ref_axes3D);
      }

      // 4D case
      std::vector<topo4D_and_ref_type> topo4D_test_cases = {
          {vec_topo4D_type{topo3, topo4}, axes, ref_all_axes2},
          {vec_topo4D_type{topo4, topo3}, axes, ref_all_axes2}};

      for (const auto& [topos4D, axes4D, ref_axes4D] : topo4D_test_cases) {
        auto all_axes_4D =
            KokkosFFT::Distributed::Impl::decompose_axes(topos4D, axes4D);
        EXPECT_EQ(all_axes_4D, ref_axes4D)
            << error_decompose_axes(topos4D, axes4D, ref_axes4D);
      }
    }
  } else {
    vec_vec_axes_type ref_all_axes_2_0_2{vec_axes_type{2, 1}, vec_axes_type{0},
                                         vec_axes_type{}},
        ref_all_axes_3_4{vec_axes_type{0, 1, 2}, vec_axes_type{}},
        ref_all_axes_4_3_ax210{vec_axes_type{1, 0}, vec_axes_type{2}},
        ref_all_axes_4_3_ax012{vec_axes_type{}, vec_axes_type{0, 1, 2}};
    // 3D case
    std::vector<topo3D_and_ref_type> topo3D_test_cases{
        {vec_topo3D_type{topo2, topo0, topo2}, axes021, ref_all_axes_2_0_2}};
    for (const auto& [topos3D, axes3D, ref_axes3D] : topo3D_test_cases) {
      auto all_axes_3D =
          KokkosFFT::Distributed::Impl::decompose_axes(topos3D, axes3D);
      EXPECT_EQ(all_axes_3D, ref_axes3D)
          << error_decompose_axes(topos3D, axes3D, ref_axes3D);
    }

    // 4D case
    std::vector<topo4D_and_ref_type> topo4D_test_cases{
        {vec_topo4D_type{topo3, topo4}, axes012, ref_all_axes_3_4},
        {vec_topo4D_type{topo4, topo3}, axes210, ref_all_axes_4_3_ax210},
        {vec_topo4D_type{topo4, topo3}, axes012, ref_all_axes_4_3_ax012}};

    for (const auto& [topos4D, axes4D, ref_axes4D] : topo4D_test_cases) {
      auto all_axes_4D =
          KokkosFFT::Distributed::Impl::decompose_axes(topos4D, axes4D);
      EXPECT_EQ(all_axes_4D, ref_axes4D)
          << error_decompose_axes(topos4D, axes4D, ref_axes4D);
    }
  }
}

void test_decompose_axes_pencil(std::size_t nprocs) {
  using topo_type         = std::array<std::size_t, 3>;
  using axes_type         = std::array<std::size_t, 3>;
  using vec_axes_type     = std::vector<std::size_t>;
  using vec_topo_type     = std::vector<topo_type>;
  using vec_vec_axes_type = std::vector<vec_axes_type>;
  using topo_and_ref_type =
      std::tuple<vec_topo_type, axes_type, vec_vec_axes_type>;
  std::size_t np0 = 4;

  // 3D topologies
  topo_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0}, topo2{np0, nprocs, 1},
      topo3{nprocs, np0, 1}, topo4{np0, 1, nprocs};

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};
  std::vector<axes_type> all_axes = {axes012, axes021, axes102,
                                     axes120, axes201, axes210};
  if (nprocs == 1) {
    // Slab geometry
    std::vector<topo_and_ref_type> topo_test_cases = {
        {std::vector<topo_type>{topo0, topo2, topo4, topo2, topo0},
         axes012,
         {{}, vec_axes_type{1, 2}, {}, {}, vec_axes_type{0}}},
        {std::vector<topo_type>{topo0, topo1, topo3, topo1, topo0},
         axes021,
         {vec_axes_type{1}, {}, vec_axes_type{0, 2}, {}, {}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1, topo0},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{1, 0}, {}, {}}},
        {std::vector<topo_type>{topo0, topo1, topo0, topo2, topo0},
         axes201,
         {vec_axes_type{0, 1}, {}, {}, vec_axes_type{2}, {}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{1, 0}, {}}}};

    for (const auto& [topos, axes, ref_axes] : topo_test_cases) {
      auto all_axes = KokkosFFT::Distributed::Impl::decompose_axes(topos, axes);
      EXPECT_EQ(all_axes, ref_axes)
          << error_decompose_axes(topos, axes, ref_axes);
    }
  } else {
    // Pencil geometry
    std::vector<topo_and_ref_type> topo_test_cases = {
        {std::vector<topo_type>{topo0, topo2, topo4, topo2, topo0},
         axes012,
         {{}, vec_axes_type{2}, vec_axes_type{1}, {}, vec_axes_type{0}}},
        {std::vector<topo_type>{topo0, topo1, topo3, topo1, topo0},
         axes021,
         {{}, vec_axes_type{1}, vec_axes_type{2}, {}, vec_axes_type{0}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1, topo0},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{0}, vec_axes_type{1}, {}}},
        {std::vector<topo_type>{topo0, topo1, topo0, topo2, topo0},
         axes201,
         {{}, vec_axes_type{1}, vec_axes_type{0}, vec_axes_type{2}, {}}},
        {std::vector<topo_type>{topo0, topo2, topo0, topo1},
         axes102,
         {{}, vec_axes_type{2}, vec_axes_type{0}, vec_axes_type{1}}}};

    for (const auto& [topos, axes, ref_axes] : topo_test_cases) {
      auto all_axes = KokkosFFT::Distributed::Impl::decompose_axes(topos, axes);
      EXPECT_EQ(all_axes, ref_axes)
          << error_decompose_axes(topos, axes, ref_axes);
    }
  }
}

void test_pencil_in_out_axes_3D(std::size_t nprocs) {
  using topo_type = std::array<std::size_t, 3>;
  using topo_and_ref_type =
      std::tuple<topo_type, topo_type, std::tuple<std::size_t, std::size_t>>;
  topo_type topo0{1, 1, nprocs}, topo1{1, nprocs, 1}, topo2{nprocs, 1, 1},
      topo3{nprocs, 1, 2}, topo4{nprocs, 2, 1};

  if (nprocs == 1) {
    // Failure tests because of size 1 case
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, {1, 2}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {2, 1}},
        {topo1, topo2, {0, 1}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {1, 0}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_in_out] :
         topo_and_ref_vec) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto inout_axis =
                KokkosFFT::Distributed::Impl::pencil_in_out_axes(topo_in,
                                                                 topo_out);
          },
          std::runtime_error);
    }
  } else {
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, {1, 2}}, {topo0, topo2, {0, 2}}, {topo1, topo0, {2, 1}},
        {topo1, topo2, {0, 1}}, {topo2, topo0, {2, 0}}, {topo2, topo1, {1, 0}},
        {topo3, topo4, {1, 2}}, {topo4, topo3, {2, 1}}};
    for (const auto& [topo_in, topo_out, ref_inout_axes] : topo_and_ref_vec) {
      auto inout_axes =
          KokkosFFT::Distributed::Impl::pencil_in_out_axes(topo_in, topo_out);
      EXPECT_EQ(inout_axes, ref_inout_axes)
          << error_in_out_axes(topo_in, topo_out, inout_axes, ref_inout_axes);
    }
  }

  // Failure tests because of shape mismatch (or size 1 case)
  std::vector<topo_and_ref_type> topo_failure_test_cases = {
      {topo3, topo0, {0, 1}}, {topo3, topo1, {0, 1}}, {topo3, topo2, {0, 2}}};
  for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_inout_axes] :
       topo_failure_test_cases) {
    EXPECT_THROW(
        {
          [[maybe_unused]] auto inout_axes =
              KokkosFFT::Distributed::Impl::pencil_in_out_axes(topo_in,
                                                               topo_out);
        },
        std::runtime_error);
  }
}

/// \brief Tests for get_all_slab_topologies for 1D FFT on 2D Views, which
/// returns the list of all valid slab topologies for a given input topology,
/// output topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform
template <bool is_R2C>
void test_get_all_slab_topologies1D_2DView(
    std::size_t nprocs, const std::array<std::size_t, 2>& gin_extents) {
  using topo_type          = std::array<std::size_t, 2>;
  using axes_type          = std::array<std::size_t, 1>;
  using topo_and_axes_type = std::array<std::size_t, 3>;
  using ref_vec_type       = std::vector<std::size_t>;
  using topo_and_ref_type  = std::map<topo_and_axes_type, ref_vec_type>;

  std::vector<std::size_t> all_dimensions{0, 1};

  if (nprocs == 1) {
    for (const auto& idx_axis : all_dimensions) {
      for (const auto& idx_topo_in : all_dimensions) {
        for (const auto& idx_topo_out : all_dimensions) {
          axes_type axes{idx_axis};
          topo_type topo_in  = to_slab_topology<2>(idx_topo_in, nprocs);
          topo_type topo_out = to_slab_topology<2>(idx_topo_out, nprocs);

          EXPECT_THROW(
              {
                auto gout_extents = gin_extents;
                gout_extents.at(idx_axis) =
                    KokkosFFT::Impl::extent_after_transform(
                        gout_extents[idx_axis], is_R2C);
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        gin_extents, gout_extents, topo_in, topo_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    topo_and_ref_type topo_test_cases = {
        {topo_and_axes_type{0, 0, 0}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 0, 1}, ref_vec_type{0}},
        {topo_and_axes_type{0, 1, 0}, ref_vec_type{0, 1}},
        {topo_and_axes_type{0, 1, 1}, ref_vec_type{0, 1}},
        {topo_and_axes_type{1, 0, 0}, ref_vec_type{1, 0}},
        {topo_and_axes_type{1, 0, 1}, ref_vec_type{1, 0}},
        {topo_and_axes_type{1, 1, 0}, ref_vec_type{1}},
        {topo_and_axes_type{1, 1, 1}, ref_vec_type{1, 0, 1}}};

    for (const auto& [topo_and_axes, ref_indices] : topo_test_cases) {
      auto [idx_topo_in, idx_topo_out, idx_axis] = topo_and_axes;
      axes_type axes{idx_axis};
      topo_type topo_in  = to_slab_topology<2>(idx_topo_in, nprocs);
      topo_type topo_out = to_slab_topology<2>(idx_topo_out, nprocs);
      auto ref_topos     = to_vec_topology<2>(ref_indices, nprocs);

      auto gout_extents        = gin_extents;
      gout_extents.at(axes[0]) = KokkosFFT::Impl::extent_after_transform(
          gout_extents[axes[0]], is_R2C);

      auto in_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
          gin_extents, topo_in);
      auto out_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
          gout_extents, topo_out);

      auto in_size  = KokkosFFT::Impl::total_size(in_extents);
      auto out_size = KokkosFFT::Impl::total_size(out_extents);

      bool transpose_needed   = topo_in.at(idx_axis) > 1;
      bool transpose_possible = false;
      for (const auto& axis : all_dimensions) {
        if (axis == idx_axis) continue;
        if (gout_extents[axis] >= nprocs) {
          transpose_possible = true;
          break;
        }
      }

      if (in_size == 0 || out_size == 0 ||
          (transpose_needed && !transpose_possible)) {
        EXPECT_THROW(
            {
              [[maybe_unused]] auto topos =
                  KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                      gin_extents, gout_extents, topo_in, topo_out, axes);
            },
            std::runtime_error);
      } else {
        auto [topos, all_axes] =
            KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                gin_extents, gout_extents, topo_in, topo_out, axes);

        // The FFT axes must be identical with decompose_axes
        EXPECT_EQ(all_axes,
                  KokkosFFT::Distributed::Impl::decompose_axes(topos, axes));

        EXPECT_EQ(topos, ref_topos)
            << error_all_topologies(gin_extents, gout_extents, topo_in,
                                    topo_out, axes, topos, ref_topos);
      }
    }
  }
}

/// \brief Tests for get_all_slab_topologies for 1D FFT, which returns the list
/// of all valid slab topologies for a given input topology, output topology,
/// and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (2D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_slab_topologies1D_impl(
    std::size_t nprocs, const std::array<std::size_t, DIM>& gin_extents) {
  using topo_type = std::array<std::size_t, DIM>;
  using axes_type = std::array<std::size_t, 1>;

  std::vector<std::size_t> all_dimensions(DIM);
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);

  for (const auto& idx_topo_in : all_dimensions) {
    for (const auto& idx_topo_out : all_dimensions) {
      for (const auto& idx_axis : all_dimensions) {
        axes_type axes{idx_axis};
        topo_type topo_in         = to_slab_topology<DIM>(idx_topo_in, nprocs);
        topo_type topo_out        = to_slab_topology<DIM>(idx_topo_out, nprocs);
        auto gout_extents         = gin_extents;
        gout_extents.at(idx_axis) = KokkosFFT::Impl::extent_after_transform(
            gout_extents[idx_axis], is_R2C);

        if (nprocs == 1) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto all_slab_topologies =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        gin_extents, gout_extents, topo_in, topo_out, axes);
              },
              std::runtime_error);
        } else {
          auto in_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
              gin_extents, topo_in);
          auto out_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
              gout_extents, topo_out);

          auto in_size            = KokkosFFT::Impl::total_size(in_extents);
          auto out_size           = KokkosFFT::Impl::total_size(out_extents);
          bool transpose_needed   = topo_in.at(idx_axis) > 1;
          bool transpose_possible = false;
          for (const auto& axis : all_dimensions) {
            if (axis == idx_axis) continue;
            if (gout_extents[axis] >= nprocs) {
              transpose_possible = true;
              break;
            }
          }

          if (in_size == 0 || out_size == 0 ||
              (transpose_needed && !transpose_possible)) {
            EXPECT_THROW(
                {
                  [[maybe_unused]] auto topos =
                      KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                          gin_extents, gout_extents, topo_in, topo_out, axes);
                },
                std::runtime_error);
          } else {
            // This must return length 1-3 vectors
            auto [topos, all_axes] =
                KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                    gin_extents, gout_extents, topo_in, topo_out, axes);

            // The FFT axes must be identical with decompose_axes
            EXPECT_EQ(all_axes, KokkosFFT::Distributed::Impl::decompose_axes(
                                    topos, axes));

            // In 1D, we can have at most 3 topologies
            // (in, out, and 1 transpose)
            std::size_t max_size = std::min(DIM, std::size_t(3));
            EXPECT_GE(topos.size(), 1);
            EXPECT_LE(topos.size(), max_size);
            EXPECT_FALSE(has_consecutive_duplicates(topos));
          }
        }
      }
    }
  }
}

/// \brief Tests for get_all_slab_topologies for 2D FFT on 2D Views, which
/// returns the list of all valid slab topologies for a given input topology,
/// output topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform
template <bool is_R2C>
void test_get_all_slab_topologies2D_2DView(
    std::size_t nprocs, const std::array<std::size_t, 2>& gin_extents) {
  using topo_type          = std::array<std::size_t, 2>;
  using axes_type          = std::array<std::size_t, 2>;
  using topo_and_axes_type = std::array<std::size_t, 4>;
  using ref_vec_type       = std::vector<std::size_t>;
  using topo_and_ref_type  = std::map<topo_and_axes_type, ref_vec_type>;

  std::vector<std::size_t> all_dimensions{0, 1};
  if (nprocs == 1) {
    for (const auto& idx_topo_in : all_dimensions) {
      for (const auto& idx_topo_out : all_dimensions) {
        for (const auto& axis0 : all_dimensions) {
          for (const auto& axis1 : all_dimensions) {
            if (axis0 == axis1) continue;
            axes_type axes{axis0, axis1};
            topo_type topo_in  = to_slab_topology<2>(idx_topo_in, nprocs);
            topo_type topo_out = to_slab_topology<2>(idx_topo_out, nprocs);
            EXPECT_THROW(
                {
                  auto gout_extents = gin_extents;
                  gout_extents.at(axis1) =
                      KokkosFFT::Impl::extent_after_transform(
                          gout_extents[axis1], is_R2C);
                  [[maybe_unused]] auto all_slab_topologies =
                      KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                          gin_extents, gout_extents, topo_in, topo_out, axes);
                },
                std::runtime_error);
          }
        }
      }
    }
  } else {
    topo_and_ref_type topo_test_cases = {
        {topo_and_axes_type{0, 0, 0, 1}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 0, 1, 0}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 1, 0, 1}, ref_vec_type{0, 1}},
        {topo_and_axes_type{0, 1, 1, 0}, ref_vec_type{0, 1, 0, 1}},
        {topo_and_axes_type{1, 0, 0, 1}, ref_vec_type{1, 0, 1, 0}},
        {topo_and_axes_type{1, 0, 1, 0}, ref_vec_type{1, 0}},
        {topo_and_axes_type{1, 1, 0, 1}, ref_vec_type{1, 0, 1}},
        {topo_and_axes_type{1, 1, 1, 0}, ref_vec_type{1, 0, 1}}};
    for (const auto& [topo_and_axes, ref_indices] : topo_test_cases) {
      auto [idx_topo_in, idx_topo_out, axis0, axis1] = topo_and_axes;
      axes_type axes{axis0, axis1};
      topo_type topo_in  = to_slab_topology<2>(idx_topo_in, nprocs);
      topo_type topo_out = to_slab_topology<2>(idx_topo_out, nprocs);
      auto ref_topos     = to_vec_topology<2>(ref_indices, nprocs);

      auto gout_extents      = gin_extents;
      gout_extents.at(axis1) = KokkosFFT::Impl::extent_after_transform(
          gout_extents.at(axis1), is_R2C);

      auto in_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
          gin_extents, topo_in);
      auto out_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
          gout_extents, topo_out);

      auto in_size  = KokkosFFT::Impl::total_size(in_extents);
      auto out_size = KokkosFFT::Impl::total_size(out_extents);

      bool transpose_needed = false;
      for (const auto& ax : axes) {
        if (topo_in.at(ax) > 1) {
          transpose_needed = true;
          break;
        }
      }
      bool transpose_possible = true;
      for (const auto& ax : axes) {
        bool transpose_possible_for_ax = false;
        for (std::size_t i = 0; i < 2; ++i) {
          if (i == ax) continue;
          if (gout_extents.at(i) >= nprocs) {
            transpose_possible_for_ax = true;
            break;
          }
        }
        if (!transpose_possible_for_ax) {
          transpose_possible = false;
          break;
        }
      }

      if (in_size == 0 || out_size == 0 ||
          (transpose_needed && !transpose_possible)) {
        EXPECT_THROW(
            {
              [[maybe_unused]] auto topos =
                  KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                      gin_extents, gout_extents, topo_in, topo_out, axes);
            },
            std::runtime_error);
      } else {
        auto [topos, all_axes] =
            KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                gin_extents, gout_extents, topo_in, topo_out, axes);

        // The FFT axes must be identical with decompose_axes
        EXPECT_EQ(all_axes,
                  KokkosFFT::Distributed::Impl::decompose_axes(topos, axes));

        EXPECT_EQ(topos, ref_topos)
            << error_all_topologies(gin_extents, gout_extents, topo_in,
                                    topo_out, axes, topos, ref_topos);
      }
    }
  }
}

/// \brief Tests for get_all_slab_topologies for 2D FFT, which returns the list
/// of all valid slab topologies for a given input topology, output topology,
/// and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (2D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_slab_topologies2D_impl(
    std::size_t nprocs, const std::array<std::size_t, DIM>& gin_extents) {
  using topo_type = std::array<std::size_t, DIM>;
  using axes_type = std::array<std::size_t, 2>;

  std::vector<std::size_t> all_dimensions(DIM);
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);

  for (const auto& idx_topo_in : all_dimensions) {
    for (const auto& idx_topo_out : all_dimensions) {
      for (const auto& axis0 : all_dimensions) {
        for (const auto& axis1 : all_dimensions) {
          if (axis0 == axis1) continue;
          axes_type axes{axis0, axis1};
          topo_type topo_in      = to_slab_topology<DIM>(idx_topo_in, nprocs);
          topo_type topo_out     = to_slab_topology<DIM>(idx_topo_out, nprocs);
          auto gout_extents      = gin_extents;
          gout_extents.at(axis1) = KokkosFFT::Impl::extent_after_transform(
              gout_extents[axis1], is_R2C);

          if (nprocs == 1) {
            EXPECT_THROW(
                {
                  [[maybe_unused]] auto all_slab_topologies =
                      KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                          gin_extents, gout_extents, topo_in, topo_out, axes);
                },
                std::runtime_error);
          } else {
            auto in_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
                gin_extents, topo_in);
            auto out_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
                gout_extents, topo_out);

            auto in_size          = KokkosFFT::Impl::total_size(in_extents);
            auto out_size         = KokkosFFT::Impl::total_size(out_extents);
            bool transpose_needed = false;
            for (const auto& ax : axes) {
              if (topo_in.at(ax) > 1) {
                transpose_needed = true;
                break;
              }
            }
            bool transpose_possible = true;
            for (const auto& ax : axes) {
              bool transpose_possible_for_ax = false;
              for (std::size_t i = 0; i < DIM; ++i) {
                if (i == ax) continue;
                if (gout_extents.at(i) >= nprocs) {
                  transpose_possible_for_ax = true;
                  break;
                }
              }
              if (!transpose_possible_for_ax) {
                transpose_possible = false;
                break;
              }
            }

            if (in_size == 0 || out_size == 0 ||
                (transpose_needed && !transpose_possible)) {
              EXPECT_THROW(
                  {
                    [[maybe_unused]] auto topos =
                        KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                            gin_extents, gout_extents, topo_in, topo_out, axes);
                  },
                  std::runtime_error);
            } else {
              // This must return length 1-3 vectors
              auto [topos, all_axes] =
                  KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                      gin_extents, gout_extents, topo_in, topo_out, axes);

              // The FFT axes must be identical with decompose_axes
              EXPECT_EQ(all_axes, KokkosFFT::Distributed::Impl::decompose_axes(
                                      topos, axes));
              // In 2D, we can have at most 4 topologies
              // (in, out, and 2 transposes)
              std::size_t max_size = std::min(DIM, std::size_t(4));
              EXPECT_GE(topos.size(), 1);
              EXPECT_LE(topos.size(), max_size);
              EXPECT_FALSE(has_consecutive_duplicates(topos));
            }
          }
        }
      }
    }
  }
}

/// \brief Tests for get_all_slab_topologies for 3D FFT on 3D Views, which
/// returns the list of all valid slab topologies for a given input topology,
/// output topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in]gin_extents The global input extents for the transform
template <bool is_R2C>
void test_get_all_slab_topologies3D_3DView(
    std::size_t nprocs, const std::array<std::size_t, 3>& gin_extents) {
  using topo_type          = std::array<std::size_t, 3>;
  using axes_type          = std::array<std::size_t, 3>;
  using topo_and_axes_type = std::array<std::size_t, 5>;
  using ref_vec_type       = std::vector<std::size_t>;
  using topo_and_ref_type  = std::map<topo_and_axes_type, ref_vec_type>;

  std::vector<std::size_t> all_dimensions{0, 1, 2};
  if (nprocs == 1) {
    for (const auto& idx_topo_in : all_dimensions) {
      for (const auto& idx_topo_out : all_dimensions) {
        for (const auto& axis0 : all_dimensions) {
          for (const auto& axis1 : all_dimensions) {
            for (const auto& axis2 : all_dimensions) {
              if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
              axes_type axes{axis0, axis1, axis2};
              topo_type topo_in  = to_slab_topology<3>(idx_topo_in, nprocs);
              topo_type topo_out = to_slab_topology<3>(idx_topo_out, nprocs);
              EXPECT_THROW(
                  {
                    auto gout_extents = gin_extents;
                    gout_extents.at(axis2) =
                        KokkosFFT::Impl::extent_after_transform(
                            gout_extents.at(axis2), is_R2C);
                    [[maybe_unused]] auto all_slab_topologies =
                        KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                            gin_extents, gout_extents, topo_in, topo_out, axes);
                  },
                  std::runtime_error);
            }
          }
        }
      }
    }
  } else {
    topo_and_ref_type topo_test_cases = {
        {topo_and_axes_type{0, 0, 0, 1, 2}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 0, 0, 2, 1}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 0, 1, 0, 2}, ref_vec_type{0, 2, 0}},
        {topo_and_axes_type{0, 0, 1, 2, 0}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 0, 2, 0, 1}, ref_vec_type{0, 1, 0}},
        {topo_and_axes_type{0, 0, 2, 1, 0}, ref_vec_type{0, 2, 0}},
        {topo_and_axes_type{0, 1, 0, 1, 2}, ref_vec_type{0, 1}},
        {topo_and_axes_type{0, 1, 0, 2, 1}, ref_vec_type{0, 1}},
        {topo_and_axes_type{0, 1, 1, 0, 2}, ref_vec_type{0, 2, 1}},
        {topo_and_axes_type{0, 1, 1, 2, 0}, ref_vec_type{0, 1, 0, 1}},
        {topo_and_axes_type{0, 1, 2, 0, 1}, ref_vec_type{0, 1}},
        {topo_and_axes_type{0, 1, 2, 1, 0}, ref_vec_type{0, 2, 1}},
        {topo_and_axes_type{0, 2, 0, 1, 2}, ref_vec_type{0, 2}},
        {topo_and_axes_type{0, 2, 0, 2, 1}, ref_vec_type{0, 2}},
        {topo_and_axes_type{0, 2, 1, 0, 2}, ref_vec_type{0, 2}},
        {topo_and_axes_type{0, 2, 1, 2, 0}, ref_vec_type{0, 1, 2}},
        {topo_and_axes_type{0, 2, 2, 0, 1}, ref_vec_type{0, 1, 2}},
        {topo_and_axes_type{0, 2, 2, 1, 0}, ref_vec_type{0, 2, 0, 2}},
        {topo_and_axes_type{1, 0, 0, 1, 2}, ref_vec_type{1, 2, 0}},
        {topo_and_axes_type{1, 0, 0, 2, 1}, ref_vec_type{1, 0, 1, 0}},
        {topo_and_axes_type{1, 0, 1, 0, 2}, ref_vec_type{1, 0}},
        {topo_and_axes_type{1, 0, 1, 2, 0}, ref_vec_type{1, 0}},
        {topo_and_axes_type{1, 0, 2, 0, 1}, ref_vec_type{1, 2, 0}},
        {topo_and_axes_type{1, 0, 2, 1, 0}, ref_vec_type{1, 0}},
        {topo_and_axes_type{1, 1, 0, 1, 2}, ref_vec_type{1, 2, 1}},
        {topo_and_axes_type{1, 1, 0, 2, 1}, ref_vec_type{1, 0, 1}},
        {topo_and_axes_type{1, 1, 1, 0, 2}, ref_vec_type{1, 0, 1}},
        {topo_and_axes_type{1, 1, 1, 2, 0}, ref_vec_type{1, 0, 1}},
        {topo_and_axes_type{1, 1, 2, 0, 1}, ref_vec_type{1, 2, 1}},
        {topo_and_axes_type{1, 1, 2, 1, 0}, ref_vec_type{1, 0, 1}},
        {topo_and_axes_type{1, 2, 0, 1, 2}, ref_vec_type{1, 2}},
        {topo_and_axes_type{1, 2, 0, 2, 1}, ref_vec_type{1, 0, 2}},
        {topo_and_axes_type{1, 2, 1, 0, 2}, ref_vec_type{1, 2}},
        {topo_and_axes_type{1, 2, 1, 2, 0}, ref_vec_type{1, 2}},
        {topo_and_axes_type{1, 2, 2, 0, 1}, ref_vec_type{1, 2, 0, 2}},
        {topo_and_axes_type{1, 2, 2, 1, 0}, ref_vec_type{1, 0, 2}},
        {topo_and_axes_type{2, 0, 0, 1, 2}, ref_vec_type{2, 0, 1, 0}},
        {topo_and_axes_type{2, 0, 0, 2, 1}, ref_vec_type{2, 1, 0}},
        {topo_and_axes_type{2, 0, 1, 0, 2}, ref_vec_type{2, 1, 0}},
        {topo_and_axes_type{2, 0, 1, 2, 0}, ref_vec_type{2, 0}},
        {topo_and_axes_type{2, 0, 2, 0, 1}, ref_vec_type{2, 0}},
        {topo_and_axes_type{2, 0, 2, 1, 0}, ref_vec_type{2, 0}},
        {topo_and_axes_type{2, 1, 0, 1, 2}, ref_vec_type{2, 0, 1}},
        {topo_and_axes_type{2, 1, 0, 2, 1}, ref_vec_type{2, 1}},
        {topo_and_axes_type{2, 1, 1, 0, 2}, ref_vec_type{2, 1, 0, 1}},
        {topo_and_axes_type{2, 1, 1, 2, 0}, ref_vec_type{2, 0, 1}},
        {topo_and_axes_type{2, 1, 2, 0, 1}, ref_vec_type{2, 1}},
        {topo_and_axes_type{2, 1, 2, 1, 0}, ref_vec_type{2, 1}},
        {topo_and_axes_type{2, 2, 0, 1, 2}, ref_vec_type{2, 0, 2}},
        {topo_and_axes_type{2, 2, 0, 2, 1}, ref_vec_type{2, 1, 2}},
        {topo_and_axes_type{2, 2, 1, 0, 2}, ref_vec_type{2, 1, 2}},
        {topo_and_axes_type{2, 2, 1, 2, 0}, ref_vec_type{2, 0, 2}},
        {topo_and_axes_type{2, 2, 2, 0, 1}, ref_vec_type{2, 0, 2}},
        {topo_and_axes_type{2, 2, 2, 1, 0}, ref_vec_type{2, 0, 2}},
    };

    // Update reference when some dimensions are smaller than nprocs
    if (gin_extents.at(0) < nprocs) {
      topo_test_cases.at(topo_and_axes_type{1, 1, 0, 2, 1}) =
          ref_vec_type{1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{1, 1, 1, 0, 2}) =
          ref_vec_type{1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{1, 1, 1, 2, 0}) =
          ref_vec_type{1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{1, 1, 2, 1, 0}) =
          ref_vec_type{1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{1, 2, 0, 2, 1}) =
          ref_vec_type{1, 2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{1, 2, 2, 0, 1}) =
          ref_vec_type{1, 2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{1, 2, 2, 1, 0}) =
          ref_vec_type{1, 2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{2, 1, 0, 1, 2}) =
          ref_vec_type{2, 1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{2, 1, 1, 0, 2}) =
          ref_vec_type{2, 1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{2, 1, 1, 2, 0}) =
          ref_vec_type{2, 1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{2, 2, 0, 1, 2}) =
          ref_vec_type{2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{2, 2, 1, 2, 0}) =
          ref_vec_type{2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{2, 2, 2, 0, 1}) =
          ref_vec_type{2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{2, 2, 2, 1, 0}) =
          ref_vec_type{2, 1, 2};
    } else if (KokkosFFT::Impl::extent_after_transform(gin_extents.at(0),
                                                       is_R2C) < nprocs) {
      topo_test_cases.at(topo_and_axes_type{0, 1, 1, 2, 0}) =
          ref_vec_type{0, 1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{0, 2, 2, 1, 0}) =
          ref_vec_type{0, 2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{1, 1, 1, 2, 0}) =
          ref_vec_type{1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{1, 1, 2, 1, 0}) =
          ref_vec_type{1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{1, 2, 2, 1, 0}) =
          ref_vec_type{1, 2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{2, 1, 1, 2, 0}) =
          ref_vec_type{2, 1, 2, 1};
      topo_test_cases.at(topo_and_axes_type{2, 2, 1, 2, 0}) =
          ref_vec_type{2, 1, 2};
      topo_test_cases.at(topo_and_axes_type{2, 2, 2, 1, 0}) =
          ref_vec_type{2, 1, 2};
    }

    if (gin_extents.at(1) < nprocs) {
      topo_test_cases.at(topo_and_axes_type{0, 0, 0, 1, 2}) =
          ref_vec_type{0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{0, 0, 0, 2, 1}) =
          ref_vec_type{0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{0, 0, 1, 2, 0}) =
          ref_vec_type{0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{0, 0, 2, 0, 1}) =
          ref_vec_type{0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{0, 2, 1, 2, 0}) =
          ref_vec_type{0, 2, 0, 2};
      topo_test_cases.at(topo_and_axes_type{0, 2, 2, 0, 1}) =
          ref_vec_type{0, 2, 0, 2};
      topo_test_cases.at(topo_and_axes_type{2, 0, 0, 1, 2}) =
          ref_vec_type{2, 0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{2, 0, 0, 2, 1}) =
          ref_vec_type{2, 0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{2, 0, 1, 0, 2}) =
          ref_vec_type{2, 0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{2, 2, 0, 2, 1}) =
          ref_vec_type{2, 0, 2};
      topo_test_cases.at(topo_and_axes_type{2, 2, 1, 0, 2}) =
          ref_vec_type{2, 0, 2};
    } else if (KokkosFFT::Impl::extent_after_transform(gin_extents.at(1),
                                                       is_R2C) < nprocs) {
      topo_test_cases.at(topo_and_axes_type{0, 0, 0, 2, 1}) =
          ref_vec_type{0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{0, 0, 2, 0, 1}) =
          ref_vec_type{0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{0, 2, 2, 0, 1}) =
          ref_vec_type{0, 2, 0, 2};
      topo_test_cases.at(topo_and_axes_type{1, 0, 0, 2, 1}) =
          ref_vec_type{1, 0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{2, 0, 0, 2, 1}) =
          ref_vec_type{2, 0, 2, 0};
      topo_test_cases.at(topo_and_axes_type{2, 2, 0, 2, 1}) =
          ref_vec_type{2, 0, 2};
    }

    if (gin_extents.at(2) < nprocs) {
      topo_test_cases.at(topo_and_axes_type{0, 0, 1, 0, 2}) =
          ref_vec_type{0, 1, 0};
      topo_test_cases.at(topo_and_axes_type{0, 0, 2, 1, 0}) =
          ref_vec_type{0, 1, 0};
      topo_test_cases.at(topo_and_axes_type{0, 1, 1, 0, 2}) =
          ref_vec_type{0, 1, 0, 1};
      topo_test_cases.at(topo_and_axes_type{0, 1, 2, 1, 0}) =
          ref_vec_type{0, 1, 0, 1};
      topo_test_cases.at(topo_and_axes_type{1, 0, 0, 1, 2}) =
          ref_vec_type{1, 0, 1, 0};
      topo_test_cases.at(topo_and_axes_type{1, 0, 2, 0, 1}) =
          ref_vec_type{1, 0, 1, 0};
      topo_test_cases.at(topo_and_axes_type{1, 1, 0, 1, 2}) =
          ref_vec_type{1, 0, 1};
      topo_test_cases.at(topo_and_axes_type{1, 1, 2, 0, 1}) =
          ref_vec_type{1, 0, 1};
    } else if (KokkosFFT::Impl::extent_after_transform(gin_extents.at(2),
                                                       is_R2C) < nprocs) {
      topo_test_cases.at(topo_and_axes_type{0, 0, 1, 0, 2}) =
          ref_vec_type{0, 1, 0};
      topo_test_cases.at(topo_and_axes_type{0, 1, 1, 0, 2}) =
          ref_vec_type{0, 1, 0, 1};
      topo_test_cases.at(topo_and_axes_type{1, 0, 0, 1, 2}) =
          ref_vec_type{1, 0, 1, 0};
      topo_test_cases.at(topo_and_axes_type{1, 1, 0, 1, 2}) =
          ref_vec_type{1, 0, 1};
    }

    for (const auto& [topo_and_axes, ref_indices] : topo_test_cases) {
      auto [idx_topo_in, idx_topo_out, axis0, axis1, axis2] = topo_and_axes;
      axes_type axes{axis0, axis1, axis2};
      topo_type topo_in  = to_slab_topology<3>(idx_topo_in, nprocs);
      topo_type topo_out = to_slab_topology<3>(idx_topo_out, nprocs);
      auto ref_topos     = to_vec_topology<3>(ref_indices, nprocs);

      auto gout_extents      = gin_extents;
      gout_extents.at(axis2) = KokkosFFT::Impl::extent_after_transform(
          gout_extents.at(axis2), is_R2C);

      auto in_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
          gin_extents, topo_in);
      auto out_extents = KokkosFFT::Distributed::Impl::divide_by_topology(
          gout_extents, topo_out);

      auto in_size  = KokkosFFT::Impl::total_size(in_extents);
      auto out_size = KokkosFFT::Impl::total_size(out_extents);

      bool transpose_needed = false;
      for (const auto& ax : axes) {
        if (topo_in.at(ax) > 1) {
          transpose_needed = true;
          break;
        }
      }
      bool transpose_possible = true;
      for (const auto& ax : axes) {
        bool transpose_possible_for_ax = false;
        for (std::size_t i = 0; i < 3; ++i) {
          if (i == ax) continue;
          if (gout_extents.at(i) >= nprocs) {
            transpose_possible_for_ax = true;
            break;
          }
        }
        if (!transpose_possible_for_ax) {
          transpose_possible = false;
          break;
        }
      }

      if (in_size == 0 || out_size == 0 ||
          (transpose_needed && !transpose_possible)) {
        EXPECT_THROW(
            {
              [[maybe_unused]] auto topos =
                  KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                      gin_extents, gout_extents, topo_in, topo_out, axes);
            },
            std::runtime_error);
      } else {
        auto [topos, all_axes] =
            KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                gin_extents, gout_extents, topo_in, topo_out, axes);

        // The FFT axes must be identical with decompose_axes
        EXPECT_EQ(all_axes,
                  KokkosFFT::Distributed::Impl::decompose_axes(topos, axes));

        EXPECT_EQ(topos, ref_topos)
            << error_all_topologies(gin_extents, gout_extents, topo_in,
                                    topo_out, axes, topos, ref_topos);
      }
    }
  }
}

/// \brief Tests for get_all_slab_topologies for 3D FFT, which returns the list
/// of all valid slab topologies for a given input topology, output topology,
/// and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (3D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_slab_topologies3D_impl(
    std::size_t nprocs, const std::array<std::size_t, DIM>& gin_extents) {
  using topo_type = std::array<std::size_t, DIM>;
  using axes_type = std::array<std::size_t, 3>;

  std::vector<std::size_t> all_dimensions(DIM);
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);

  for (const auto& idx_topo_in : all_dimensions) {
    for (const auto& idx_topo_out : all_dimensions) {
      for (const auto& axis0 : all_dimensions) {
        for (const auto& axis1 : all_dimensions) {
          for (const auto& axis2 : all_dimensions) {
            if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
            axes_type axes{axis0, axis1, axis2};
            topo_type topo_in  = to_slab_topology<DIM>(idx_topo_in, nprocs);
            topo_type topo_out = to_slab_topology<DIM>(idx_topo_out, nprocs);
            auto gout_extents  = gin_extents;
            gout_extents.at(axis2) = KokkosFFT::Impl::extent_after_transform(
                gout_extents[axis2], is_R2C);

            if (nprocs == 1) {
              EXPECT_THROW(
                  {
                    [[maybe_unused]] auto all_slab_topologies =
                        KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                            gin_extents, gout_extents, topo_in, topo_out, axes);
                  },
                  std::runtime_error);
            } else {
              auto in_extents =
                  KokkosFFT::Distributed::Impl::divide_by_topology(gin_extents,
                                                                   topo_in);
              auto out_extents =
                  KokkosFFT::Distributed::Impl::divide_by_topology(gout_extents,
                                                                   topo_out);

              auto in_size          = KokkosFFT::Impl::total_size(in_extents);
              auto out_size         = KokkosFFT::Impl::total_size(out_extents);
              bool transpose_needed = false;
              for (const auto& ax : axes) {
                if (topo_in.at(ax) > 1) {
                  transpose_needed = true;
                  break;
                }
              }
              bool transpose_possible = true;
              for (const auto& ax : axes) {
                bool transpose_possible_for_ax = false;
                for (std::size_t i = 0; i < DIM; ++i) {
                  if (i == ax) continue;
                  if (gout_extents.at(i) >= nprocs) {
                    transpose_possible_for_ax = true;
                    break;
                  }
                }
                if (!transpose_possible_for_ax) {
                  transpose_possible = false;
                  break;
                }
              }

              if (in_size == 0 || out_size == 0 ||
                  (transpose_needed && !transpose_possible)) {
                EXPECT_THROW(
                    {
                      [[maybe_unused]] auto topos =
                          KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                              gin_extents, gout_extents, topo_in, topo_out,
                              axes);
                    },
                    std::runtime_error);
              } else {
                // This must return length 1-3 vectors
                auto [topos, all_axes] =
                    KokkosFFT::Distributed::Impl::get_all_slab_topologies(
                        gin_extents, gout_extents, topo_in, topo_out, axes);

                // The FFT axes must be identical with decompose_axes
                EXPECT_EQ(
                    all_axes,
                    KokkosFFT::Distributed::Impl::decompose_axes(topos, axes));
                // In 3D, we can have at most 5 topologies
                // (in, out, and 3 transposes)
                std::size_t max_size = std::min(DIM, std::size_t(5));
                EXPECT_GE(topos.size(), 1);
                EXPECT_LE(topos.size(), max_size);
                EXPECT_FALSE(has_consecutive_duplicates(topos));
              }
            }
          }
        }
      }
    }
  }
}

/// \brief Tests for get_all_slab_topologies for 1D FFT, which returns the list
/// of all valid slab topologies for a given input topology, output topology,
/// and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (3D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_slab_topologies1D(std::size_t nprocs) {
  using extents_type = std::array<std::size_t, DIM>;
  extents_type gin_extents{};
  for (std::size_t i = 0; i < DIM; ++i) {
    gin_extents.fill(2);
    gin_extents.at(i) = 7;
    test_get_all_slab_topologies1D_impl<is_R2C, DIM>(nprocs, gin_extents);
  }
  gin_extents.fill(8);
  test_get_all_slab_topologies1D_impl<is_R2C, DIM>(nprocs, gin_extents);
}

/// \brief Tests for get_all_slab_topologies for 2D FFT, which returns the list
/// of all valid slab topologies for a given input topology, output topology,
/// and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (3D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_slab_topologies2D(std::size_t nprocs) {
  using extents_type = std::array<std::size_t, DIM>;
  extents_type gin_extents{};
  for (std::size_t i = 0; i < DIM; ++i) {
    gin_extents.fill(2);
    gin_extents.at(i) = 7;
    test_get_all_slab_topologies2D_impl<is_R2C, DIM>(nprocs, gin_extents);
  }
  gin_extents.fill(8);
  test_get_all_slab_topologies2D_impl<is_R2C, DIM>(nprocs, gin_extents);
}

/// \brief Tests for get_all_slab_topologies for 3D FFT, which returns the list
/// of all valid slab topologies for a given input topology, output topology,
/// and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (4D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_slab_topologies3D(std::size_t nprocs) {
  using extents_type = std::array<std::size_t, DIM>;
  extents_type gin_extents{};
  for (std::size_t i = 0; i < DIM; ++i) {
    gin_extents.fill(2);
    gin_extents.at(i) = 7;
    test_get_all_slab_topologies3D_impl<is_R2C, DIM>(nprocs, gin_extents);
  }
  gin_extents.fill(8);
  test_get_all_slab_topologies3D_impl<is_R2C, DIM>(nprocs, gin_extents);
}

/// \brief Tests for get_all_pencil_topologies for 1D FFT, which returns the
/// list of all valid pencil topologies for a given input topology, output
/// topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (2D-7D).
/// \tparam InLayoutType The layout type of the input View (e.g.,
/// Kokkos::LayoutRight or Kokkos::LayoutLeft).
/// \tparam OutLayoutType The layout type of the output View (e.g.,
/// Kokkos::LayoutRight or Kokkos::LayoutLeft).
/// \param[in] npx The number of processes in the x-dimension.
/// \param[in] npy The number of processes in the y-dimension.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM, typename InLayoutType,
          typename OutLayoutType>
void test_get_all_pencil_topologies1D_impl(
    std::size_t npx, std::size_t npy,
    const std::array<std::size_t, DIM>& gin_extents) {
  using axes_type = std::array<std::size_t, 1>;

  std::vector<std::size_t> all_dimensions(DIM);
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);

  for (const auto& idx_topo_in1 : all_dimensions) {
    for (const auto& idx_topo_in2 : all_dimensions) {
      if (idx_topo_in1 == idx_topo_in2) continue;
      if (std::is_same_v<InLayoutType, Kokkos::LayoutRight>) {
        if (idx_topo_in1 > idx_topo_in2) continue;
      } else {
        if (idx_topo_in1 < idx_topo_in2) continue;
      }
      for (const auto& idx_topo_out1 : all_dimensions) {
        for (const auto& idx_topo_out2 : all_dimensions) {
          if (idx_topo_out1 == idx_topo_out2) continue;
          if (std::is_same_v<OutLayoutType, Kokkos::LayoutRight>) {
            if (idx_topo_out1 > idx_topo_out2) continue;
          } else {
            if (idx_topo_out1 < idx_topo_out2) continue;
          }
          for (const auto& idx_axis : all_dimensions) {
            axes_type axes{idx_axis};
            // LayoutRight: (npx, npy)
            // LayoutLeft: (npy, npx)

            auto topo_in = to_pencil_topology<InLayoutType, DIM>(
                idx_topo_in1, idx_topo_in2, npx, npy);
            auto topo_out = to_pencil_topology<OutLayoutType, DIM>(
                idx_topo_out1, idx_topo_out2, npx, npy);
            auto gout_extents         = gin_extents;
            gout_extents.at(idx_axis) = KokkosFFT::Impl::extent_after_transform(
                gout_extents[idx_axis], is_R2C);

            if (npx == 1 || npy == 1) {
              EXPECT_THROW(
                  {
                    [[maybe_unused]] auto topo_and_axes =
                        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                            gin_extents, gout_extents, topo_in, topo_out, axes);
                  },
                  std::runtime_error);
            } else {
              auto in_extents =
                  KokkosFFT::Distributed::Impl::divide_by_topology(gin_extents,
                                                                   topo_in);
              auto out_extents =
                  KokkosFFT::Distributed::Impl::divide_by_topology(gout_extents,
                                                                   topo_out);

              auto in_size  = KokkosFFT::Impl::total_size(in_extents);
              auto out_size = KokkosFFT::Impl::total_size(out_extents);
              bool transpose_needed =
                  topo_in.at(idx_axis) > 1 ||
                  (topo_in.array() != topo_out.array()) ||
                  (topo_in.array() == topo_out.array() &&
                   !std::is_same_v<InLayoutType, OutLayoutType>);

              bool transpose_possible = false;
              try {
                auto axes_vec = KokkosFFT::Impl::to_vector(axes);
                auto all_states =
                    KokkosFFT::Distributed::Impl::find_minimal_path(
                        gin_extents, gout_extents, topo_in, topo_out,
                        axes_vec);  // At least one minimal path must exist
                transpose_possible = true;
              } catch (...) {
              }

              if (in_size == 0 || out_size == 0 ||
                  (transpose_needed && !transpose_possible)) {
                EXPECT_THROW(
                    {
                      [[maybe_unused]] auto topo_and_axes = KokkosFFT::
                          Distributed::Impl::get_all_pencil_topologies(
                              gin_extents, gout_extents, topo_in, topo_out,
                              axes);
                    },
                    std::runtime_error);
              } else {
                // This must return length 1-3 vectors
                auto [topos, trans_axes, layouts, all_axes] =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        gin_extents, gout_extents, topo_in, topo_out, axes);

                // The FFT axes must be identical with decompose_axes
                EXPECT_EQ(
                    all_axes,
                    KokkosFFT::Distributed::Impl::decompose_axes(topos, axes));

                // In 1D, we can have at most 3 topologies
                // (in, out, and 1 transpose)
                std::size_t max_size = std::min(DIM, std::size_t(3)) + 2;
                EXPECT_GE(topos.size(), 1);
                EXPECT_LE(topos.size(), max_size);
                EXPECT_FALSE(has_consecutive_duplicates(topos));
              }
            }
          }
        }
      }
    }
  }
}

/// \brief Tests for get_all_pencil_topologies for 2D FFT, which returns the
/// list of all valid pencil topologies for a given input topology, output
/// topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (3D-7D).
/// \param[in] nprocs The number of processes in the communicator
/// ranges from 1 to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM, typename InLayoutType,
          typename OutLayoutType>
void test_get_all_pencil_topologies2D_impl(
    std::size_t npx, std::size_t npy,
    const std::array<std::size_t, DIM>& gin_extents) {
  using axes_type = std::array<std::size_t, 2>;

  std::vector<std::size_t> all_dimensions(DIM);
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);

  for (const auto& idx_topo_in1 : all_dimensions) {
    for (const auto& idx_topo_in2 : all_dimensions) {
      if (idx_topo_in1 == idx_topo_in2) continue;
      if (std::is_same_v<InLayoutType, Kokkos::LayoutRight>) {
        if (idx_topo_in1 > idx_topo_in2) continue;
      } else {
        if (idx_topo_in1 < idx_topo_in2) continue;
      }
      for (const auto& idx_topo_out1 : all_dimensions) {
        for (const auto& idx_topo_out2 : all_dimensions) {
          if (idx_topo_out1 == idx_topo_out2) continue;
          if (std::is_same_v<OutLayoutType, Kokkos::LayoutRight>) {
            if (idx_topo_out1 > idx_topo_out2) continue;
          } else {
            if (idx_topo_out1 < idx_topo_out2) continue;
          }
          for (const auto& axis0 : all_dimensions) {
            for (const auto& axis1 : all_dimensions) {
              if (axis0 == axis1) continue;

              axes_type axes{axis0, axis1};
              // LayoutRight: (npx, npy)
              // LayoutLeft: (npy, npx)

              auto topo_in = to_pencil_topology<InLayoutType, DIM>(
                  idx_topo_in1, idx_topo_in2, npx, npy);
              auto topo_out = to_pencil_topology<OutLayoutType, DIM>(
                  idx_topo_out1, idx_topo_out2, npx, npy);
              auto gout_extents      = gin_extents;
              gout_extents.at(axis1) = KokkosFFT::Impl::extent_after_transform(
                  gout_extents[axis1], is_R2C);

              if (npx == 1 || npy == 1) {
                EXPECT_THROW(
                    {
                      [[maybe_unused]] auto topo_and_axes = KokkosFFT::
                          Distributed::Impl::get_all_pencil_topologies(
                              gin_extents, gout_extents, topo_in, topo_out,
                              axes);
                    },
                    std::runtime_error);
              } else {
                auto in_extents =
                    KokkosFFT::Distributed::Impl::divide_by_topology(
                        gin_extents, topo_in);
                auto out_extents =
                    KokkosFFT::Distributed::Impl::divide_by_topology(
                        gout_extents, topo_out);

                auto in_size  = KokkosFFT::Impl::total_size(in_extents);
                auto out_size = KokkosFFT::Impl::total_size(out_extents);
                bool transpose_needed =
                    (topo_in.at(axis0) > 1 || topo_in.at(axis1) > 1) ||
                    (topo_in.array() != topo_out.array()) ||
                    (topo_in.array() == topo_out.array() &&
                     !std::is_same_v<InLayoutType, OutLayoutType>);

                bool transpose_possible = false;
                try {
                  auto axes_vec = KokkosFFT::Impl::to_vector(axes);
                  auto all_states =
                      KokkosFFT::Distributed::Impl::find_minimal_path(
                          gin_extents, gout_extents, topo_in, topo_out,
                          axes_vec);  // At least one minimal path must exist
                  transpose_possible = true;
                } catch (...) {
                }

                if (in_size == 0 || out_size == 0 ||
                    (transpose_needed && !transpose_possible)) {
                  EXPECT_THROW(
                      {
                        [[maybe_unused]] auto topo_and_axes = KokkosFFT::
                            Distributed::Impl::get_all_pencil_topologies(
                                gin_extents, gout_extents, topo_in, topo_out,
                                axes);
                      },
                      std::runtime_error);
                } else {
                  // This must return length 1-3 vectors
                  auto [topos, trans_axes, layouts, all_axes] =
                      KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                          gin_extents, gout_extents, topo_in, topo_out, axes);

                  // The FFT axes must be identical with decompose_axes
                  EXPECT_EQ(all_axes,
                            KokkosFFT::Distributed::Impl::decompose_axes(topos,
                                                                         axes));

                  // In 1D, we can have at most 3 topologies
                  // (in, out, and 1 transpose)
                  std::size_t max_size = std::min(DIM, std::size_t(3)) + 3;
                  EXPECT_GE(topos.size(), 1);
                  EXPECT_LE(topos.size(), max_size);
                  EXPECT_FALSE(has_consecutive_duplicates(topos));
                }
              }
            }
          }
        }
      }
    }
  }
}

/// \brief Tests for get_all_pencil_topologies for 3D FFT, which returns the
/// list of all valid pencil topologies for a given input topology, output
/// topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (3D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM, typename InLayoutType,
          typename OutLayoutType>
void test_get_all_pencil_topologies3D_impl(
    std::size_t npx, std::size_t npy,
    const std::array<std::size_t, DIM>& gin_extents) {
  using axes_type = std::array<std::size_t, 3>;

  std::vector<std::size_t> all_dimensions(DIM);
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);

  for (const auto& idx_topo_in1 : all_dimensions) {
    for (const auto& idx_topo_in2 : all_dimensions) {
      if (idx_topo_in1 == idx_topo_in2) continue;
      if (std::is_same_v<InLayoutType, Kokkos::LayoutRight>) {
        if (idx_topo_in1 > idx_topo_in2) continue;
      } else {
        if (idx_topo_in1 < idx_topo_in2) continue;
      }
      for (const auto& idx_topo_out1 : all_dimensions) {
        for (const auto& idx_topo_out2 : all_dimensions) {
          if (idx_topo_out1 == idx_topo_out2) continue;
          if (std::is_same_v<OutLayoutType, Kokkos::LayoutRight>) {
            if (idx_topo_out1 > idx_topo_out2) continue;
          } else {
            if (idx_topo_out1 < idx_topo_out2) continue;
          }
          for (const auto& axis0 : all_dimensions) {
            for (const auto& axis1 : all_dimensions) {
              for (const auto& axis2 : all_dimensions) {
                if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2)
                  continue;

                axes_type axes{axis0, axis1, axis2};
                // LayoutRight: (npx, npy)
                // LayoutLeft: (npy, npx)

                auto topo_in = to_pencil_topology<InLayoutType, DIM>(
                    idx_topo_in1, idx_topo_in2, npx, npy);
                auto topo_out = to_pencil_topology<OutLayoutType, DIM>(
                    idx_topo_out1, idx_topo_out2, npx, npy);
                auto gout_extents = gin_extents;
                gout_extents.at(axis2) =
                    KokkosFFT::Impl::extent_after_transform(gout_extents[axis2],
                                                            is_R2C);

                if (npx == 1 || npy == 1) {
                  EXPECT_THROW(
                      {
                        [[maybe_unused]] auto topo_and_axes = KokkosFFT::
                            Distributed::Impl::get_all_pencil_topologies(
                                gin_extents, gout_extents, topo_in, topo_out,
                                axes);
                      },
                      std::runtime_error);
                } else {
                  auto in_extents =
                      KokkosFFT::Distributed::Impl::divide_by_topology(
                          gin_extents, topo_in);
                  auto out_extents =
                      KokkosFFT::Distributed::Impl::divide_by_topology(
                          gout_extents, topo_out);

                  auto in_size  = KokkosFFT::Impl::total_size(in_extents);
                  auto out_size = KokkosFFT::Impl::total_size(out_extents);
                  bool transpose_needed =
                      (topo_in.at(axis0) > 1 || topo_in.at(axis1) > 1 ||
                       topo_in.at(axis2) > 1) ||
                      (topo_in.array() != topo_out.array()) ||
                      (topo_in.array() == topo_out.array() &&
                       !std::is_same_v<InLayoutType, OutLayoutType>);

                  bool transpose_possible = false;
                  try {
                    auto axes_vec = KokkosFFT::Impl::to_vector(axes);
                    auto all_states =
                        KokkosFFT::Distributed::Impl::find_minimal_path(
                            gin_extents, gout_extents, topo_in, topo_out,
                            axes_vec);  // At least one minimal path must exist
                    transpose_possible = true;
                  } catch (...) {
                  }

                  if (in_size == 0 || out_size == 0 ||
                      (transpose_needed && !transpose_possible)) {
                    EXPECT_THROW(
                        {
                          [[maybe_unused]] auto topo_and_axes = KokkosFFT::
                              Distributed::Impl::get_all_pencil_topologies(
                                  gin_extents, gout_extents, topo_in, topo_out,
                                  axes);
                        },
                        std::runtime_error);
                  } else {
                    // This must return length 1-3 vectors
                    auto [topos, trans_axes, layouts, all_axes] =
                        KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                            gin_extents, gout_extents, topo_in, topo_out, axes);

                    // The FFT axes must be identical with decompose_axes
                    EXPECT_EQ(all_axes,
                              KokkosFFT::Distributed::Impl::decompose_axes(
                                  topos, axes));

                    // In 1D, we can have at most 3 topologies
                    // (in, out, and 1 transpose)
                    std::size_t max_size = std::min(DIM, std::size_t(3)) + 5;
                    EXPECT_GE(topos.size(), 1);
                    EXPECT_LE(topos.size(), max_size);
                    EXPECT_FALSE(has_consecutive_duplicates(topos));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/// \brief Tests for get_all_pencil_topologies for 1D FFT, which returns the
/// list of all valid pencil topologies for a given input topology, output
/// topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (2D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_pencil_topologies1D(std::size_t npx, std::size_t npy) {
  using extents_type = std::array<std::size_t, DIM>;
  extents_type gin_extents{};
  gin_extents.fill(7);
  for (std::size_t i = 0; i < DIM; ++i) {
    gin_extents.at(i) = 2;
    test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                          Kokkos::LayoutLeft>(npx, npy,
                                                              gin_extents);
    test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                          Kokkos::LayoutRight>(npx, npy,
                                                               gin_extents);
    test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                          Kokkos::LayoutLeft>(npx, npy,
                                                              gin_extents);
    test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                          Kokkos::LayoutRight>(npx, npy,
                                                               gin_extents);
  }
  gin_extents.fill(8);
  test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                        Kokkos::LayoutLeft>(npx, npy,
                                                            gin_extents);
  test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                        Kokkos::LayoutRight>(npx, npy,
                                                             gin_extents);
  test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                        Kokkos::LayoutLeft>(npx, npy,
                                                            gin_extents);
  test_get_all_pencil_topologies1D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                        Kokkos::LayoutRight>(npx, npy,
                                                             gin_extents);
}

/// \brief Tests for get_all_pencil_topologies for 1D FFT, which returns the
/// list of all valid pencil topologies for a given input topology, output
/// topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (2D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_pencil_topologies2D(std::size_t npx, std::size_t npy) {
  using extents_type = std::array<std::size_t, DIM>;
  extents_type gin_extents{};
  gin_extents.fill(7);
  for (std::size_t i = 0; i < DIM; ++i) {
    gin_extents.at(i) = 2;
    test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                          Kokkos::LayoutLeft>(npx, npy,
                                                              gin_extents);
    test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                          Kokkos::LayoutRight>(npx, npy,
                                                               gin_extents);
    test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                          Kokkos::LayoutLeft>(npx, npy,
                                                              gin_extents);
    test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                          Kokkos::LayoutRight>(npx, npy,
                                                               gin_extents);
  }
  gin_extents.fill(8);
  test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                        Kokkos::LayoutLeft>(npx, npy,
                                                            gin_extents);
  test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                        Kokkos::LayoutRight>(npx, npy,
                                                             gin_extents);
  test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                        Kokkos::LayoutLeft>(npx, npy,
                                                            gin_extents);
  test_get_all_pencil_topologies2D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                        Kokkos::LayoutRight>(npx, npy,
                                                             gin_extents);
}

/// \brief Tests for get_all_pencil_topologies for 3D FFT, which returns the
/// list of all valid pencil topologies for a given input topology, output
/// topology, and transform axes.
/// \tparam is_R2C Whether the transform is real-to-complex (true) or
/// complex-to-complex (false).
/// \tparam DIM The dimensionality of the View (3D-7D).
/// \param[in] nprocs The number of processes in the communicator ranges from 1
/// to 6.
/// \param[in] gin_extents The global input extents for the transform.
template <bool is_R2C, std::size_t DIM>
void test_get_all_pencil_topologies3D(std::size_t npx, std::size_t npy) {
  using extents_type = std::array<std::size_t, DIM>;
  extents_type gin_extents{};
  gin_extents.fill(7);
  for (std::size_t i = 0; i < DIM; ++i) {
    gin_extents.at(i) = 2;
    test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                          Kokkos::LayoutLeft>(npx, npy,
                                                              gin_extents);
    test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                          Kokkos::LayoutRight>(npx, npy,
                                                               gin_extents);
    test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                          Kokkos::LayoutLeft>(npx, npy,
                                                              gin_extents);
    test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                          Kokkos::LayoutRight>(npx, npy,
                                                               gin_extents);
  }
  gin_extents.fill(8);
  test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                        Kokkos::LayoutLeft>(npx, npy,
                                                            gin_extents);
  test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutLeft,
                                        Kokkos::LayoutRight>(npx, npy,
                                                             gin_extents);
  test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                        Kokkos::LayoutLeft>(npx, npy,
                                                            gin_extents);
  test_get_all_pencil_topologies3D_impl<is_R2C, DIM, Kokkos::LayoutRight,
                                        Kokkos::LayoutRight>(npx, npy,
                                                             gin_extents);
}

void test_get_all_pencil_topologies1D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using extents_type  = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 1>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;

  std::size_t np0 = 4;

  topo_r_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{nprocs, np0, 1};
  topo_l_type topo3{np0, nprocs, 1}, topo4{np0, 1, nprocs},
      topo5{1, np0, nprocs};
  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array();

  axes_type axes0{0}, axes1{1}, axes2{2};
  std::vector<axes_type> all_axes{axes0, axes1, axes2};
  extents_type extents{8, 8, 8};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in : vec_topo_r_type{topo0, topo1, topo2}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in : vec_topo_l_type{topo3, topo4}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes0, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo0}},
        {topo0, topo0, axes1, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes2, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo1, axes0, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes1, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes2, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo2, axes0, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes1, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes2, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes0, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes1, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes2, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo1, axes0, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes1, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo1}},
        {topo1, topo1, axes2, vec_axis_type{0, 1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo2, axes0, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes1, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo1, topo2, axes2, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo2, topo0, axes0, vec_axis_type{0, 1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes1, vec_axis_type{0, 1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes2, vec_axis_type{0, 1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo1, axes0, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes1, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo2, ref_topo1}},
        {topo2, topo1, axes2, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo2, ref_topo1}},
        {topo2, topo2, axes0, vec_axis_type{0, 0, 0}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes1, vec_axis_type{0, 1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes2, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo3, axes0, vec_axis_type{0, 1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo3}},
        {topo0, topo3, axes1, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes2, vec_axis_type{0, 1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo3}},
        {topo0, topo4, axes0, vec_axis_type{0, 1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes1, vec_axis_type{0, 1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes2, vec_axis_type{0, 1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}}};
    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo3, topo0, axes0, vec_axis_type{0, 1}, vec_layout_type{0, 1},
         vec_topo_type{ref_topo3, ref_topo0}},
        {topo3, topo0, axes1, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo3, topo0, axes2, vec_axis_type{0, 1}, vec_layout_type{0, 1},
         vec_topo_type{ref_topo3, ref_topo0}},
        {topo4, topo0, axes0, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes1, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes2, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo3, topo3, axes0, vec_axis_type{0, 1, 1}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes1, vec_axis_type{0, 0, 0}, vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo3, axes2, vec_axis_type{0}, vec_layout_type{0},
         vec_topo_type{ref_topo3}}};
    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies2D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using extents_type  = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 2>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  std::size_t np0 = 4;
  extents_type extents{8, 8, 8};

  topo_r_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{nprocs, np0, 1};
  topo_l_type topo3{np0, nprocs, 1}, topo4{np0, 1, nprocs},
      topo5{1, np0, nprocs};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array();

  axes_type axes01{0, 1}, axes02{0, 2}, axes10{1, 0}, axes12{1, 2},
      axes20{2, 0}, axes21{2, 1};
  std::vector<axes_type> all_axes{axes01, axes02, axes10,
                                  axes12, axes20, axes21};

  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in : vec_topo_r_type{topo0, topo1, topo2}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in : vec_topo_l_type{topo3, topo4, topo5}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes01, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes02, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes10, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes12, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo0, topo0, axes20, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes21, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo1, axes01, vec_axis_type{0, 0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1}},
        {topo0, topo1, axes02, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes12, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo1, axes20, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo1, axes21, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo2, axes01, vec_axis_type{0, 0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes02, vec_axis_type{0, 0, 1, 0, 0},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo0, topo2, axes10, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes12, vec_axis_type{0, 0, 1, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo2}},
        {topo0, topo2, axes20, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes21, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes02, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo0, axes10, vec_axis_type{0, 0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0}},
        {topo1, topo0, axes12, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes20, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo0, axes21, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo1, axes01, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes02, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo1, topo1, axes10, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes12, vec_axis_type{0, 1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo1, axes20, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo1, axes21, vec_axis_type{0, 1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo2, axes01, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes02, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo1, topo2, axes10, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes12, vec_axis_type{0, 1, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo2}},
        {topo1, topo2, axes20, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes21, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo2, topo0, axes01, vec_axis_type{0, 1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes02, vec_axis_type{0, 1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes10, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes12, vec_axis_type{0, 1, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes20, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes21, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo1, axes01, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1}},
        {topo2, topo1, axes02, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes10, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes12, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo2, ref_topo1}},
        {topo2, topo1, axes20, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes21, vec_axis_type{0, 1, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo1}},
        {topo2, topo2, axes01, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2}},
        {topo2, topo2, axes02, vec_axis_type{0, 0, 0}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes10, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes12, vec_axis_type{0, 1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes20, vec_axis_type{0, 0, 0}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes21, vec_axis_type{0, 1, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo3, axes01, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes02, vec_axis_type{0, 1, 1, 1},
         vec_layout_type{1, 0, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo3}},
        {topo0, topo3, axes10, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes12, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3}},
        {topo0, topo3, axes20, vec_axis_type{0, 1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo3}},
        {topo0, topo3, axes21, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo4, axes01, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes02, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4}},
        {topo0, topo4, axes10, vec_axis_type{0, 1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes12, vec_axis_type{0, 1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes20, vec_axis_type{0, 1, 0}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes21, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo4}}};
    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo3, topo0, axes01, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo3, topo0, axes02, vec_axis_type{0, 1}, vec_layout_type{0, 1},
         vec_topo_type{ref_topo3, ref_topo0}},
        {topo3, topo0, axes10, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{0, 1, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo0}},
        {topo3, topo0, axes12, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo3, topo0, axes20, vec_axis_type{0, 1, 1, 1},
         vec_layout_type{0, 1, 0, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo0}},
        {topo3, topo0, axes21, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes01, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes02, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes10, vec_axis_type{0, 0, 1, 0, 0},
         vec_layout_type{0, 0, 1, 1, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0, ref_topo1, ref_topo0}},
        {topo4, topo0, axes12, vec_axis_type{0, 0, 0, 0, 1},
         vec_layout_type{0, 0, 0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo4, ref_topo3, ref_topo0}},
        {topo4, topo0, axes20, vec_axis_type{0, 0, 1, 1, 1},
         vec_layout_type{0, 0, 1, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0, ref_topo3, ref_topo0}},
        {topo4, topo0, axes21, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo4, ref_topo3, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo3, topo3, axes01, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes02, vec_axis_type{0, 1, 1}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes10, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3}},
        {topo3, topo3, axes12, vec_axis_type{0, 0, 0}, vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo3, axes20, vec_axis_type{0, 1, 1}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes21, vec_axis_type{0, 0, 0}, vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo4, axes01, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes02, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes10, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes12, vec_axis_type{0, 0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo3, ref_topo4}},
        {topo3, topo4, axes20, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes21, vec_axis_type{0, 0, 0, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo4}}};
    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies2D_4DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 4>;
  using extents_type  = std::array<std::size_t, 4>;
  using axes_type     = std::array<std::size_t, 2>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;

  std::size_t np0 = 4;
  topo_r_type topo0{1, 1, nprocs, np0}, topo1{1, nprocs, 1, np0},
      topo2{1, nprocs, np0, 1}, topo3{nprocs, 1, 1, np0},
      topo4{nprocs, 1, np0, 1}, topo5{nprocs, np0, 1, 1};

  topo_l_type topo6{1, np0, nprocs, 1}, topo7{1, np0, 1, nprocs},
      topo8{1, 1, np0, nprocs}, topo9{np0, 1, nprocs, 1},
      topo10{np0, nprocs, 1, 1}, topo11{np0, 1, 1, nprocs};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array(),
            ref_topo6 = topo6.array(), ref_topo7 = topo7.array(),
            ref_topo9 = topo9.array(), ref_topo10 = topo10.array(),
            ref_topo11 = topo11.array();

  axes_type axes01{0, 1}, axes02{0, 2}, axes10{1, 0}, axes12{1, 2},
      axes20{2, 0}, axes21{2, 1}, axes13{1, 3}, axes23{2, 3}, axes03{0, 3};

  std::vector<axes_type> all_axes{axes01, axes02, axes10, axes12, axes20,
                                  axes21, axes13, axes23, axes03};
  extents_type extents{8, 8, 8, 8};
  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in :
           vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
        for (const auto& topo_r_out :
             vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out :
             vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in :
           vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
        for (const auto& topo_r_out :
             vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out :
             vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes01, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo0}},
        {topo0, topo0, axes02, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes10, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo0}},
        {topo0, topo0, axes12, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes20, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes21, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes13, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo0}},
        {topo0, topo0, axes23, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo5, ref_topo6, ref_topo0}},
        {topo0, topo0, axes03, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo0}},
        {topo0, topo1, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes02, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes12, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo1}},
        {topo0, topo1, axes20, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes21, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes13, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo10, ref_topo1}},
        {topo0, topo1, axes23, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo10, ref_topo1}},
        {topo0, topo1, axes03, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo10, ref_topo1}},
        {topo0, topo2, axes01, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes02, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes10, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes12, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo1, ref_topo2}},
        {topo0, topo2, axes20, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes21, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes13, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo2}},
        {topo0, topo2, axes23, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo10, ref_topo2}},
        {topo0, topo2, axes03, vec_axis_type{0, 0, 1}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes01, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes02, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes10, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes12, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes20, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes21, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo0}},
        {topo1, topo0, axes13, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo9, ref_topo0}},
        {topo1, topo0, axes23, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo9, ref_topo0}},
        {topo1, topo0, axes03, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo9, ref_topo0}},
        {topo1, topo1, axes01, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes02, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo1}},
        {topo1, topo1, axes10, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes12, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes20, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo1}},
        {topo1, topo1, axes21, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes13, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo5, ref_topo3, ref_topo1}},
        {topo1, topo1, axes23, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo1}},
        {topo1, topo1, axes03, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo1}},
        {topo1, topo2, axes01, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes02, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo1, topo2, axes10, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes12, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes20, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo1, topo2, axes21, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes13, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo4, ref_topo2}},
        {topo1, topo2, axes23, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo2}},
        {topo1, topo2, axes03, vec_axis_type{0, 1}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo2}},
        {topo2, topo2, axes01, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2}},
        {topo2, topo2, axes02, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes10, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2}},
        {topo2, topo2, axes12, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2}},
        {topo2, topo2, axes20, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes21, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes13, vec_axis_type{0, 0, 0}, vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2}},
        {topo2, topo2, axes23, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes03, vec_axis_type{0}, vec_layout_type{1},
         vec_topo_type{ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo6, axes01, vec_axis_type{0, 1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo6}},
        {topo0, topo6, axes02, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes10, vec_axis_type{0, 1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo6}},
        {topo0, topo6, axes12, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes20, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes21, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes13, vec_axis_type{0, 1, 1}, vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo9, ref_topo6}},
        {topo0, topo6, axes23, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo5, ref_topo6}},
        {topo0, topo6, axes03, vec_axis_type{0, 1}, vec_layout_type{1, 0},
         vec_topo_type{ref_topo0, ref_topo6}}};
    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo7, topo0, axes01, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes02, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes10, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes12, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes20, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes21, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{0, 1, 1, 1},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo3, ref_topo0}},
        {topo7, topo0, axes13, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes23, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 1, 0, 1},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo6, ref_topo0}},
        {topo7, topo0, axes03, vec_axis_type{0, 0, 1}, vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo7, topo6, axes01, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo9, ref_topo6}},
        {topo7, topo6, axes02, vec_axis_type{0, 0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo7, ref_topo6}},
        {topo7, topo6, axes10, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo9, ref_topo6}},
        {topo7, topo6, axes12, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo9, ref_topo6}},
        {topo7, topo6, axes20, vec_axis_type{0, 0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo7, ref_topo6}},
        {topo7, topo6, axes21, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo11, ref_topo9, ref_topo6}},
        {topo7, topo6, axes13, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo9, ref_topo6}},
        {topo7, topo6, axes23, vec_axis_type{0, 0, 0}, vec_layout_type{0, 1, 0},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo6}},
        {topo7, topo6, axes03, vec_axis_type{0, 0}, vec_layout_type{0, 0},
         vec_topo_type{ref_topo7, ref_topo6}}};
    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies3D_3DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 3>;
  using extents_type  = std::array<std::size_t, 3>;
  using axes_type     = std::array<std::size_t, 3>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  std::size_t np0 = 4;

  topo_r_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{nprocs, np0, 1};
  topo_l_type topo3{np0, nprocs, 1}, topo4{np0, 1, nprocs},
      topo5{1, np0, nprocs};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array();

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0};

  std::vector<axes_type> all_axes{axes012, axes021, axes102,
                                  axes120, axes201, axes210};
  extents_type extents{8, 8, 8};
  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in : vec_topo_r_type{topo0, topo1, topo2}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
      for (const auto& topo_l_in : vec_topo_l_type{topo3, topo4, topo5}) {
        for (const auto& topo_r_out : vec_topo_r_type{topo0, topo1, topo2}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out : vec_topo_l_type{topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes012, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo0, topo0, axes021, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes102, vec_axis_type{0, 1, 1, 0, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo0}},
        {topo0, topo0, axes120, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo0, topo0, axes201, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes210, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo1, axes012, vec_axis_type{0, 0, 1, 1, 0, 0},
         vec_layout_type{1, 1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0,
                       ref_topo1}},
        {topo0, topo1, axes021, vec_axis_type{0, 0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0,
                       ref_topo1}},
        {topo0, topo1, axes102, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes120, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo1, axes201, vec_axis_type{0, 0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo2,
                       ref_topo1}},
        {topo0, topo1, axes210, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo0, topo2, axes012, vec_axis_type{0, 1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo2}},
        {topo0, topo2, axes021, vec_axis_type{0, 0, 1, 0, 0},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo0, topo2, axes102, vec_axis_type{0, 1, 1, 0, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes120, vec_axis_type{0, 0, 1, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo2}},
        {topo0, topo2, axes201, vec_axis_type{0, 0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes210, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo0, axes012, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0}},
        {topo1, topo0, axes021, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo0, axes102, vec_axis_type{0, 0, 1, 1, 0, 0},
         vec_layout_type{1, 1, 0, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1,
                       ref_topo0}},
        {topo1, topo0, axes120, vec_axis_type{0, 0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1,
                       ref_topo0}},
        {topo1, topo0, axes201, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo1, topo0, axes210, vec_axis_type{0, 0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo0, ref_topo3,
                       ref_topo0}},
        {topo1, topo1, axes012, vec_axis_type{0, 1, 1, 0, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo1, ref_topo0, ref_topo1}},
        {topo1, topo1, axes021, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo1, topo1, axes102, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 0, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo1, topo1, axes120, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo1, axes201, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo1, axes210, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1}},
        {topo1, topo2, axes012, vec_axis_type{0, 0, 1, 0, 1, 0},
         vec_layout_type{1, 1, 0, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo4, ref_topo5,
                       ref_topo2}},
        {topo1, topo2, axes021, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo1, topo2, axes102, vec_axis_type{0, 0, 1, 1, 0, 1},
         vec_layout_type{1, 1, 0, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo3, ref_topo0, ref_topo1,
                       ref_topo2}},
        {topo1, topo2, axes120, vec_axis_type{0, 0, 0, 1, 1, 1},
         vec_layout_type{1, 1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2, ref_topo1,
                       ref_topo2}},
        {topo1, topo2, axes201, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo1, topo2, axes210, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo1, ref_topo2}},
        {topo2, topo0, axes012, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes021, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo0, axes102, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes120, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes201, vec_axis_type{0, 1, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo3, ref_topo0}},
        {topo2, topo0, axes210, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo1, axes012, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0, ref_topo1}},
        {topo2, topo1, axes021, vec_axis_type{0, 0, 1, 0, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo3, ref_topo0,
                       ref_topo1}},
        {topo2, topo1, axes102, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes120, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1}},
        {topo2, topo1, axes201, vec_axis_type{0, 0, 1, 1, 0, 1},
         vec_layout_type{1, 0, 0, 0, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2,
                       ref_topo1}},
        {topo2, topo1, axes210, vec_axis_type{0, 0, 0, 1, 1, 1},
         vec_layout_type{1, 0, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo2,
                       ref_topo1}},
        {topo2, topo2, axes012, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2}},
        {topo2, topo2, axes021, vec_axis_type{0, 1, 1, 0, 0},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo2, ref_topo5, ref_topo2}},
        {topo2, topo2, axes102, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes120, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo2}},
        {topo2, topo2, axes201, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo4, ref_topo5, ref_topo2}},
        {topo2, topo2, axes210, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 0, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo5, ref_topo2, ref_topo1, ref_topo2}}};

    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo3, axes012, vec_axis_type{0, 0, 1, 1, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo1, ref_topo0,
                       ref_topo3}},
        {topo0, topo3, axes021, vec_axis_type{0, 0, 0, 1, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo0,
                       ref_topo3}},
        {topo0, topo3, axes102, vec_axis_type{0, 0, 1, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4,
                       ref_topo3}},
        {topo0, topo3, axes120, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo3}},
        {topo0, topo3, axes201, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo3, axes210, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3}},
        {topo0, topo4, axes012, vec_axis_type{0, 1, 0, 1, 1},
         vec_layout_type{1, 0, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo0, topo4, axes021, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4}},
        {topo0, topo4, axes102, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2, ref_topo5, ref_topo4}},
        {topo0, topo4, axes120, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes201, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo4}},
        {topo0, topo4, axes210, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo0, ref_topo3, ref_topo4}}};

    for (const auto& [topo_r_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo3, topo1, axes012, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo1}},
        {topo3, topo1, axes021, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo1}},
        {topo3, topo1, axes102, vec_axis_type{0, 1, 0},
         vec_layout_type{0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1}},
        {topo3, topo1, axes120, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo2, ref_topo1}},
        {topo3, topo1, axes201, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{0, 0, 0, 1, 1},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo2, ref_topo1}},
        {topo3, topo1, axes210, vec_axis_type{0, 1, 0, 1, 1},
         vec_layout_type{0, 1, 1, 1, 1},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo1, ref_topo2, ref_topo1}}};

    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo3, topo3, axes012, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes021, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes102, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3}},
        {topo3, topo3, axes120, vec_axis_type{0, 1, 1, 0, 0},
         vec_layout_type{0, 1, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4, ref_topo3}},
        {topo3, topo3, axes201, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3}},
        {topo3, topo3, axes210, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3}},
        {topo3, topo4, axes012, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes021, vec_axis_type{0, 0, 0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo4, ref_topo5,
                       ref_topo4}},
        {topo3, topo4, axes102, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4}},
        {topo3, topo4, axes120, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo0, ref_topo3, ref_topo4}},
        {topo3, topo4, axes201, vec_axis_type{0, 0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo3, ref_topo0, ref_topo3,
                       ref_topo4}},
        {topo3, topo4, axes210, vec_axis_type{0, 0, 1, 1, 0, 0},
         vec_layout_type{0, 0, 0, 0, 0, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo5, ref_topo4, ref_topo3,
                       ref_topo4}}};

    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

void test_get_all_pencil_topologies3D_4DView(std::size_t nprocs) {
  using topo_type     = std::array<std::size_t, 4>;
  using extents_type  = std::array<std::size_t, 4>;
  using axes_type     = std::array<std::size_t, 3>;
  using vec_topo_type = std::vector<topo_type>;
  using topo_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutRight>;
  using topo_l_type =
      KokkosFFT::Distributed::Topology<std::size_t, 4, Kokkos::LayoutLeft>;
  using vec_topo_r_type = std::vector<topo_r_type>;
  using vec_topo_l_type = std::vector<topo_l_type>;
  using vec_axis_type   = std::vector<std::size_t>;
  using vec_layout_type = std::vector<std::size_t>;
  using topo_rr_and_ref_type =
      std::tuple<topo_r_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_rl_and_ref_type =
      std::tuple<topo_r_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_lr_and_ref_type =
      std::tuple<topo_l_type, topo_r_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;
  using topo_ll_and_ref_type =
      std::tuple<topo_l_type, topo_l_type, axes_type, vec_axis_type,
                 vec_layout_type, vec_topo_type>;

  std::size_t np0 = 4;
  topo_r_type topo0{1, 1, nprocs, np0}, topo1{1, nprocs, 1, np0},
      topo2{1, nprocs, np0, 1}, topo3{nprocs, 1, 1, np0},
      topo4{nprocs, 1, np0, 1}, topo5{nprocs, np0, 1, 1};

  topo_l_type topo6{1, np0, nprocs, 1}, topo7{1, np0, 1, nprocs},
      topo8{1, 1, np0, nprocs}, topo9{np0, 1, nprocs, 1},
      topo10{np0, nprocs, 1, 1}, topo11{np0, 1, 1, nprocs};

  topo_type ref_topo0 = topo0.array(), ref_topo1 = topo1.array(),
            ref_topo2 = topo2.array(), ref_topo3 = topo3.array(),
            ref_topo4 = topo4.array(), ref_topo5 = topo5.array(),
            ref_topo6 = topo6.array(), ref_topo7 = topo7.array(),
            ref_topo8 = topo8.array(), ref_topo9 = topo9.array(),
            ref_topo10 = topo10.array(), ref_topo11 = topo11.array();

  axes_type axes012{0, 1, 2}, axes021{0, 2, 1}, axes102{1, 0, 2},
      axes120{1, 2, 0}, axes201{2, 0, 1}, axes210{2, 1, 0}, axes123{1, 2, 3},
      axes132{1, 3, 2};

  std::vector<axes_type> all_axes{axes012, axes021, axes102, axes120,
                                  axes201, axes210, axes123, axes132};
  extents_type extents{8, 8, 8, 8};
  if (nprocs == 1) {
    for (const auto& axes : all_axes) {
      // Failure tests because only two elements differ (slabs)
      for (const auto& topo_r_in :
           vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
        for (const auto& topo_r_out :
             vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out :
             vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_r_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }

      for (const auto& topo_l_in :
           vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
        for (const auto& topo_r_out :
             vec_topo_r_type{topo0, topo1, topo2, topo3, topo4, topo5}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_r_out, axes);
              },
              std::runtime_error);
        }
        for (const auto& topo_l_out :
             vec_topo_l_type{topo6, topo7, topo8, topo9, topo10}) {
          EXPECT_THROW(
              {
                [[maybe_unused]] auto topologies_and_axes =
                    KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
                        extents, extents, topo_l_in, topo_l_out, axes);
              },
              std::runtime_error);
        }
      }
    }
  } else {
    std::vector<topo_rr_and_ref_type> topo_rr_test_cases = {
        {topo0, topo0, axes012, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes021, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes102, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes120, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes201, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes210, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0}},
        {topo0, topo0, axes123, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo5, ref_topo6, ref_topo0}},
        {topo0, topo0, axes132, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo9, ref_topo0}},
        {topo0, topo1, axes012, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo1}},
        {topo0, topo1, axes021, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes102, vec_axis_type{0, 0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1}},
        {topo0, topo1, axes120, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo1}},
        {topo0, topo1, axes201, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes210, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo0, ref_topo1}},
        {topo0, topo1, axes123, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo5, ref_topo3, ref_topo1}},
        {topo0, topo1, axes132, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo5, ref_topo3, ref_topo1}},
        {topo0, topo2, axes012, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo1, ref_topo2}},
        {topo0, topo2, axes021, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes102, vec_axis_type{0, 0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes120, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo1, ref_topo2}},
        {topo0, topo2, axes201, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes210, vec_axis_type{0, 0, 1},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo2}},
        {topo0, topo2, axes123, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo5, ref_topo4, ref_topo2}},
        {topo0, topo2, axes132, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo4, ref_topo2}},
        {topo1, topo0, axes012, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes021, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo0}},
        {topo1, topo0, axes102, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes120, vec_axis_type{0, 0}, vec_layout_type{1, 1},
         vec_topo_type{ref_topo1, ref_topo0}},
        {topo1, topo0, axes201, vec_axis_type{0, 0, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo0}},
        {topo1, topo0, axes210, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo0}},
        {topo1, topo0, axes123, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 1},
         vec_topo_type{ref_topo1, ref_topo10, ref_topo9, ref_topo0}},
        {topo1, topo0, axes132, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{1, 1, 0, 1},
         vec_topo_type{ref_topo1, ref_topo0, ref_topo9, ref_topo0}},
        {topo1, topo1, axes012, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes021, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes102, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes120, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes201, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes210, vec_axis_type{0, 0, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1}},
        {topo1, topo1, axes123, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo5, ref_topo3, ref_topo1}},
        {topo1, topo1, axes132, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo5, ref_topo3, ref_topo1}},
        {topo1, topo2, axes012, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes021, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes102, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes120, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes201, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes210, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo1, ref_topo2}},
        {topo1, topo2, axes123, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo5, ref_topo4, ref_topo2}},
        {topo1, topo2, axes132, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo1, ref_topo3, ref_topo4, ref_topo2}},
        {topo2, topo0, axes012, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes021, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo0, axes102, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes120, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes201, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes210, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo3, ref_topo0}},
        {topo2, topo0, axes123, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo0}},
        {topo2, topo0, axes132, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo10, ref_topo9, ref_topo0}},
        {topo2, topo1, axes012, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo3, ref_topo1}},
        {topo2, topo1, axes021, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo1}},
        {topo2, topo1, axes102, vec_axis_type{0, 1, 0, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo1, ref_topo3, ref_topo1}},
        {topo2, topo1, axes120, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo3, ref_topo1}},
        {topo2, topo1, axes201, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo1}},
        {topo2, topo1, axes210, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo1}},
        {topo2, topo1, axes123, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo3, ref_topo1}},
        {topo2, topo1, axes132, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo3, ref_topo1}},
        {topo2, topo2, axes012, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2}},
        {topo2, topo2, axes021, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes102, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 0, 0, 0, 1},
         vec_topo_type{ref_topo2, ref_topo8, ref_topo11, ref_topo8, ref_topo2}},
        {topo2, topo2, axes120, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2}},
        {topo2, topo2, axes201, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes210, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo2, ref_topo10, ref_topo2}},
        {topo2, topo2, axes123, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2}},
        {topo2, topo2, axes132, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{1, 1, 1, 1, 1},
         vec_topo_type{ref_topo2, ref_topo4, ref_topo5, ref_topo4, ref_topo2}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_rl_and_ref_type> topo_rl_test_cases = {
        {topo0, topo6, axes012, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes021, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes102, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes120, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes201, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes210, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo6}},
        {topo0, topo6, axes123, vec_axis_type{0, 0, 1, 0, 1},
         vec_layout_type{1, 1, 0, 0, 0},
         vec_topo_type{ref_topo0, ref_topo1, ref_topo10, ref_topo9, ref_topo6}},
        {topo0, topo6, axes132, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{1, 1, 1, 0, 0},
         vec_topo_type{ref_topo0, ref_topo3, ref_topo0, ref_topo9, ref_topo6}},
        {topo3, topo8, axes012, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo8}},
        {topo3, topo8, axes021, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo8}},
        {topo3, topo8, axes102, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo8}},
        {topo3, topo8, axes120, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo1, ref_topo2, ref_topo8}},
        {topo3, topo8, axes201, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo1, ref_topo2, ref_topo8}},
        {topo3, topo8, axes210, vec_axis_type{0, 0, 0, 1, 0},
         vec_layout_type{1, 1, 1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo1, ref_topo3, ref_topo4, ref_topo8}},
        {topo3, topo8, axes123, vec_axis_type{0, 1, 0, 1},
         vec_layout_type{1, 1, 0, 0},
         vec_topo_type{ref_topo3, ref_topo5, ref_topo7, ref_topo8}},
        {topo3, topo8, axes132, vec_axis_type{0, 1, 0},
         vec_layout_type{1, 1, 0},
         vec_topo_type{ref_topo3, ref_topo4, ref_topo8}}};
    for (const auto& [topo_r_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_rl_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_r_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_r_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_lr_and_ref_type> topo_lr_test_cases = {
        {topo7, topo0, axes012, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes021, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{0, 1, 1, 1},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo3, ref_topo0}},
        {topo7, topo0, axes102, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes120, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}},
        {topo7, topo0, axes201, vec_axis_type{0, 0, 1, 0, 0},
         vec_layout_type{0, 1, 1, 1, 1},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo3, ref_topo1, ref_topo0}},
        {topo7, topo0, axes210, vec_axis_type{0, 0, 1, 0},
         vec_layout_type{0, 1, 1, 1},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo3, ref_topo0}},
        {topo7, topo0, axes123, vec_axis_type{0, 0, 0, 1},
         vec_layout_type{0, 1, 0, 1},
         vec_topo_type{ref_topo7, ref_topo5, ref_topo6, ref_topo0}},
        {topo7, topo0, axes132, vec_axis_type{0, 0, 1},
         vec_layout_type{0, 0, 1},
         vec_topo_type{ref_topo7, ref_topo6, ref_topo0}}};
    for (const auto& [topo_l_in, topo_r_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_lr_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_r_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_r_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }

    std::vector<topo_ll_and_ref_type> topo_ll_test_cases = {
        {topo9, topo10, axes012, vec_axis_type{0, 0, 0, 1, 1},
         vec_layout_type{0, 0, 0, 1, 0},
         vec_topo_type{ref_topo9, ref_topo11, ref_topo10, ref_topo2,
                       ref_topo10}},
        {topo9, topo10, axes021, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 1, 0},
         vec_topo_type{ref_topo9, ref_topo10, ref_topo2, ref_topo10}},
        {topo9, topo10, axes102, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo9, ref_topo11, ref_topo7, ref_topo11,
                       ref_topo10}},
        {topo9, topo10, axes120, vec_axis_type{0, 0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0, 0},
         vec_topo_type{ref_topo9, ref_topo11, ref_topo7, ref_topo11,
                       ref_topo10}},
        {topo9, topo10, axes201, vec_axis_type{0, 0, 1, 1},
         vec_layout_type{0, 0, 1, 0},
         vec_topo_type{ref_topo9, ref_topo10, ref_topo2, ref_topo10}},
        {topo9, topo10, axes210, vec_axis_type{0, 1, 1, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo9, ref_topo6, ref_topo9, ref_topo10}},
        {topo9, topo10, axes123, vec_axis_type{0, 0, 0},
         vec_layout_type{0, 0, 0},
         vec_topo_type{ref_topo9, ref_topo11, ref_topo10}},
        {topo9, topo10, axes132, vec_axis_type{0, 0, 0, 0},
         vec_layout_type{0, 0, 0, 0},
         vec_topo_type{ref_topo9, ref_topo10, ref_topo9, ref_topo10}}};

    for (const auto& [topo_l_in, topo_l_out, axes, ref_axes, ref_layouts,
                      ref_topos] : topo_ll_test_cases) {
      auto topo_and_axes =
          KokkosFFT::Distributed::Impl::get_all_pencil_topologies(
              extents, extents, topo_l_in, topo_l_out, axes);
      auto ref_topo_and_axes = std::make_tuple(
          ref_topos, ref_axes, ref_layouts,
          KokkosFFT::Distributed::Impl::decompose_axes(ref_topos, axes));
      EXPECT_EQ(topo_and_axes, ref_topo_and_axes)
          << error_all_pencil_topologies(topo_l_in, topo_l_out, axes,
                                         topo_and_axes, ref_topo_and_axes);
    }
  }
}

// A minimal state of a path, for FFTAxesProgress::consumed_along
struct RemainingState {
  std::size_t m_remaining;
};

void test_fft_axes_progress() {
  using axes_type     = std::vector<std::size_t>;
  using topology_type = std::array<std::size_t, 3>;
  axes_type axes{0, 1, 2};
  KokkosFFT::Distributed::Impl::FFTAxesProgress<std::size_t> progress(axes);

  // The axes are consumed from right to left, up to the first distributed one
  EXPECT_EQ(progress.initial(topology_type{1, 1, 4}), 3u);
  EXPECT_EQ(progress.initial(topology_type{1, 4, 1}), 2u);
  EXPECT_EQ(progress.initial(topology_type{4, 1, 1}), 1u);
  EXPECT_EQ(progress.initial(topology_type{1, 1, 1}), 0u);
  EXPECT_EQ(progress.advance(topology_type{4, 1, 1}, 2), 1u);

  // Consumed axes are never reconsidered
  EXPECT_EQ(progress.advance(topology_type{1, 1, 4}, 1), 0u);

  EXPECT_TRUE(progress.none_consumed(3));
  EXPECT_FALSE(progress.none_consumed(2));

  // The first state consumes the axes local on the input topology
  std::vector<RemainingState> path{{2}, {2}, {0}};
  std::vector<axes_type> ref_axes{{2}, {}, {0, 1}};
  EXPECT_EQ(progress.consumed_along(path), ref_axes);
}

/// \brief The axes consumed along the paths of both searches must cover every
/// FFT axis exactly once, and every path must end with all the axes consumed
void test_consumed_axes_along_paths() {
  using topology_type = std::array<std::size_t, 3>;
  using topology_r_type =
      KokkosFFT::Distributed::Topology<std::size_t, 3, Kokkos::LayoutRight>;
  using axes_type = std::vector<std::size_t>;
  const topology_type extents{8, 8, 8};
  const std::vector<axes_type> all_axes = {
      {0},       {1},       {2},       {0, 1},    {0, 2},
      {1, 0},    {1, 2},    {2, 0},    {2, 1},    {0, 1, 2},
      {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

  // Slab topologies: P on each of the three axes
  for (std::size_t in_pos = 0; in_pos < 3; ++in_pos) {
    for (std::size_t out_pos = 0; out_pos < 3; ++out_pos) {
      topology_type in_topology{1, 1, 1}, out_topology{1, 1, 1};
      in_topology.at(in_pos)   = 4;
      out_topology.at(out_pos) = 4;
      for (const auto& axes : all_axes) {
        const std::string case_name =
            "slab: in_pos=" + std::to_string(in_pos) +
            ", out_pos=" + std::to_string(out_pos) +
            KokkosFFT::Impl::container_to_string(", axes=", axes);

        auto states = KokkosFFT::Distributed::Impl::find_minimal_slab_path(
            extents, extents, in_topology, out_topology, axes);
        auto consumed_axes =
            KokkosFFT::Distributed::Impl::FFTAxesProgress<std::size_t>(axes)
                .consumed_along(states);
        ASSERT_EQ(consumed_axes.size(), states.size()) << case_name;

        axes_type consumed;
        for (const auto& state_axes : consumed_axes) {
          consumed.insert(consumed.end(), state_axes.begin(), state_axes.end());
        }
        std::sort(consumed.begin(), consumed.end());
        auto ref_axes = axes;
        std::sort(ref_axes.begin(), ref_axes.end());
        EXPECT_EQ(consumed, ref_axes) << case_name;
        EXPECT_EQ(states.back().m_remaining, 0u) << case_name;
      }
    }
  }

  // Pencil topologies: x-, y- and z-pencils
  const std::vector<topology_r_type> pencils = {topology_r_type{1, 2, 4},
                                                topology_r_type{2, 1, 4},
                                                topology_r_type{2, 4, 1}};
  for (std::size_t in_idx = 0; in_idx < pencils.size(); ++in_idx) {
    for (std::size_t out_idx = 0; out_idx < pencils.size(); ++out_idx) {
      for (const auto& axes : all_axes) {
        const std::string case_name =
            "pencil: in_topology=" + std::to_string(in_idx) +
            ", out_topology=" + std::to_string(out_idx) +
            KokkosFFT::Impl::container_to_string(", axes=", axes);

        auto states = KokkosFFT::Distributed::Impl::find_minimal_path(
            extents, extents, pencils.at(in_idx), pencils.at(out_idx), axes);
        auto consumed_axes =
            KokkosFFT::Distributed::Impl::FFTAxesProgress<std::size_t>(axes)
                .consumed_along(states);
        ASSERT_EQ(consumed_axes.size(), states.size()) << case_name;

        axes_type consumed;
        for (const auto& state_axes : consumed_axes) {
          consumed.insert(consumed.end(), state_axes.begin(), state_axes.end());
        }
        std::sort(consumed.begin(), consumed.end());
        auto ref_axes = axes;
        std::sort(ref_axes.begin(), ref_axes.end());
        EXPECT_EQ(consumed, ref_axes) << case_name;
        EXPECT_EQ(states.back().m_remaining, 0u) << case_name;
      }
    }
  }
}

}  // namespace

TEST_P(TopologyParamTests, GetTopologyType_std_array) {
  int n0 = GetParam();
  test_to_topology_type<true>(n0);
}

TEST_P(TopologyParamTests, GetTopologyType_topology) {
  int n0 = GetParam();
  test_to_topology_type<false>(n0);
}

TEST_P(TopologyParamTests, GetCommonTopologyType_std_array) {
  int n0 = GetParam();
  test_get_common_topology_type<true>(n0);
}

TEST_P(TopologyParamTests, GetCommonTopologyType_topology) {
  int n0 = GetParam();
  test_get_common_topology_type<false>(n0);
}

TEST_P(TopologyParamTests, is_topology_std_array) {
  int n0 = GetParam();
  test_is_topology<true>(n0);
}

TEST_P(TopologyParamTests, is_topology_topology) {
  int n0 = GetParam();
  test_is_topology<false>(n0);
}

TEST_P(TopologyParamTests, are_topologies_std_array) {
  int n0 = GetParam();
  test_are_topologies<true>(n0);
}

TEST_P(TopologyParamTests, are_topologies_topology) {
  int n0 = GetParam();
  test_are_topologies<false>(n0);
}

TEST_P(TopologyParamTests, decompose_axes_slab) {
  int n0 = GetParam();
  test_decompose_axes_slab(n0);
}

TEST_P(TopologyParamTests, decompose_axes_pencil) {
  int n0 = GetParam();
  test_decompose_axes_pencil(n0);
}

TEST_P(TopologyParamTests, slab_in_out_axes_2D) {
  int n0 = GetParam();
  test_slab_in_out_axes_2D(n0);
}

TEST_P(TopologyParamTests, slab_in_out_axes_3D) {
  int n0 = GetParam();
  test_slab_in_out_axes_3D(n0);
}

TEST_P(TopologyParamTests, pencil_in_out_axes_3D) {
  int n0 = GetParam();
  test_pencil_in_out_axes_3D(n0);
}

INSTANTIATE_TEST_SUITE_P(TopologyTests, TopologyParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TEST_P(SlabParamTests, GetAllSlabTopologies1D_2DView_R2C) {
  using extents_type = std::array<std::size_t, 2>;
  int nprocs         = GetParam();
  std::size_t n0_big = 8, n1_big = 9;
  std::size_t n_small = 2;

  test_get_all_slab_topologies1D_2DView<true>(nprocs,
                                              extents_type{n0_big, n1_big});
  test_get_all_slab_topologies1D_2DView<true>(nprocs,
                                              extents_type{n0_big, n_small});
  test_get_all_slab_topologies1D_2DView<true>(nprocs,
                                              extents_type{n_small, n1_big});
  test_get_all_slab_topologies1D_2DView<true>(nprocs,
                                              extents_type{n_small, n_small});
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_2DView_C2C) {
  using extents_type = std::array<std::size_t, 2>;
  int nprocs         = GetParam();
  std::size_t n0_big = 8, n1_big = 9;
  std::size_t n_small = 2;

  test_get_all_slab_topologies1D_2DView<false>(nprocs,
                                               extents_type{n0_big, n1_big});
  test_get_all_slab_topologies1D_2DView<false>(nprocs,
                                               extents_type{n0_big, n_small});
  test_get_all_slab_topologies1D_2DView<false>(nprocs,
                                               extents_type{n_small, n1_big});
  test_get_all_slab_topologies1D_2DView<false>(nprocs,
                                               extents_type{n_small, n_small});
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_3DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<true, 3>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_3DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<false, 3>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_4DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<true, 4>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_4DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<false, 4>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_5DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<true, 5>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_5DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<false, 5>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_6DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<true, 6>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_6DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<false, 6>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_7DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<true, 7>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies1D_7DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies1D<false, 7>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_2DView_R2C) {
  using extents_type = std::array<std::size_t, 2>;
  int nprocs         = GetParam();
  std::size_t n0 = 8, n1 = 9;
  std::size_t n_small = 2;
  test_get_all_slab_topologies2D_2DView<true>(nprocs, extents_type{n0, n1});
  test_get_all_slab_topologies2D_2DView<true>(nprocs,
                                              extents_type{n0, n_small});
  test_get_all_slab_topologies2D_2DView<true>(nprocs,
                                              extents_type{n_small, n1});
  test_get_all_slab_topologies2D_2DView<true>(nprocs,
                                              extents_type{n_small, n_small});
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_2DView_C2C) {
  using extents_type = std::array<std::size_t, 2>;
  int nprocs         = GetParam();
  std::size_t n0 = 8, n1 = 9;
  std::size_t n_small = 2;
  test_get_all_slab_topologies2D_2DView<false>(nprocs, extents_type{n0, n1});
  test_get_all_slab_topologies2D_2DView<false>(nprocs,
                                               extents_type{n0, n_small});
  test_get_all_slab_topologies2D_2DView<false>(nprocs,
                                               extents_type{n_small, n1});
  test_get_all_slab_topologies2D_2DView<false>(nprocs,
                                               extents_type{n_small, n_small});
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_3DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<true, 3>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_3DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<false, 3>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_4DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<true, 4>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_4DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<false, 4>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_5DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<true, 5>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_5DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<false, 5>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_6DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<true, 6>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_6DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<false, 6>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_7DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<true, 7>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies2D_7DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies2D<false, 7>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_3DView_R2C) {
  using extents_type = std::array<std::size_t, 3>;
  int nprocs         = GetParam();
  std::size_t n0 = 8, n1 = 9, n2 = 7;
  std::size_t n_small = 2;

  test_get_all_slab_topologies3D_3DView<true>(nprocs, extents_type{n0, n1, n2});
  test_get_all_slab_topologies3D_3DView<true>(nprocs,
                                              extents_type{n0, n1, n_small});
  test_get_all_slab_topologies3D_3DView<true>(nprocs,
                                              extents_type{n0, n_small, n2});
  test_get_all_slab_topologies3D_3DView<true>(nprocs,
                                              extents_type{n_small, n1, n2});
  test_get_all_slab_topologies3D_3DView<true>(
      nprocs, extents_type{n0, n_small, n_small});
  test_get_all_slab_topologies3D_3DView<true>(
      nprocs, extents_type{n_small, n1, n_small});
  test_get_all_slab_topologies3D_3DView<true>(
      nprocs, extents_type{n_small, n_small, n2});
  test_get_all_slab_topologies3D_3DView<true>(
      nprocs, extents_type{n_small, n_small, n_small});
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_3DView_C2C) {
  using extents_type = std::array<std::size_t, 3>;
  int nprocs         = GetParam();
  std::size_t n0 = 8, n1 = 9, n2 = 7;
  std::size_t n_small = 2;

  test_get_all_slab_topologies3D_3DView<false>(nprocs,
                                               extents_type{n0, n1, n2});
  test_get_all_slab_topologies3D_3DView<false>(nprocs,
                                               extents_type{n0, n1, n_small});
  test_get_all_slab_topologies3D_3DView<false>(nprocs,
                                               extents_type{n0, n_small, n2});
  test_get_all_slab_topologies3D_3DView<false>(nprocs,
                                               extents_type{n_small, n1, n2});
  test_get_all_slab_topologies3D_3DView<false>(
      nprocs, extents_type{n0, n_small, n_small});
  test_get_all_slab_topologies3D_3DView<false>(
      nprocs, extents_type{n_small, n1, n_small});
  test_get_all_slab_topologies3D_3DView<false>(
      nprocs, extents_type{n_small, n_small, n2});
  test_get_all_slab_topologies3D_3DView<false>(
      nprocs, extents_type{n_small, n_small, n_small});
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_4DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<true, 4>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_4DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<false, 4>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_5DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<true, 5>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_5DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<false, 5>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_6DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<true, 6>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_6DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<false, 6>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_7DView_R2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<true, 7>(nprocs);
}

TEST_P(SlabParamTests, GetAllSlabTopologies3D_7DView_C2C) {
  int nprocs = GetParam();
  test_get_all_slab_topologies3D<false, 7>(nprocs);
}

INSTANTIATE_TEST_SUITE_P(SlabTests, SlabParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

// Hardcoded tests for 1D, 2D, and 3D pencil topologies
TEST_P(PencilParamTests, GetAllPencilTopologies1D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies1D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies2D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_4DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies2D_4DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_3DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies3D_3DView(n0);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_4DView) {
  int n0 = GetParam();
  test_get_all_pencil_topologies3D_4DView(n0);
}

// General tests
TEST_P(PencilParamTests, GetAllPencilTopologies1D_3DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies1D<true, 3>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies1D_3DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies1D<false, 3>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies1D_4DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies1D<true, 4>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies1D_4DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies1D<false, 4>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies1D_5DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies1D<true, 5>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies1D_5DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies1D<false, 5>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_3DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies2D<true, 3>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_3DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies2D<false, 3>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_4DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies2D<true, 4>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_4DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies2D<false, 4>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_5DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies2D<true, 5>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies2D_5DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies2D<false, 5>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_3DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies3D<true, 3>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_3DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies3D<false, 3>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_4DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies3D<true, 4>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_4DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies3D<false, 4>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_5DView_R2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies3D<true, 5>(n0, nprocs);
}

TEST_P(PencilParamTests, GetAllPencilTopologies3D_5DView_C2C) {
  int nprocs   = GetParam();
  const int n0 = 4;
  test_get_all_pencil_topologies3D<false, 5>(n0, nprocs);
}

INSTANTIATE_TEST_SUITE_P(PencilTests, PencilParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));

TEST(TestTopologies, FFTAxesProgress) { test_fft_axes_progress(); }

TEST(TestTopologies, ConsumedAxesAlongPaths) {
  test_consumed_axes_along_paths();
}
