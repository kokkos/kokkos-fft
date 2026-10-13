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
using execution_space = Kokkos::DefaultExecutionSpace;
class TopologyParamTests : public ::testing::TestWithParam<int> {};
class SlabParamTests : public ::testing::TestWithParam<int> {};

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
