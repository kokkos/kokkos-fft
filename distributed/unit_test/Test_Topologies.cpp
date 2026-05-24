// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
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
  msg += " and " +
         KokkosFFT::Impl::container_to_string("Input topologies: ", topo2);
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

/// \brief Generate error message for mid_topology test failures.
/// \tparam TopologyType The type of the topology input.
/// \param[in] topo1 The first input topology that caused the failure.
/// \param[in] topo2 The second input topology that caused the failure.
/// \param[in] expected The expected topology
/// \return Error message including the input topologies and the expected
/// topology
template <typename TopologyType>
std::string error_mid_topology(const TopologyType& topo1,
                               const TopologyType& topo2,
                               const TopologyType& expected) {
  auto actual = KokkosFFT::Distributed::Impl::propose_mid_array(topo1, topo2);
  std::string msg = "Input topologies: ";
  msg += KokkosFFT::Impl::container_to_string("topology1: ", topo1);
  msg += " and ";
  msg += KokkosFFT::Impl::container_to_string("topology2: ", topo2);
  msg += ", should have a mid topology: (";
  msg += std::to_string(expected.at(0));
  for (std::size_t i = 1; i < expected.size(); ++i) {
    msg += ", ";
    msg += std::to_string(expected.at(i));
  }
  msg += "), but got (";
  msg += std::to_string(actual.at(0));
  for (std::size_t i = 1; i < actual.size(); ++i) {
    msg += ", ";
    msg += std::to_string(actual.at(i));
  }
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

/// \brief Generate error message for compute_trans_axis test failures.
/// \tparam iType The index type
/// \tparam DIM The dimension
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \param[in] first_non_one The first non-one element in the input or output
/// \param[in] expected The expected transformation axis (0 or 1)
/// \return Error message including the input topologies and the expected in/out
/// axes.
template <typename iType, std::size_t DIM>
std::string error_trans_axis(const std::array<iType, DIM>& in_topology,
                             const std::array<iType, DIM>& out_topology,
                             iType first_non_one, iType expected) {
  auto actual = KokkosFFT::Distributed::Impl::compute_trans_axis(
      in_topology, out_topology, first_non_one);
  std::string msg = "Input topologies: ";
  msg += KokkosFFT::Impl::container_to_string("in_topology: ", in_topology);
  msg += " and ";
  msg += KokkosFFT::Impl::container_to_string("out_topology: ", out_topology);
  msg += ", should have trans_axis: ";
  msg += std::to_string(expected);
  msg += ", but got: ";
  msg += std::to_string(actual);
  msg += ".";
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

void test_compute_trans_axis(std::size_t nprocs) {
  using topo3D_type         = std::array<std::size_t, 3>;
  using topo4D_type         = std::array<std::size_t, 4>;
  using topo3D_and_ref_type = std::tuple<topo3D_type, topo3D_type, std::size_t>;
  using topo4D_and_ref_type = std::tuple<topo4D_type, topo4D_type, std::size_t>;

  std::size_t np0 = 4;

  // 3D topologies
  topo3D_type topo0{1, nprocs, np0}, topo1{nprocs, 1, np0},
      topo2{np0, nprocs, 1};

  // 4D topologies
  topo4D_type topo3{1, 1, np0, nprocs}, topo4{1, np0, 1, nprocs};

  if (nprocs == 1 || nprocs == np0) {
    // Failure tests because these are not pencils for nprocs == 1 or they
    // include identical non-one elements for nprocs == np0
    std::vector<topo3D_and_ref_type> topo3D_failure_test_cases = {
        {topo0, topo1, 0}, {topo0, topo2, 1}, {topo1, topo0, 0},
        {topo1, topo2, 1}, {topo2, topo0, 1}, {topo2, topo1, 1}};

    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_trans_axis] :
         topo3D_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto trans_axis =
                KokkosFFT::Distributed::Impl::compute_trans_axis(
                    topo_in, topo_out, nprocs);
          },
          std::runtime_error);
    }

    std::vector<topo4D_and_ref_type> topo4D_failure_test_cases = {
        {topo3, topo4, 0}, {topo4, topo3, 0}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_trans_axis] :
         topo4D_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto trans_axis =
                KokkosFFT::Distributed::Impl::compute_trans_axis(
                    topo_in, topo_out, nprocs);
          },
          std::runtime_error);
    }
  } else {
    // 3D case
    std::vector<topo3D_and_ref_type> topo3D_test_cases = {{topo0, topo1, 0},
                                                          {topo0, topo2, 1},
                                                          {topo1, topo0, 0},
                                                          {topo2, topo0, 1}};

    for (const auto& [topo_in, topo_out, ref_trans_axis] : topo3D_test_cases) {
      auto trans_axis = KokkosFFT::Distributed::Impl::compute_trans_axis(
          topo_in, topo_out, nprocs);
      EXPECT_EQ(trans_axis, ref_trans_axis)
          << error_trans_axis(topo_in, topo_out, nprocs, ref_trans_axis);
    }

    std::vector<topo3D_and_ref_type> topo3D_failure_test_cases = {
        {topo1, topo2, 0}, {topo2, topo1, 1}};

    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_trans_axis] :
         topo3D_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto trans_axis =
                KokkosFFT::Distributed::Impl::compute_trans_axis(
                    topo_in, topo_out, nprocs);
          },
          std::runtime_error);
    }

    // 4D case
    std::vector<topo4D_and_ref_type> topo4D_test_cases = {{topo3, topo4, 0},
                                                          {topo4, topo3, 0}};

    for (const auto& [topo_in, topo_out, ref_trans_axis] : topo4D_test_cases) {
      auto trans_axis = KokkosFFT::Distributed::Impl::compute_trans_axis(
          topo_in, topo_out, np0);
      EXPECT_EQ(trans_axis, ref_trans_axis)
          << error_trans_axis(topo_in, topo_out, np0, ref_trans_axis);
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

void test_get_mid_array_pencil_3D(std::size_t nprocs) {
  using topo_type         = std::array<std::size_t, 3>;
  using topo_and_ref_type = std::tuple<topo_type, topo_type, topo_type>;
  topo_type topo0{nprocs, 1, 8}, topo1{nprocs, 8, 1}, topo2{8, nprocs, 1},
      topo3{1, 2, nprocs}, topo4{2, nprocs, 1};

  if (nprocs == 1) {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, topo_type{}}, {topo0, topo2, topo_type{}},
        {topo1, topo0, topo_type{}}, {topo1, topo2, topo_type{}},
        {topo2, topo0, topo_type{}}, {topo2, topo1, topo_type{}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_mid] :
         topo_and_ref_vec) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }
  } else {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_failure_test_cases = {
        {topo0, topo1, topo_type{}},
        {topo1, topo0, topo_type{}},
        {topo1, topo2, topo_type{}},
        {topo2, topo1, topo_type{}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_mid] :
         topo_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }
    topo_type ref_mid02{1, nprocs, 8}, ref_mid34{2, 1, nprocs};
    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo2, ref_mid02},
        {topo2, topo0, ref_mid02},
        {topo3, topo4, ref_mid34},
        {topo4, topo3, ref_mid34}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_test_cases) {
      auto mid =
          KokkosFFT::Distributed::Impl::propose_mid_array(topo_in, topo_out);
      EXPECT_EQ(mid, ref_mid) << error_mid_topology(topo_in, topo_out, ref_mid);
    }
  }
}

void test_get_mid_array_pencil_4D(std::size_t nprocs) {
  using topo_type         = std::array<std::size_t, 4>;
  using topo_and_ref_type = std::tuple<topo_type, topo_type, topo_type>;
  topo_type topo0{1, 1, nprocs, 8}, topo1{1, nprocs, 1, 8},
      topo2{1, 8, nprocs, 1}, topo3{1, nprocs, 8, 1}, topo4{1, 8, 1, nprocs},
      topo5{1, 1, 8, nprocs};

  if (nprocs == 1) {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_and_ref_vec = {
        {topo0, topo1, topo_type{}}, {topo0, topo2, topo_type{}},
        {topo1, topo0, topo_type{}}, {topo1, topo2, topo_type{}},
        {topo2, topo0, topo_type{}}, {topo2, topo1, topo_type{}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_mid] :
         topo_and_ref_vec) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }
  } else {
    // Failure tests because only two elements differ
    std::vector<topo_and_ref_type> topo_failure_test_cases = {
        {topo0, topo1, topo_type{}}, {topo0, topo2, topo_type{}},
        {topo0, topo5, topo_type{}}, {topo1, topo3, topo_type{}},
        {topo1, topo4, topo_type{}}, {topo2, topo3, topo_type{}},
        {topo2, topo4, topo_type{}}, {topo3, topo5, topo_type{}},
        {topo4, topo5, topo_type{}}};
    for ([[maybe_unused]] const auto& [topo_in, topo_out, ref_mid] :
         topo_failure_test_cases) {
      EXPECT_THROW(
          {
            [[maybe_unused]] auto mid =
                KokkosFFT::Distributed::Impl::propose_mid_array(topo_in,
                                                                topo_out);
          },
          std::runtime_error);
    }

    std::vector<topo_and_ref_type> topo_test_cases = {
        {topo0, topo3, topo1}, {topo0, topo4, topo2}, {topo1, topo2, topo0},
        {topo1, topo5, topo3}, {topo2, topo5, topo4}, {topo3, topo4, topo5}};
    for (const auto& [topo_in, topo_out, ref_mid] : topo_test_cases) {
      auto mid =
          KokkosFFT::Distributed::Impl::propose_mid_array(topo_in, topo_out);
      EXPECT_EQ(mid, ref_mid) << error_mid_topology(topo_in, topo_out, ref_mid);
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

TEST_P(TopologyParamTests, compute_trans_axis) {
  int n0 = GetParam();
  test_compute_trans_axis(n0);
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

TEST_P(TopologyParamTests, get_mid_array_3D) {
  int n0 = GetParam();
  test_get_mid_array_pencil_3D(n0);
}

TEST_P(TopologyParamTests, get_mid_array_4D) {
  int n0 = GetParam();
  test_get_mid_array_pencil_4D(n0);
}

INSTANTIATE_TEST_SUITE_P(TopologyTests, TopologyParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
