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
  std::string msg;
  msg += "Input topology: (";
  msg += std::to_string(topology.at(0));
  for (std::size_t i = 1; i < topology.size(); ++i) {
    msg += ", " + std::to_string(topology.at(i));
  }
  msg += "), should be: " + topology_type_to_string(ref) + ", but got: " +
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
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(topo1.at(0));
  for (std::size_t i = 1; i < topo1.size(); ++i) {
    msg += ", " + std::to_string(topo1.at(i));
  }
  msg += ") and (";
  msg += std::to_string(topo2.at(0));
  for (std::size_t i = 1; i < topo2.size(); ++i) {
    msg += ", " + std::to_string(topo2.at(i));
  }
  msg +=
      "), should be: " + topology_type_to_string(ref) + ", but got: " +
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
  std::string msg;
  msg += "Input topology: (";
  msg += std::to_string(topology.at(0));
  for (std::size_t i = 1; i < topology.size(); ++i) {
    msg += ", " + std::to_string(topology.at(i));
  }
  if (expected) {
    msg += "), should be identified as " + topology_type_to_string(specified) +
           ", but it is not.";
  } else {
    msg += "), should not be identified as " +
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
  std::string msg;
  msg += "Input topologies: ";
  msg += "(" + std::to_string(topo1.at(0));
  for (std::size_t i = 1; i < topo1.size(); ++i) {
    msg += ", " + std::to_string(topo1.at(i));
  }
  msg += ") and (";
  msg += std::to_string(topo2.at(0));
  for (std::size_t i = 1; i < topo2.size(); ++i) {
    msg += ", " + std::to_string(topo2.at(i));
  }
  if (expected) {
    msg += "), should be identified as " + topology_type_to_string(specified) +
           ", but it is not.";
  } else {
    msg += "), should not be identified as " +
           topology_type_to_string(specified) + ", but it is.";
  }
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

INSTANTIATE_TEST_SUITE_P(TopologyTests, TopologyParamTests,
                         ::testing::Values(1, 2, 3, 4, 5, 6));
