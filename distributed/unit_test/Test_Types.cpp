// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include "KokkosFFT_Distributed_Types.hpp"

namespace {
// Type definitions for parameterized tests
using TopologyTypes =
    ::testing::Types<KokkosFFT::Distributed::Topology<int, 3>,
                     KokkosFFT::Distributed::Topology<double, 4>,
                     KokkosFFT::Distributed::Topology<float, 5>,
                     KokkosFFT::Distributed::Topology<std::size_t, 2> >;

using DataTypes = ::testing::Types<int, double, float, std::size_t>;

// Test fixture for Topology class
template <typename TopologyType>
struct TestTopology : public ::testing::Test {
  using value_type                  = typename TopologyType::value_type;
  static constexpr std::size_t size = TopologyType{}.size();

  std::array<value_type, size> m_test_data;

 protected:
  virtual void SetUp() override {
    // Initialize test data
    for (std::size_t i = 0; i < size; ++i) {
      m_test_data[i] = static_cast<value_type>(i + 1);
    }
  }
};

template <typename T>
struct CompileTestTopologyTypes : public ::testing::Test {
  static constexpr std::size_t rank = 3;
  using value_type                  = T;
  using vector_type                 = std::vector<T>;
  using std_array_type              = std::array<T, rank>;
  using topology_left_type =
      KokkosFFT::Distributed::Topology<T, rank, Kokkos::LayoutLeft>;
  using topology_right_type =
      KokkosFFT::Distributed::Topology<T, rank, Kokkos::LayoutRight>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

}  // namespace

TYPED_TEST_SUITE(CompileTestTopologyTypes, DataTypes);
TYPED_TEST_SUITE(TestTopology, TopologyTypes);

TYPED_TEST(CompileTestTopologyTypes, is_topology) {
  using vector_type         = typename TestFixture::vector_type;
  using std_array_type      = typename TestFixture::std_array_type;
  using topology_left_type  = typename TestFixture::topology_left_type;
  using topology_right_type = typename TestFixture::topology_right_type;

  static_assert(!KokkosFFT::Distributed::is_topology_v<vector_type>,
                "std::vector should not be recognized as Topology");
  static_assert(!KokkosFFT::Distributed::is_topology_v<std_array_type>,
                "std::array should not be recognized as Topology");
  static_assert(KokkosFFT::Distributed::is_topology_v<topology_left_type>,
                "Topology with LayoutLeft should be recognized as Topology");
  static_assert(KokkosFFT::Distributed::is_topology_v<topology_right_type>,
                "Topology with LayoutRight should be recognized as Topology");
}

TYPED_TEST(CompileTestTopologyTypes, is_allowed_topology) {
  using vector_type         = typename TestFixture::vector_type;
  using std_array_type      = typename TestFixture::std_array_type;
  using topology_left_type  = typename TestFixture::topology_left_type;
  using topology_right_type = typename TestFixture::topology_right_type;

  static_assert(!KokkosFFT::Distributed::is_allowed_topology_v<vector_type>,
                "std::vector should not be allowed as Topology");
  static_assert(KokkosFFT::Distributed::is_allowed_topology_v<std_array_type>,
                "std::array should be allowed as Topology");
  static_assert(
      KokkosFFT::Distributed::is_allowed_topology_v<topology_left_type>,
      "Topology with LayoutLeft should be allowed as Topology");
  static_assert(
      KokkosFFT::Distributed::is_allowed_topology_v<topology_right_type>,
      "Topology with LayoutRight should be allowed as Topology");
}

TYPED_TEST(CompileTestTopologyTypes, are_allowed_topologies) {
  using vector_type         = typename TestFixture::vector_type;
  using std_array_type      = typename TestFixture::std_array_type;
  using topology_left_type  = typename TestFixture::topology_left_type;
  using topology_right_type = typename TestFixture::topology_right_type;

  static_assert(!KokkosFFT::Distributed::are_allowed_topologies_v<vector_type>,
                "std::vector should not be recognized as Topology");
  static_assert(
      KokkosFFT::Distributed::are_allowed_topologies_v<std_array_type>,
      "std::array should be recognized as Topology");
  static_assert(
      KokkosFFT::Distributed::are_allowed_topologies_v<topology_left_type>,
      "Topology with LayoutLeft should be recognized as Topology");
  static_assert(
      KokkosFFT::Distributed::are_allowed_topologies_v<topology_right_type>,
      "Topology with LayoutRight should be recognized as Topology");

  static_assert(
      !KokkosFFT::Distributed::are_allowed_topologies_v<std_array_type,
                                                        vector_type>,
      "Combination with std::vector should not be allowed");
  static_assert(
      KokkosFFT::Distributed::are_allowed_topologies_v<std_array_type,
                                                       topology_left_type>,
      "Combination with std::array and Topology should be allowed");
  static_assert(
      KokkosFFT::Distributed::are_allowed_topologies_v<topology_left_type,
                                                       topology_right_type>,
      "Combination with Topologies should be allowed");
}

// Test type definitions and aliases
TYPED_TEST(CompileTestTopologyTypes, internal_types) {
  using value_type          = typename TestFixture::value_type;
  using topology_left_type  = typename TestFixture::topology_left_type;
  using topology_right_type = typename TestFixture::topology_right_type;

  // Topology Left
  testing::StaticAssertTypeEq<typename topology_left_type::value_type,
                              value_type>();
  testing::StaticAssertTypeEq<typename topology_left_type::size_type,
                              std::size_t>();
  testing::StaticAssertTypeEq<typename topology_left_type::difference_type,
                              std::ptrdiff_t>();
  testing::StaticAssertTypeEq<typename topology_left_type::reference,
                              value_type&>();
  testing::StaticAssertTypeEq<typename topology_left_type::const_reference,
                              const value_type&>();
  testing::StaticAssertTypeEq<typename topology_left_type::pointer,
                              value_type*>();
  testing::StaticAssertTypeEq<typename topology_left_type::const_pointer,
                              const value_type*>();
  testing::StaticAssertTypeEq<typename topology_left_type::layout_type,
                              Kokkos::LayoutLeft>();

  // Topology Right
  testing::StaticAssertTypeEq<typename topology_right_type::value_type,
                              value_type>();
  testing::StaticAssertTypeEq<typename topology_right_type::size_type,
                              std::size_t>();
  testing::StaticAssertTypeEq<typename topology_right_type::difference_type,
                              std::ptrdiff_t>();
  testing::StaticAssertTypeEq<typename topology_right_type::reference,
                              value_type&>();
  testing::StaticAssertTypeEq<typename topology_right_type::const_reference,
                              const value_type&>();
  testing::StaticAssertTypeEq<typename topology_right_type::pointer,
                              value_type*>();
  testing::StaticAssertTypeEq<typename topology_right_type::const_pointer,
                              const value_type*>();
  testing::StaticAssertTypeEq<typename topology_right_type::layout_type,
                              Kokkos::LayoutRight>();
}

// Test default constructor
TYPED_TEST(TestTopology, default_constructor) {
  TypeParam topology;
  EXPECT_EQ(topology.size(), this->m_test_data.size());
  EXPECT_EQ(topology.empty(), this->m_test_data.empty());
}

// Test constructor from std::array
TYPED_TEST(TestTopology, array_constructor) {
  TypeParam topology(this->m_test_data);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology[i], this->m_test_data[i]);
  }
}

// Test constructor from initializer list
TEST(TestTopologyConstructor, initializer_list_constructor) {
  KokkosFFT::Distributed::Topology<int, 3> topology{1, 2, 3};

  EXPECT_EQ(topology[0], 1);
  EXPECT_EQ(topology[1], 2);
  EXPECT_EQ(topology[2], 3);
}

// Test initializer list constructor with wrong size
TEST(TestTopologyConstructor, initializer_list_wrong_size) {
  EXPECT_THROW(({
                 KokkosFFT::Distributed::Topology<int, 3> topology{
                     1, 2, 3, 4};  // Too many elements
               }),
               std::length_error);

  EXPECT_THROW(({
                 KokkosFFT::Distributed::Topology<int, 3> topology{
                     1, 2};  // Too few elements
               }),
               std::length_error);
}

// Test copy constructor
TYPED_TEST(TestTopology, copy_constructor) {
  TypeParam original(this->m_test_data);
  TypeParam copy(original);

  EXPECT_EQ(original, copy);
  for (std::size_t i = 0; i < copy.size(); ++i) {
    EXPECT_EQ(copy[i], original[i]);
  }
}

// Test move constructor
TYPED_TEST(TestTopology, move_constructor) {
  TypeParam original(this->m_test_data);
  TypeParam expected = original;
  TypeParam moved(std::move(original));

  EXPECT_EQ(moved, expected);
}

// Test copy assignment
TYPED_TEST(TestTopology, copy_assignment) {
  TypeParam original(this->m_test_data);
  TypeParam assigned;

  assigned = original;
  EXPECT_EQ(assigned, original);
}

// Test move assignment
TYPED_TEST(TestTopology, move_assignment) {
  TypeParam original(this->m_test_data);
  TypeParam expected = original;
  TypeParam assigned;

  assigned = std::move(original);
  EXPECT_EQ(assigned, expected);
}

// Test element access with bounds checking
TYPED_TEST(TestTopology, element_access_at) {
  TypeParam topology(this->m_test_data);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology.at(i), this->m_test_data[i]);
  }

  // Test const version
  const TypeParam& const_topology = topology;
  for (std::size_t i = 0; i < const_topology.size(); ++i) {
    EXPECT_EQ(const_topology.at(i), this->m_test_data[i]);
  }
}

// Test bounds checking in at() method
TYPED_TEST(TestTopology, element_access_at_bounds_check) {
  TypeParam topology(this->m_test_data);

  EXPECT_THROW(topology.at(topology.size()), std::out_of_range);
  EXPECT_THROW(topology.at(topology.size() + 1), std::out_of_range);
}

// Test element access without bounds checking
TYPED_TEST(TestTopology, element_access_brackets) {
  TypeParam topology(this->m_test_data);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology[i], this->m_test_data[i]);
  }

  // Test const version
  const TypeParam& const_topology = topology;
  for (std::size_t i = 0; i < const_topology.size(); ++i) {
    EXPECT_EQ(const_topology[i], this->m_test_data[i]);
  }
}

// Test front() and back() methods
TYPED_TEST(TestTopology, front_and_back) {
  if (TypeParam{}.size() == 0) return;  // Skip for empty arrays

  TypeParam topology(this->m_test_data);

  EXPECT_EQ(topology.front(), this->m_test_data.front());
  EXPECT_EQ(topology.back(), this->m_test_data.back());

  // Test const versions
  const TypeParam& const_topology = topology;
  EXPECT_EQ(const_topology.front(), this->m_test_data.front());
  EXPECT_EQ(const_topology.back(), this->m_test_data.back());
}

// Test data() method
TYPED_TEST(TestTopology, data_access) {
  TypeParam topology(this->m_test_data);

  auto* data_ptr = topology.data();
  EXPECT_NE(data_ptr, nullptr);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(data_ptr[i], this->m_test_data[i]);
  }

  // Test const version
  const TypeParam& const_topology = topology;
  const auto* const_data_ptr      = const_topology.data();
  EXPECT_NE(const_data_ptr, nullptr);

  for (std::size_t i = 0; i < const_topology.size(); ++i) {
    EXPECT_EQ(const_data_ptr[i], this->m_test_data[i]);
  }
}

// Test iterators
TYPED_TEST(TestTopology, iterators) {
  TypeParam topology(this->m_test_data);

  // Test begin/end
  auto it = topology.begin();
  for (std::size_t i = 0; i < topology.size(); ++i, ++it) {
    EXPECT_EQ(*it, this->m_test_data[i]);
  }
  EXPECT_EQ(it, topology.end());

  // Test const iterators
  const TypeParam& const_topology = topology;
  auto const_it                   = const_topology.begin();
  for (std::size_t i = 0; i < const_topology.size(); ++i, ++const_it) {
    EXPECT_EQ(*const_it, this->m_test_data[i]);
  }
  EXPECT_EQ(const_it, const_topology.end());

  // Test cbegin/cend
  auto cit = topology.cbegin();
  for (std::size_t i = 0; i < topology.size(); ++i, ++cit) {
    EXPECT_EQ(*cit, this->m_test_data[i]);
  }
  EXPECT_EQ(cit, topology.cend());
}

// Test reverse iterators
TYPED_TEST(TestTopology, reverse_iterators) {
  TypeParam topology(this->m_test_data);

  // Test rbegin/rend
  auto rit = topology.rbegin();
  for (std::size_t i = topology.size(); i > 0; --i, ++rit) {
    EXPECT_EQ(*rit, this->m_test_data[i - 1]);
  }
  EXPECT_EQ(rit, topology.rend());

  // Test const reverse iterators
  const TypeParam& const_topology = topology;
  auto const_rit                  = const_topology.rbegin();
  for (std::size_t i = const_topology.size(); i > 0; --i, ++const_rit) {
    EXPECT_EQ(*const_rit, this->m_test_data[i - 1]);
  }
  EXPECT_EQ(const_rit, const_topology.rend());

  // Test crbegin/crend
  auto crit = topology.crbegin();
  for (std::size_t i = topology.size(); i > 0; --i, ++crit) {
    EXPECT_EQ(*crit, this->m_test_data[i - 1]);
  }
  EXPECT_EQ(crit, topology.crend());
}

// Test capacity methods
TYPED_TEST(TestTopology, capacity) {
  TypeParam topology;

  EXPECT_EQ(topology.size(), TypeParam{}.size());
  EXPECT_EQ(topology.max_size(), TypeParam{}.size());
  EXPECT_EQ(topology.empty(), TypeParam{}.size() == 0);
}

// Test fill method
TYPED_TEST(TestTopology, fill) {
  TypeParam topology;
  using ValueType      = typename TypeParam::value_type;
  ValueType fill_value = static_cast<ValueType>(42);

  topology.fill(fill_value);

  for (std::size_t i = 0; i < topology.size(); ++i) {
    EXPECT_EQ(topology[i], fill_value);
  }
}

// Test swap method
TYPED_TEST(TestTopology, swap) {
  TypeParam topology1(this->m_test_data);
  TypeParam topology2;

  using ValueType      = typename TypeParam::value_type;
  ValueType fill_value = static_cast<ValueType>(99);
  topology2.fill(fill_value);

  TypeParam expected1 = topology1;
  TypeParam expected2 = topology2;

  topology1.swap(topology2);

  EXPECT_EQ(topology1, expected2);
  EXPECT_EQ(topology2, expected1);
}

// Test comparison operators
TYPED_TEST(TestTopology, comparison_operators) {
  TypeParam topology1(this->m_test_data);
  TypeParam topology2(this->m_test_data);
  TypeParam topology3;

  using ValueType      = typename TypeParam::value_type;
  ValueType fill_value = static_cast<ValueType>(99);
  topology3.fill(fill_value);

  // Equality
  EXPECT_TRUE(topology1 == topology2);
  EXPECT_FALSE(topology1 == topology3);

  // Inequality
  EXPECT_FALSE(topology1 != topology2);
  EXPECT_TRUE(topology1 != topology3);

  // Less than
  EXPECT_FALSE(topology1 < topology2);
  EXPECT_TRUE(topology1 < topology3);

  // Less than or equal
  EXPECT_TRUE(topology1 <= topology2);
  EXPECT_FALSE(topology3 <= topology2);

  // Greater than
  EXPECT_FALSE(topology1 > topology2);
  EXPECT_TRUE(topology3 > topology2);

  // Greater than or equal
  EXPECT_TRUE(topology1 >= topology2);
  EXPECT_FALSE(topology1 >= topology3);
}

// Test array() method
TYPED_TEST(TestTopology, array_access) {
  TypeParam topology(this->m_test_data);

  const auto& const_array = topology.array();
  EXPECT_EQ(const_array, this->m_test_data);

  auto& array = topology.array();
  EXPECT_EQ(array, this->m_test_data);

  // Modify through array reference
  if (array.size() > 0) {
    using ValueType     = typename TypeParam::value_type;
    ValueType new_value = static_cast<ValueType>(999);
    array[0]            = new_value;
    EXPECT_EQ(topology[0], new_value);
  }
}

// Test non-member swap function
TEST(TestTopologyNonMember, swap_function) {
  KokkosFFT::Distributed::Topology<int, 3> topology1{1, 2, 3};
  KokkosFFT::Distributed::Topology<int, 3> topology2{4, 5, 6};

  KokkosFFT::Distributed::Topology<int, 3> expected1 = topology1;
  KokkosFFT::Distributed::Topology<int, 3> expected2 = topology2;

  swap(topology1, topology2);

  EXPECT_EQ(topology1, expected2);
  EXPECT_EQ(topology2, expected1);
}

// Test non-member get functions
TEST(TestTopologyNonMember, get_functions) {
  KokkosFFT::Distributed::Topology<int, 3> topology{10, 20, 30};

  // Non-const get
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(topology), 10);
  EXPECT_EQ(KokkosFFT::Distributed::get<1>(topology), 20);
  EXPECT_EQ(KokkosFFT::Distributed::get<2>(topology), 30);

  // Const get
  const auto& const_topology = topology;
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(const_topology), 10);
  EXPECT_EQ(KokkosFFT::Distributed::get<1>(const_topology), 20);
  EXPECT_EQ(KokkosFFT::Distributed::get<2>(const_topology), 30);

  // Rvalue get
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(
                KokkosFFT::Distributed::Topology<int, 3>{10, 20, 30}),
            10);
  EXPECT_EQ(KokkosFFT::Distributed::get<1>(
                KokkosFFT::Distributed::Topology<int, 3>{10, 20, 30}),
            20);
  EXPECT_EQ(KokkosFFT::Distributed::get<2>(
                KokkosFFT::Distributed::Topology<int, 3>{10, 20, 30}),
            30);

  // Const rvalue get
  const KokkosFFT::Distributed::Topology<int, 3> const_rvalue{10, 20, 30};
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(std::move(const_rvalue)), 10);
}

// Test get function with modification
TEST(TestTopologyNonMember, get_function_modification) {
  KokkosFFT::Distributed::Topology<int, 3> topology{10, 20, 30};

  KokkosFFT::Distributed::get<0>(topology) = 100;
  EXPECT_EQ(topology[0], 100);
  EXPECT_EQ(KokkosFFT::Distributed::get<0>(topology), 100);
}

// Test with different layout types
TEST(TestTopologyLayout, different_layouts) {
  using TopologyRight =
      KokkosFFT::Distributed::Topology<int, 3, Kokkos::LayoutRight>;
  using TopologyLeft =
      KokkosFFT::Distributed::Topology<int, 3, Kokkos::LayoutLeft>;

  TopologyRight right_topology{1, 2, 3};
  TopologyLeft left_topology{1, 2, 3};

  // Both should have the same data
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(right_topology[i], left_topology[i]);
  }

  // But different layout types
  testing::StaticAssertTypeEq<typename TopologyRight::layout_type,
                              Kokkos::LayoutRight>();
  testing::StaticAssertTypeEq<typename TopologyLeft::layout_type,
                              Kokkos::LayoutLeft>();
}

// Test empty topology (size 0)
TEST(TestTopologySpecial, empty_topology) {
  KokkosFFT::Distributed::Topology<int, 0> empty_topology;

  EXPECT_EQ(empty_topology.size(), 0);
  EXPECT_EQ(empty_topology.max_size(), 0);
  EXPECT_TRUE(empty_topology.empty());

  EXPECT_EQ(empty_topology.begin(), empty_topology.end());
  EXPECT_EQ(empty_topology.cbegin(), empty_topology.cend());
  EXPECT_EQ(empty_topology.rbegin(), empty_topology.rend());
  EXPECT_EQ(empty_topology.crbegin(), empty_topology.crend());
}

// Test large topology
TEST(TestTopologySpecial, large_topology) {
  constexpr std::size_t large_size = 1000;
  KokkosFFT::Distributed::Topology<std::size_t, large_size> large_topology;

  // Fill with indices
  for (std::size_t i = 0; i < large_size; ++i) {
    large_topology[i] = i;
  }

  // Verify
  for (std::size_t i = 0; i < large_size; ++i) {
    EXPECT_EQ(large_topology[i], i);
  }

  EXPECT_EQ(large_topology.size(), large_size);
  EXPECT_FALSE(large_topology.empty());
}

// Test constexpr operations
TEST(TestTopologyConstexpr, constexpr_operations) {
  constexpr KokkosFFT::Distributed::Topology<int, 3> topology{1, 2, 3};

  static_assert(topology.size() == 3);
  static_assert(!topology.empty());
  static_assert(topology.max_size() == 3);

  // These should compile as constexpr
  constexpr auto size     = topology.size();
  constexpr auto empty    = topology.empty();
  constexpr auto max_size = topology.max_size();

  EXPECT_EQ(size, 3);
  EXPECT_FALSE(empty);
  EXPECT_EQ(max_size, 3);
}

// Test range-based for loop
TYPED_TEST(TestTopology, range_based_for_loop) {
  TypeParam topology(this->m_test_data);

  std::size_t index = 0;
  for (const auto& element : topology) {
    EXPECT_EQ(element, this->m_test_data[index]);
    ++index;
  }
  EXPECT_EQ(index, topology.size());
}

// Test STL algorithm compatibility
TYPED_TEST(TestTopology, stl_algorithm_compatibility) {
  TypeParam topology(this->m_test_data);

  // Test std::find
  auto it = std::find(topology.begin(), topology.end(), this->m_test_data[0]);
  EXPECT_NE(it, topology.end());
  EXPECT_EQ(*it, this->m_test_data[0]);

  // Test std::count
  auto count =
      std::count(topology.begin(), topology.end(), this->m_test_data[0]);
  EXPECT_EQ(count, 1);

  // Test std::reduce
  std::size_t sum          = std::reduce(topology.begin(), topology.end(),
                                         std::size_t{0}, std::plus<std::size_t>{});
  std::size_t expected_sum = 0;
  for (const auto& val : this->m_test_data) {
    expected_sum += static_cast<std::size_t>(val);
  }
  EXPECT_EQ(sum, expected_sum);
}
