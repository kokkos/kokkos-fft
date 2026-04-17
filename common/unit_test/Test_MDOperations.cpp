// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Traits.hpp"
#include "KokkosFFT_MDOperations.hpp"
#include "Test_Utils.hpp"

namespace {
#if defined(KOKKOS_ENABLE_SERIAL)
using base_execution_space_types =
    std::tuple<Kokkos::Serial, Kokkos::DefaultHostExecutionSpace,
               Kokkos::DefaultExecutionSpace>;
#else
using base_execution_space_types = std::tuple<Kokkos::DefaultHostExecutionSpace,
                                              Kokkos::DefaultExecutionSpace>;
#endif

using base_int_types    = std::tuple<int, std::size_t>;
using base_layout_types = std::tuple<Kokkos::LayoutLeft, Kokkos::LayoutRight>;

using test_types =
    tuple_to_types_t<cartesian_product_t<base_execution_space_types,
                                         base_int_types, base_layout_types>>;

template <typename T>
struct TestGetMDPolicy : public ::testing::Test {
  using execution_space_type = typename std::tuple_element_t<0, T>;
  using index_type           = typename std::tuple_element_t<1, T>;
  using layout_type          = typename std::tuple_element_t<2, T>;
};

template <typename ExecutionSpace, typename IndexType, typename LayoutType,
          std::size_t DIM>
void test_get_mdpolicy() {
  using data_type    = KokkosFFT::Impl::add_pointer_n_t<double, DIM>;
  using ViewType     = Kokkos::View<data_type, LayoutType, ExecutionSpace>;
  using extents_type = std::array<std::size_t, DIM>;

  extents_type extents{};
  for (std::size_t i = 0; i < extents.size(); i++) {
    extents.at(i) = i % 2 == 0 ? 3 : 5;
  }
  auto layout = KokkosFFT::Impl::create_layout<LayoutType>(extents);

  ExecutionSpace exec;
  ViewType x("x", layout);
  if constexpr (DIM == 1) {
    // Range Policy
    using range_policy_type =
        Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<IndexType>>;
    auto policy = KokkosFFT::Impl::get_mdpolicy<IndexType>(exec, x);
    testing::StaticAssertTypeEq<decltype(policy), range_policy_type>();
    ASSERT_EQ(policy.space(), exec);
    ASSERT_EQ(policy.begin(), 0);
    ASSERT_EQ(policy.end(), x.extent(0));
  } else {
    // MDRange policy
    constexpr std::size_t rank_truncated = std::min(DIM, std::size_t(6));
    static const Kokkos::Iterate outer_iteration_pattern =
        KokkosFFT::Impl::layout_iterate_type_selector<
            LayoutType>::outer_iteration_pattern;
    static const Kokkos::Iterate inner_iteration_pattern =
        KokkosFFT::Impl::layout_iterate_type_selector<
            LayoutType>::inner_iteration_pattern;
    using iterate_type = Kokkos::Rank<rank_truncated, outer_iteration_pattern,
                                      inner_iteration_pattern>;
    using mdrange_policy_type =
        Kokkos::MDRangePolicy<ExecutionSpace, iterate_type,
                              Kokkos::IndexType<IndexType>>;

    using point_type = typename mdrange_policy_type::point_type;
    auto policy      = KokkosFFT::Impl::get_mdpolicy<IndexType>(exec, x);
    testing::StaticAssertTypeEq<decltype(policy), mdrange_policy_type>();

    point_type ref_lower{}, ref_upper{};
    for (std::size_t i = 0; i < rank_truncated; ++i) {
      ref_lower[i] = 0;
      ref_upper[i] = x.extent(i);
    }

    ASSERT_EQ(policy.space(), exec);
    ASSERT_EQ(policy.m_lower, ref_lower);
    ASSERT_EQ(policy.m_upper, ref_upper);
  }
}

}  // namespace

TYPED_TEST_SUITE(TestGetMDPolicy, test_types);

// Test create policy for 1D
TYPED_TEST(TestGetMDPolicy, Range1D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using index_type           = typename TestFixture::index_type;
  using layout_type          = typename TestFixture::layout_type;
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 1>();
}

// Test create policy for 2D-8D
TYPED_TEST(TestGetMDPolicy, Range2Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using index_type           = typename TestFixture::index_type;
  using layout_type          = typename TestFixture::layout_type;
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 2>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 3>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 4>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 5>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 6>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 7>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 8>();
}
