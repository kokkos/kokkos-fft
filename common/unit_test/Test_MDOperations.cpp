// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Traits.hpp"
#include "KokkosFFT_MDOperations.hpp"
#include "KokkosFFT_UnaryOps.hpp"
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

using test_ops_types = tuple_to_types_t<
    cartesian_product_t<base_execution_space_types, base_layout_types>>;

template <typename T>
struct TestGetMDPolicy : public ::testing::Test {
  using execution_space_type = typename std::tuple_element_t<0, T>;
  using index_type           = typename std::tuple_element_t<1, T>;
  using layout_type          = typename std::tuple_element_t<2, T>;
};

template <typename T>
struct TestMDUnaryOperation : public ::testing::Test {
  using execution_space_type = typename std::tuple_element_t<0, T>;
  using layout_type          = typename std::tuple_element_t<1, T>;

  const double m_init_scalar = 1.0;
  const double m_scalar      = 2.0;
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

template <typename ExecutionSpace, typename LayoutType, std::size_t DIM,
          typename T, typename UnaryOpType>
void test_md_unary_operation(UnaryOpType op, T init_scalar, T ref_scalar) {
  using data_type    = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using ViewType     = Kokkos::View<data_type, LayoutType, ExecutionSpace>;
  using extents_type = std::array<std::size_t, DIM>;

  extents_type extents{};
  for (std::size_t i = 0; i < extents.size(); i++) {
    extents.at(i) = i % 2 == 0 ? 3 : 5;
  }
  auto layout = KokkosFFT::Impl::create_layout<LayoutType>(extents);

  ExecutionSpace exec;
  ViewType x("x", layout), x_ref("x_ref", layout);
  Kokkos::deep_copy(exec, x, init_scalar);
  Kokkos::deep_copy(exec, x_ref, ref_scalar);

  KokkosFFT::Impl::md_unary_operation<int>("TestMDUnaryOperation", exec, x, op);

  EXPECT_TRUE(allclose(exec, x, x_ref, 1.e-5, 1.e-12));
  exec.fence();
}

}  // namespace

TYPED_TEST_SUITE(TestGetMDPolicy, test_types);
TYPED_TEST_SUITE(TestMDUnaryOperation, test_ops_types);

// Test create policy for 1D-8D
TYPED_TEST(TestGetMDPolicy, Range1Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using index_type           = typename TestFixture::index_type;
  using layout_type          = typename TestFixture::layout_type;
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 1>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 2>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 3>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 4>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 5>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 6>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 7>();
  test_get_mdpolicy<execution_space_type, index_type, layout_type, 8>();
}

// Test conj operation for 1D-8D
TYPED_TEST(TestMDUnaryOperation, Conj1Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using layout_type          = typename TestFixture::layout_type;
  using complex_type         = Kokkos::complex<double>;
  const complex_type init_scalar(this->m_init_scalar, this->m_init_scalar);
  const complex_type ref_scalar(this->m_init_scalar, -this->m_init_scalar);

  KokkosFFT::Impl::Conjugate conj_op;
  test_md_unary_operation<execution_space_type, layout_type, 1>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 2>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 3>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 4>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 5>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 6>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 7>(
      conj_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 8>(
      conj_op, init_scalar, ref_scalar);
}

// Test add operation for 1D-8D
TYPED_TEST(TestMDUnaryOperation, Add1Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using layout_type          = typename TestFixture::layout_type;
  const double init_scalar   = this->m_init_scalar;
  const double scalar        = this->m_scalar;
  const double ref_scalar    = init_scalar + scalar;

  KokkosFFT::Impl::Add add_op(scalar);
  test_md_unary_operation<execution_space_type, layout_type, 1>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 2>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 3>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 4>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 5>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 6>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 7>(
      add_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 8>(
      add_op, init_scalar, ref_scalar);
}

// Test subtract operation for 1D-8D
TYPED_TEST(TestMDUnaryOperation, Subtract1Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using layout_type          = typename TestFixture::layout_type;
  const double init_scalar   = this->m_init_scalar;
  const double scalar        = this->m_scalar;
  const double ref_scalar    = init_scalar - scalar;

  KokkosFFT::Impl::Subtract sub_op(scalar);
  test_md_unary_operation<execution_space_type, layout_type, 1>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 2>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 3>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 4>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 5>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 6>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 7>(
      sub_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 8>(
      sub_op, init_scalar, ref_scalar);
}

// Test multiply operation for 1D-8D
TYPED_TEST(TestMDUnaryOperation, Multiply1Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using layout_type          = typename TestFixture::layout_type;
  const double init_scalar   = this->m_init_scalar;
  const double scalar        = this->m_scalar;
  const double ref_scalar    = init_scalar * scalar;

  KokkosFFT::Impl::Multiply mul_op(scalar);
  test_md_unary_operation<execution_space_type, layout_type, 1>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 2>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 3>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 4>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 5>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 6>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 7>(
      mul_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 8>(
      mul_op, init_scalar, ref_scalar);
}

// Test divide operation for 1D-8D
TYPED_TEST(TestMDUnaryOperation, Divide1Dto8D) {
  using execution_space_type = typename TestFixture::execution_space_type;
  using layout_type          = typename TestFixture::layout_type;
  const double init_scalar   = this->m_init_scalar;
  const double scalar        = this->m_scalar;
  const double ref_scalar    = init_scalar / scalar;

  KokkosFFT::Impl::Divide div_op(scalar);
  test_md_unary_operation<execution_space_type, layout_type, 1>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 2>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 3>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 4>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 5>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 6>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 7>(
      div_op, init_scalar, ref_scalar);
  test_md_unary_operation<execution_space_type, layout_type, 8>(
      div_op, init_scalar, ref_scalar);
}
