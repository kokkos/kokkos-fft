// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_traits.hpp"
#include "Test_Utils.hpp"

// All the tests in this file are compile time tests, so we skip all the tests
// by GTEST_SKIP().

// Define the types to combine
using base_real_types = std::tuple<float, double, long double>;

// Define the layouts to combine
using base_layout_types =
    std::tuple<Kokkos::LayoutLeft, Kokkos::LayoutRight, Kokkos::LayoutStride>;

using real_types = ::testing::Types<float, double, long double>;
using view_types =
    ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                     std::pair<float, Kokkos::LayoutRight>,
                     std::pair<float, Kokkos::LayoutStride>,
                     std::pair<double, Kokkos::LayoutLeft>,
                     std::pair<double, Kokkos::LayoutRight>,
                     std::pair<double, Kokkos::LayoutStride>,
                     std::pair<long double, Kokkos::LayoutLeft>,
                     std::pair<long double, Kokkos::LayoutRight>,
                     std::pair<long double, Kokkos::LayoutStride>>;

// Define all the combinations
using paired_value_types =
    tuple_to_types_t<cartesian_product_t<base_real_types, base_real_types>>;

using paired_layout_types =
    tuple_to_types_t<cartesian_product_t<base_layout_types, base_layout_types>>;

using paired_view_types =
    tuple_to_types_t<cartesian_product_t<base_real_types, base_layout_types,
                                         base_real_types, base_layout_types>>;

template <typename T>
struct RealAndComplexTypes : public ::testing::Test {
  using real_type    = T;
  using complex_type = Kokkos::complex<T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct RealAndComplexViewTypes : public ::testing::Test {
  using real_type    = typename T::first_type;
  using complex_type = Kokkos::complex<real_type>;
  using layout_type  = typename T::second_type;
  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct PairedValueTypes : public ::testing::Test {
  using real_type1 = typename std::tuple_element_t<0, T>;
  using real_type2 = typename std::tuple_element_t<1, T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct PairedLayoutTypes : public ::testing::Test {
  using layout_type1 = typename std::tuple_element_t<0, T>;
  using layout_type2 = typename std::tuple_element_t<1, T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct PairedViewTypes : public ::testing::Test {
  using real_type1   = typename std::tuple_element_t<0, T>;
  using layout_type1 = typename std::tuple_element_t<1, T>;
  using real_type2   = typename std::tuple_element_t<2, T>;
  using layout_type2 = typename std::tuple_element_t<3, T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

TYPED_TEST_SUITE(RealAndComplexTypes, real_types);
TYPED_TEST_SUITE(RealAndComplexViewTypes, view_types);
TYPED_TEST_SUITE(PairedValueTypes, paired_value_types);
TYPED_TEST_SUITE(PairedLayoutTypes, paired_layout_types);
TYPED_TEST_SUITE(PairedViewTypes, paired_view_types);

// Tests for real type deduction
template <typename RealType, typename ComplexType>
void test_get_real_type() {
  using real_type_from_RealType =
      KokkosFFT::Impl::base_floating_point_type<RealType>;
  using real_type_from_ComplexType =
      KokkosFFT::Impl::base_floating_point_type<ComplexType>;

  static_assert(std::is_same_v<real_type_from_RealType, RealType>,
                "Real type not deduced correctly from real type");
  static_assert(std::is_same_v<real_type_from_ComplexType, RealType>,
                "Real type not deduced correctly from complex type");
}

// Tests for admissible real types (float or double)
template <typename T>
void test_admissible_real_type() {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    static_assert(KokkosFFT::Impl::is_real_v<T>,
                  "Real type must be float or double");
  } else {
    static_assert(!KokkosFFT::Impl::is_real_v<T>,
                  "Real type must be float or double");
  }
}

template <typename T>
void test_admissible_complex_type() {
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (std::is_same_v<real_type, float> ||
                std::is_same_v<real_type, double>) {
    static_assert(KokkosFFT::Impl::is_complex_v<T>,
                  "Complex type must be Kokkos::complex<float> or "
                  "Kokkos::complex<double>");
  } else {
    static_assert(!KokkosFFT::Impl::is_complex_v<T>,
                  "Complex type must be Kokkos::complex<float> or "
                  "Kokkos::complex<double>");
  }
}

TYPED_TEST(RealAndComplexTypes, get_real_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;

  test_get_real_type<real_type, complex_type>();
}

TYPED_TEST(RealAndComplexTypes, admissible_real_type) {
  using real_type = typename TestFixture::real_type;

  test_admissible_real_type<real_type>();
}

TYPED_TEST(RealAndComplexTypes, admissible_complex_type) {
  using complex_type = typename TestFixture::complex_type;

  test_admissible_complex_type<complex_type>();
}

// Tests for admissible view types
template <typename T, typename LayoutType>
void test_admissible_value_type() {
  using ViewType  = Kokkos::View<T*, LayoutType>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (std::is_same_v<real_type, float> ||
                std::is_same_v<real_type, double>) {
    static_assert(KokkosFFT::Impl::is_admissible_value_type_v<ViewType>,
                  "Real type must be float or double");
  } else {
    static_assert(!KokkosFFT::Impl::is_admissible_value_type_v<ViewType>,
                  "Real type must be float or double");
  }
}

template <typename T, typename LayoutType>
void test_admissible_layout_type() {
  using ViewType = Kokkos::View<T*, LayoutType>;
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<ViewType>,
                  "View Layout must be either LayoutLeft or LayoutRight.");
  } else {
    static_assert(!KokkosFFT::Impl::is_layout_left_or_right_v<ViewType>,
                  "View Layout must be either LayoutLeft or LayoutRight.");
  }
}

template <typename T, typename LayoutType>
void test_admissible_view_type() {
  using ViewType  = Kokkos::View<T*, LayoutType>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (
      (std::is_same_v<real_type, float> || std::is_same_v<real_type, double>)&&(
          std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
          std::is_same_v<LayoutType, Kokkos::LayoutRight>)) {
    static_assert(KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                  "View value type must be float, double, "
                  "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
                  "must be either LayoutLeft or LayoutRight.");
  } else {
    static_assert(!KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                  "View value type must be float, double, "
                  "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
                  "must be either LayoutLeft or LayoutRight.");
  }
}

// \brief Test if a View is operatable
// \tparam ExecutionSpace1 Execution space for the device
// \tparam ExecutionSpace2 Execution space for the View memory space
// \tparam T Value type of the View
// \tparam LayoutType Layout type of the View
template <typename ExecutionSpace1, typename ExecutionSpace2, typename T,
          typename LayoutType>
void test_operatable_view_type() {
  using ViewType  = Kokkos::View<T*, LayoutType, ExecutionSpace2>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  if constexpr (Kokkos::SpaceAccessibility<
                    ExecutionSpace1,
                    typename ViewType::memory_space>::accessible) {
    if constexpr ((std::is_same_v<real_type, float> ||
                   std::is_same_v<
                       real_type,
                       double>)&&(std::is_same_v<LayoutType,
                                                 Kokkos::LayoutLeft> ||
                                  std::is_same_v<LayoutType,
                                                 Kokkos::LayoutRight>)) {
      static_assert(
          KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
          "View value type must be float, double, "
          "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
          "must be either LayoutLeft or LayoutRight.");
    } else {
      static_assert(
          !KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
          "View value type must be float, double, "
          "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
          "must be either LayoutLeft or LayoutRight.");
    }
  } else {
    static_assert(
        !KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
        "execution_space cannot access data in ViewType");
  }
}

TYPED_TEST(RealAndComplexViewTypes, admissible_value_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_value_type<real_type, layout_type>();
  test_admissible_value_type<complex_type, layout_type>();
}

TYPED_TEST(RealAndComplexViewTypes, admissible_layout_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_layout_type<real_type, layout_type>();
  test_admissible_layout_type<complex_type, layout_type>();
}

TYPED_TEST(RealAndComplexViewTypes, admissible_view_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_view_type<real_type, layout_type>();
  test_admissible_view_type<complex_type, layout_type>();
}

TYPED_TEST(RealAndComplexViewTypes, operatable_view_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  using host_space   = Kokkos::DefaultHostExecutionSpace;
  using device_space = Kokkos::DefaultExecutionSpace;

  test_operatable_view_type<host_space, host_space, real_type, layout_type>();
  test_operatable_view_type<host_space, device_space, real_type, layout_type>();
  test_operatable_view_type<device_space, host_space, real_type, layout_type>();
  test_operatable_view_type<device_space, device_space, real_type,
                            layout_type>();

  test_operatable_view_type<host_space, host_space, complex_type,
                            layout_type>();
  test_operatable_view_type<host_space, device_space, complex_type,
                            layout_type>();
  test_operatable_view_type<device_space, host_space, complex_type,
                            layout_type>();
  test_operatable_view_type<device_space, device_space, complex_type,
                            layout_type>();
}

// Tests for multiple Views
template <typename RealType1, typename RealType2>
void test_have_same_precision() {
  using real_type1    = RealType1;
  using real_type2    = RealType2;
  using complex_type1 = Kokkos::complex<real_type1>;
  using complex_type2 = Kokkos::complex<real_type2>;

  using RealViewType1    = Kokkos::View<real_type1*>;
  using ComplexViewType1 = Kokkos::View<complex_type1*>;
  using RealViewType2    = Kokkos::View<real_type2*>;
  using ComplexViewType2 = Kokkos::View<complex_type2*>;

  if constexpr (std::is_same_v<real_type1, real_type2>) {
    // Tests for values
    static_assert(
        KokkosFFT::Impl::have_same_precision_v<real_type1, complex_type1>,
        "Values have the same base precisions");
    static_assert(
        KokkosFFT::Impl::have_same_precision_v<complex_type1, real_type2>,
        "Values have the same base precisions");
    static_assert(
        KokkosFFT::Impl::have_same_precision_v<complex_type1, complex_type2>,
        "Values have the same base precisions");

    // Tests for Views
    static_assert(
        KokkosFFT::Impl::have_same_precision_v<RealViewType1, RealViewType2>,
        "ViewTypes have the same base precisions");
    static_assert(
        KokkosFFT::Impl::have_same_precision_v<RealViewType1, ComplexViewType2>,
        "ViewTypes have the same base precisions");
    static_assert(
        KokkosFFT::Impl::have_same_precision_v<ComplexViewType1, RealViewType2>,
        "ViewTypes have the same base precisions");
    static_assert(KokkosFFT::Impl::have_same_precision_v<ComplexViewType1,
                                                         ComplexViewType2>,
                  "ViewTypes have the same base precisions");
  } else {
    // Tests for values
    static_assert(
        !KokkosFFT::Impl::have_same_precision_v<complex_type1, real_type2>,
        "Values have the same base precisions");
    static_assert(
        !KokkosFFT::Impl::have_same_precision_v<complex_type1, complex_type2>,
        "Values have the same base precisions");

    // Tests for Views
    static_assert(
        !KokkosFFT::Impl::have_same_precision_v<RealViewType1, RealViewType2>,
        "ViewTypes have the same base precisions");
    static_assert(!KokkosFFT::Impl::have_same_precision_v<RealViewType1,
                                                          ComplexViewType2>,
                  "ViewTypes have the same base precisions");
    static_assert(!KokkosFFT::Impl::have_same_precision_v<ComplexViewType1,
                                                          RealViewType2>,
                  "ViewTypes have the same base precisions");
    static_assert(!KokkosFFT::Impl::have_same_precision_v<ComplexViewType1,
                                                          ComplexViewType2>,
                  "ViewTypes have the same base precisions");
  }
}

template <typename LayoutType1, typename LayoutType2>
void test_have_same_layout() {
  using RealType  = double;
  using ViewType1 = Kokkos::View<RealType*, LayoutType1>;
  using ViewType2 = Kokkos::View<RealType*, LayoutType2>;

  if constexpr (std::is_same_v<LayoutType1, LayoutType2>) {
    // Tests for Views
    static_assert(KokkosFFT::Impl::have_same_layout_v<ViewType1, ViewType2>,
                  "ViewTypes have the same layout");
  } else {
    // Tests for Views
    static_assert(!KokkosFFT::Impl::have_same_layout_v<ViewType1, ViewType2>,
                  "ViewTypes have the same layout");
  }
}

template <typename LayoutType1, typename LayoutType2>
void test_have_same_rank() {
  using RealType                   = double;
  using DynamicRank1ViewType       = Kokkos::View<RealType*, LayoutType1>;
  using DynamicRank2ViewType       = Kokkos::View<RealType**, LayoutType1>;
  using StaticRank1ViewType        = Kokkos::View<RealType[3], LayoutType2>;
  using StaticRank2ViewType        = Kokkos::View<RealType[2][5], LayoutType2>;
  using DynamicStaticRank2ViewType = Kokkos::View<RealType* [5], LayoutType1>;
  static_assert(KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                  StaticRank1ViewType>,
                "ViewTypes have the same rank");
  static_assert(KokkosFFT::Impl::have_same_rank_v<DynamicRank2ViewType,
                                                  StaticRank2ViewType>,
                "ViewTypes have the same rank");
  static_assert(KokkosFFT::Impl::have_same_rank_v<DynamicRank2ViewType,
                                                  DynamicStaticRank2ViewType>,
                "ViewTypes have the same rank");

  static_assert(!KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                   DynamicRank2ViewType>,
                "ViewTypes have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                   StaticRank2ViewType>,
                "ViewTypes have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                   DynamicStaticRank2ViewType>,
                "ViewTypes have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<StaticRank1ViewType,
                                                   DynamicRank2ViewType>,
                "ViewTypes have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<StaticRank1ViewType,
                                                   StaticRank2ViewType>,
                "ViewTypes have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<StaticRank1ViewType,
                                                   DynamicStaticRank2ViewType>,
                "ViewTypes have the same rank");
}

// \brief Test if two Views are operatable
// \tparam ExecutionSpace1 Execution space for the device
// \tparam ExecutionSpace2 Execution space for the View memory space
// \tparam RealType1 Base Real Value type of the View1
// \tparam LayoutType1 Layout type of the View1
// \tparam RealType2 Base Real Value type of the View2
// \tparam LayoutType2 Layout type of the View2
template <typename ExecutionSpace1, typename ExecutionSpace2,
          typename RealType1, typename LayoutType1, typename RealType2,
          typename LayoutType2>
void test_are_operatable_views() {
  using real_type1    = RealType1;
  using real_type2    = RealType2;
  using complex_type1 = Kokkos::complex<real_type1>;
  using complex_type2 = Kokkos::complex<real_type2>;

  using RealViewType1 = Kokkos::View<real_type1*, LayoutType1, ExecutionSpace2>;
  using ComplexViewType1 =
      Kokkos::View<complex_type1*, LayoutType1, ExecutionSpace2>;
  using RealViewType2 = Kokkos::View<real_type2*, LayoutType2, ExecutionSpace2>;
  using ComplexViewType2 =
      Kokkos::View<complex_type2*, LayoutType2, ExecutionSpace2>;
  using RealViewType3 =
      Kokkos::View<real_type2**, LayoutType2, ExecutionSpace2>;
  using ComplexViewType3 =
      Kokkos::View<complex_type2* [3], LayoutType2, ExecutionSpace2>;

  // Tests that the Views are accessible from the ExecutionSpace
  if constexpr (Kokkos::SpaceAccessibility<
                    ExecutionSpace1,
                    typename RealViewType1::memory_space>::accessible) {
    // Tests that the Views have the same precision in float or double
    if constexpr (std::is_same_v<RealType1, RealType2> &&
                  (std::is_same_v<RealType1, float> ||
                   std::is_same_v<RealType1, double>)) {
      // Tests that the Views have the same layout in LayoutLeft or LayoutRight
      if constexpr (std::is_same_v<LayoutType1, LayoutType2> &&
                    (std::is_same_v<LayoutType1, Kokkos::LayoutLeft> ||
                     std::is_same_v<LayoutType1, Kokkos::LayoutRight>)) {
        // Tests that the Views are operatable if they have the same rank
        static_assert(KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, RealViewType1, RealViewType2>,
                      "InViewType and OutViewType must have the same rank");
        static_assert(KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, RealViewType1, ComplexViewType2>,
                      "InViewType and OutViewType must have the same rank");
        static_assert(KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, ComplexViewType1, RealViewType2>,
                      "InViewType and OutViewType must have the same rank");
        static_assert(KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, ComplexViewType1, ComplexViewType2>,
                      "InViewType and OutViewType must have the same rank");

        // Tests that the Views are not operatable if the ranks are not the same
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, RealViewType1, RealViewType3>,
                      "InViewType and OutViewType must have the same rank");
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, RealViewType1, ComplexViewType3>,
                      "InViewType and OutViewType must have the same rank");
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, ComplexViewType1, RealViewType3>,
                      "InViewType and OutViewType must have the same rank");
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, ComplexViewType1, ComplexViewType3>,
                      "InViewType and OutViewType must have the same rank");
      } else {
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, RealViewType1, RealViewType2>,
                      "Layouts are not identical or one of them is not "
                      "LayoutLeft or LayoutRight");
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, RealViewType1, ComplexViewType2>,
                      "Layouts are not identical or one of them is not "
                      "LayoutLeft or LayoutRight");
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, ComplexViewType1, RealViewType2>,
                      "Layouts are not identical or one of them is not "
                      "LayoutLeft or LayoutRight");
        static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                          ExecutionSpace1, ComplexViewType1, ComplexViewType2>,
                      "Layouts are not identical or one of them is not "
                      "LayoutLeft or LayoutRight");
      }
    } else {
      static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                        ExecutionSpace1, RealViewType1, RealViewType2>,
                    "Base value types are not identical or one of them is not "
                    "float or double");
      static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                        ExecutionSpace1, RealViewType1, ComplexViewType2>,
                    "Base value types are not identical or one of them is not "
                    "float or double");
      static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                        ExecutionSpace1, ComplexViewType1, RealViewType2>,
                    "Base value types are not identical or one of them is not "
                    "float or double");
      static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                        ExecutionSpace1, ComplexViewType1, ComplexViewType2>,
                    "Base value types are not identical or one of them is not "
                    "float or double");
    }
  } else {
    // Views are not operatable because they are not accessible from
    // ExecutionSpace
    static_assert(
        !KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace1, RealViewType1,
                                                 RealViewType2>,
        "Either InViewType or OutViewType is not accessible from "
        "ExecutionSpace");
    static_assert(
        !KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace1, RealViewType1,
                                                 ComplexViewType2>,
        "Either InViewType or OutViewType is not accessible from "
        "ExecutionSpace");
    static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                      ExecutionSpace1, ComplexViewType1, RealViewType2>,
                  "Either InViewType or OutViewType is not accessible from "
                  "ExecutionSpace");
    static_assert(!KokkosFFT::Impl::are_operatable_views_v<
                      ExecutionSpace1, ComplexViewType1, ComplexViewType2>,
                  "Either InViewType or OutViewType is not accessible from "
                  "ExecutionSpace");
  }
}

TYPED_TEST(PairedValueTypes, have_same_precision) {
  using real_type1 = typename TestFixture::real_type1;
  using real_type2 = typename TestFixture::real_type2;

  test_have_same_precision<real_type1, real_type2>();
}

TYPED_TEST(PairedLayoutTypes, have_same_layout) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_have_same_layout<layout_type1, layout_type2>();
}

TYPED_TEST(PairedLayoutTypes, have_same_rank) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_have_same_rank<layout_type1, layout_type2>();
}

TYPED_TEST(PairedViewTypes, are_operatable_views) {
  using real_type1   = typename TestFixture::real_type1;
  using layout_type1 = typename TestFixture::layout_type1;
  using real_type2   = typename TestFixture::real_type2;
  using layout_type2 = typename TestFixture::layout_type2;
  using host_space   = Kokkos::DefaultHostExecutionSpace;
  using device_space = Kokkos::DefaultExecutionSpace;

  test_are_operatable_views<host_space, host_space, real_type1, layout_type1,
                            real_type2, layout_type2>();
  test_are_operatable_views<host_space, device_space, real_type1, layout_type1,
                            real_type2, layout_type2>();
  test_are_operatable_views<device_space, host_space, real_type1, layout_type1,
                            real_type2, layout_type2>();
  test_are_operatable_views<device_space, device_space, real_type1,
                            layout_type1, real_type2, layout_type2>();
}
