// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_traits.hpp"
#include "Test_Utils.hpp"

// All the tests in this file are compile time tests, so we skip all the tests
// by GTEST_SKIP(). gtest is used for type parameterization.

namespace {

// Int like types
using base_int_types = ::testing::Types<int, std::size_t>;

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
struct CompileTestContainerTypes : public ::testing::Test {
  static constexpr std::size_t rank = 3;
  using value_type                  = T;
  using vector_type                 = std::vector<T>;
  using std_array_type              = std::array<T, rank>;
  using Kokkos_array_type           = Kokkos::Array<T, rank>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestRealAndComplexTypes : public ::testing::Test {
  using real_type    = T;
  using complex_type = Kokkos::complex<T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestRealAndComplexViewTypes : public ::testing::Test {
  using real_type    = typename T::first_type;
  using complex_type = Kokkos::complex<real_type>;
  using layout_type  = typename T::second_type;
  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestPairedValueTypes : public ::testing::Test {
  using real_type1 = typename std::tuple_element_t<0, T>;
  using real_type2 = typename std::tuple_element_t<1, T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestPairedLayoutTypes : public ::testing::Test {
  using layout_type1 = typename std::tuple_element_t<0, T>;
  using layout_type2 = typename std::tuple_element_t<1, T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

template <typename T>
struct CompileTestPairedViewTypes : public ::testing::Test {
  using real_type1   = typename std::tuple_element_t<0, T>;
  using layout_type1 = typename std::tuple_element_t<1, T>;
  using real_type2   = typename std::tuple_element_t<2, T>;
  using layout_type2 = typename std::tuple_element_t<3, T>;

  virtual void SetUp() {
    GTEST_SKIP() << "Skipping all tests for this fixture";
  }
};

// Tests for host execution space
void test_is_any_host_exec_space() {
#if defined(KOKKOS_ENABLE_SERIAL)
  static_assert(KokkosFFT::Impl::is_AnyHostSpace_v<Kokkos::Serial>,
                "Kokkos::Serial must be a HostSpace");
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
  static_assert(KokkosFFT::Impl::is_AnyHostSpace_v<Kokkos::OpenMP>,
                "Kokkos::OpenMP must be a HostSpace");
#endif
#if defined(KOKKOS_ENABLE_THREADS)
  static_assert(KokkosFFT::Impl::is_AnyHostSpace_v<Kokkos::Threads>,
                "Kokkos::Threads must be a HostSpace");
#endif
}

// Tests for base value type deduction
template <typename ValueType, typename ContainerType>
void test_get_container_value_type() {
  using value_type_ContainerType =
      KokkosFFT::Impl::base_container_value_type<ContainerType>;

  // base value type of ContainerType is ValueType
  static_assert(std::is_same_v<value_type_ContainerType, ValueType>,
                "Value type not deduced correctly from ContainerType");
}

// Tests for real type deduction
template <typename RealType, typename ComplexType>
void test_get_real_type() {
  using real_type_from_RealType =
      KokkosFFT::Impl::base_floating_point_type<RealType>;
  using real_type_from_ComplexType =
      KokkosFFT::Impl::base_floating_point_type<ComplexType>;

  // base floating point type of RealType is RealType
  static_assert(std::is_same_v<real_type_from_RealType, RealType>,
                "Real type not deduced correctly from real type");

  // base floating point type of Kokkos::complex<RealType> is RealType
  static_assert(std::is_same_v<real_type_from_ComplexType, RealType>,
                "Real type not deduced correctly from complex type");
}

// Tests for admissible real types (float or double)
template <typename T>
void test_admissible_real_type() {
  // Tests that a real type is float or double
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    // T is float or double
    static_assert(KokkosFFT::Impl::is_real_v<T>,
                  "Real type must be float or double");
  } else {
    // T is not float or double
    static_assert(!KokkosFFT::Impl::is_real_v<T>,
                  "Real type must not be float or double");
  }
}

template <typename T>
void test_admissible_complex_type() {
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  // Tests that a base floating point type of complex value is float or double
  if constexpr (std::is_same_v<real_type, float> ||
                std::is_same_v<real_type, double>) {
    // T is Kokkos::complex<float> or Kokkos::complex<double>
    static_assert(KokkosFFT::Impl::is_complex_v<T>,
                  "Complex type must be Kokkos::complex<float> or "
                  "Kokkos::complex<double>");
  } else {
    // T is not Kokkos::complex<float> or Kokkos::complex<double>
    static_assert(!KokkosFFT::Impl::is_complex_v<T>,
                  "Complex type must not be Kokkos::complex<float> or "
                  "Kokkos::complex<double>");
  }
}

// Tests for admissible view types
template <typename T, typename LayoutType>
void test_admissible_value_type() {
  using ViewType  = Kokkos::View<T*, LayoutType>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;
  // Tests that a Value or View has a admissible value type
  if constexpr (std::is_same_v<real_type, float> ||
                std::is_same_v<real_type, double>) {
    // Base floating point type of a Value is float or double
    static_assert(KokkosFFT::Impl::is_admissible_value_type_v<T>,
                  "Base value type must be float or double");

    // Base floating point type of a View is float or double
    static_assert(KokkosFFT::Impl::is_admissible_value_type_v<ViewType>,
                  "Base value type of a View must be float or double");
  } else {
    // Base floating point type of a Value is not float or double
    static_assert(!KokkosFFT::Impl::is_admissible_value_type_v<T>,
                  "Base value type of a View must not be float or double");

    // Base floating point type of a View is not float or double
    static_assert(!KokkosFFT::Impl::is_admissible_value_type_v<ViewType>,
                  "Base value type of a View must not be float or double");
  }
}

template <typename T, typename LayoutType>
void test_admissible_layout_type() {
  using ViewType = Kokkos::View<T*, LayoutType>;
  // Tests that the View has a layout in LayoutLeft or LayoutRight
  if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
    static_assert(KokkosFFT::Impl::is_layout_left_or_right_v<ViewType>,
                  "View Layout must be either LayoutLeft or LayoutRight.");
  } else {
    static_assert(!KokkosFFT::Impl::is_layout_left_or_right_v<ViewType>,
                  "View Layout must not be either LayoutLeft or LayoutRight.");
  }
}

template <typename T, typename LayoutType>
void test_admissible_view_type() {
  using ViewType  = Kokkos::View<T*, LayoutType>;
  using real_type = KokkosFFT::Impl::base_floating_point_type<T>;

  // Tests that the View has a base floating point type in float or double
  if constexpr ((std::is_same_v<real_type, float> ||
                 std::is_same_v<real_type, double>)) {
    // Tests that the View has a layout in LayoutLeft or LayoutRight
    if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                  std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
      static_assert(KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                    "View value type must be float, double, "
                    "Kokkos::Complex<float>, Kokkos::Complex<double>. Layout "
                    "must be either LayoutLeft or LayoutRight.");
    } else {
      // View is not admissible because layout is not in LayoutLeft or
      // LayoutRight
      static_assert(!KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                    "Layout must be either LayoutLeft or LayoutRight.");
    }
  } else {
    // View is not admissible because the base floating point type is not in
    // float or double
    static_assert(!KokkosFFT::Impl::is_admissible_view_v<ViewType>,
                  "Base value type must be float or double");
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

  // Tests that a View is accessible from the ExecutionSpace
  if constexpr (Kokkos::SpaceAccessibility<
                    ExecutionSpace1,
                    typename ViewType::memory_space>::accessible) {
    // Tests that the View has a base floating point type in float or double
    if constexpr ((std::is_same_v<real_type, float> ||
                   std::is_same_v<real_type, double>)) {
      // Tests that the View has a layout in LayoutLeft or LayoutRight
      if constexpr (std::is_same_v<LayoutType, Kokkos::LayoutLeft> ||
                    std::is_same_v<LayoutType, Kokkos::LayoutRight>) {
        // View is operatable
        static_assert(
            KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
            "View value type must be float, double, "
            "Kokkos::Complex<float>, or Kokkos::Complex<double>. Layout "
            "must be either LayoutLeft or LayoutRight.");
      } else {
        // View is not operatable because layout is not in LayoutLeft or
        // LayoutRight
        static_assert(
            !KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
            "Layout must be either LayoutLeft or LayoutRight.");
      }
    } else {
      // View is not operatable because the base floating point type is not in
      // float or double
      static_assert(
          !KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
          "Base value type must be float or double");
    }
  } else {
    // View is not operatable because it is not accessible from
    // ExecutionSpace
    static_assert(
        !KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace1, ViewType>,
        "ExecutionSpace cannot access data in ViewType");
  }
}

// Tests for multiple Views
template <typename RealType1, typename RealType2>
void test_have_same_base_floating_point_type() {
  using real_type1    = RealType1;
  using real_type2    = RealType2;
  using complex_type1 = Kokkos::complex<real_type1>;
  using complex_type2 = Kokkos::complex<real_type2>;

  using RealViewType1    = Kokkos::View<real_type1*>;
  using ComplexViewType1 = Kokkos::View<complex_type1*>;
  using RealViewType2    = Kokkos::View<real_type2*>;
  using ComplexViewType2 = Kokkos::View<complex_type2*>;

  // Tests that Values or Views have the same base floating point type
  if constexpr (std::is_same_v<real_type1, real_type2>) {
    // Values must have the same base floating point type
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<real_type1,
                                                              complex_type1>,
        "Values must have the same base floating point type");
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<complex_type1,
                                                              real_type2>,
        "Values must have the same base floating point type");
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<complex_type1,
                                                              complex_type2>,
        "Values must have the same base floating point type");

    // Views must have the same base floating point type
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<RealViewType1,
                                                              RealViewType2>,
        "ViewTypes must have the same base floating point type");
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<RealViewType1,
                                                              ComplexViewType2>,
        "ViewTypes must have the same base floating point type");
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<ComplexViewType1,
                                                              RealViewType2>,
        "ViewTypes must have the same base floating point type");
    static_assert(
        KokkosFFT::Impl::have_same_base_floating_point_type_v<ComplexViewType1,
                                                              ComplexViewType2>,
        "ViewTypes must have the same base floating point type");
  } else {
    // Values must not have the same base floating point type
    static_assert(
        !KokkosFFT::Impl::have_same_base_floating_point_type_v<complex_type1,
                                                               real_type2>,
        "Values must not have the same base floating point type");
    static_assert(
        !KokkosFFT::Impl::have_same_base_floating_point_type_v<complex_type1,
                                                               complex_type2>,
        "Values must not have the same base floating point type");

    // Views must not have the same base floating point type
    static_assert(
        !KokkosFFT::Impl::have_same_base_floating_point_type_v<RealViewType1,
                                                               RealViewType2>,
        "ViewTypes must not have the same base floating point type");
    static_assert(!KokkosFFT::Impl::have_same_base_floating_point_type_v<
                      RealViewType1, ComplexViewType2>,
                  "ViewTypes must not have the same base floating point type");
    static_assert(
        !KokkosFFT::Impl::have_same_base_floating_point_type_v<ComplexViewType1,
                                                               RealViewType2>,
        "ViewTypes must not have the same base floating point type");
    static_assert(!KokkosFFT::Impl::have_same_base_floating_point_type_v<
                      ComplexViewType1, ComplexViewType2>,
                  "ViewTypes must not have the same base floating point type");
  }
}

template <typename LayoutType1, typename LayoutType2>
void test_have_same_layout() {
  using RealType  = double;
  using ViewType1 = Kokkos::View<RealType*, LayoutType1>;
  using ViewType2 = Kokkos::View<RealType*, LayoutType2>;

  // Tests that Views have the same layout
  if constexpr (std::is_same_v<LayoutType1, LayoutType2>) {
    // Views must have the same layout
    static_assert(KokkosFFT::Impl::have_same_layout_v<ViewType1, ViewType2>,
                  "ViewTypes must have the same layout");
  } else {
    // Views must not have the same layout
    static_assert(!KokkosFFT::Impl::have_same_layout_v<ViewType1, ViewType2>,
                  "ViewTypes must not have the same layout");
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

  // Views must have the same rank
  static_assert(KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                  StaticRank1ViewType>,
                "ViewTypes must have the same rank");
  static_assert(KokkosFFT::Impl::have_same_rank_v<DynamicRank2ViewType,
                                                  StaticRank2ViewType>,
                "ViewTypes must have the same rank");
  static_assert(KokkosFFT::Impl::have_same_rank_v<DynamicRank2ViewType,
                                                  DynamicStaticRank2ViewType>,
                "ViewTypes must have the same rank");

  // Views must not have the same rank
  static_assert(!KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                   DynamicRank2ViewType>,
                "ViewTypes must not have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                   StaticRank2ViewType>,
                "ViewTypes must not have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<DynamicRank1ViewType,
                                                   DynamicStaticRank2ViewType>,
                "ViewTypes must not have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<StaticRank1ViewType,
                                                   DynamicRank2ViewType>,
                "ViewTypes must not have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<StaticRank1ViewType,
                                                   StaticRank2ViewType>,
                "ViewTypes must not have the same rank");
  static_assert(!KokkosFFT::Impl::have_same_rank_v<StaticRank1ViewType,
                                                   DynamicStaticRank2ViewType>,
                "ViewTypes must not have the same rank");
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
    // Tests that the Views have the same base floating point type in float or
    // double
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
        // Views are not operatable because they do not have the same layout in
        // LayoutLeft or LayoutRight
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
      // Views are not operatable because they do not have the same base
      // floating point type in float or double
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

}  // namespace

TYPED_TEST_SUITE(CompileTestContainerTypes, base_int_types);
TYPED_TEST_SUITE(CompileTestRealAndComplexTypes, real_types);
TYPED_TEST_SUITE(CompileTestRealAndComplexViewTypes, view_types);
TYPED_TEST_SUITE(CompileTestPairedValueTypes, paired_value_types);
TYPED_TEST_SUITE(CompileTestPairedLayoutTypes, paired_layout_types);
TYPED_TEST_SUITE(CompileTestPairedViewTypes, paired_view_types);

TEST(CompileTestExecutionSpace, test_is_any_host_exec_space) {
  GTEST_SKIP() << "Skipping all tests";
  test_is_any_host_exec_space();
}

TYPED_TEST(CompileTestContainerTypes, get_value_type_from_vector) {
  using value_type     = typename TestFixture::value_type;
  using container_type = typename TestFixture::vector_type;

  test_get_container_value_type<value_type, container_type>();
}

TYPED_TEST(CompileTestContainerTypes, get_value_type_from_std_array) {
  using value_type     = typename TestFixture::value_type;
  using container_type = typename TestFixture::std_array_type;

  test_get_container_value_type<value_type, container_type>();
}

TYPED_TEST(CompileTestContainerTypes, get_value_type_from_kokkos_array) {
  using value_type     = typename TestFixture::value_type;
  using container_type = typename TestFixture::Kokkos_array_type;

  test_get_container_value_type<value_type, container_type>();
}

TYPED_TEST(CompileTestRealAndComplexTypes, get_real_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;

  test_get_real_type<real_type, complex_type>();
}

TYPED_TEST(CompileTestRealAndComplexTypes, admissible_real_type) {
  using real_type = typename TestFixture::real_type;

  test_admissible_real_type<real_type>();
}

TYPED_TEST(CompileTestRealAndComplexTypes, admissible_complex_type) {
  using complex_type = typename TestFixture::complex_type;

  test_admissible_complex_type<complex_type>();
}

TYPED_TEST(CompileTestRealAndComplexViewTypes, admissible_value_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_value_type<real_type, layout_type>();
  test_admissible_value_type<complex_type, layout_type>();
}

TYPED_TEST(CompileTestRealAndComplexViewTypes, admissible_layout_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_layout_type<real_type, layout_type>();
  test_admissible_layout_type<complex_type, layout_type>();
}

TYPED_TEST(CompileTestRealAndComplexViewTypes, admissible_view_type) {
  using real_type    = typename TestFixture::real_type;
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;

  test_admissible_view_type<real_type, layout_type>();
  test_admissible_view_type<complex_type, layout_type>();
}

TYPED_TEST(CompileTestRealAndComplexViewTypes, operatable_view_type) {
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

TYPED_TEST(CompileTestPairedValueTypes, have_same_base_floating_point_type) {
  using real_type1 = typename TestFixture::real_type1;
  using real_type2 = typename TestFixture::real_type2;

  test_have_same_base_floating_point_type<real_type1, real_type2>();
}

TYPED_TEST(CompileTestPairedLayoutTypes, have_same_layout) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_have_same_layout<layout_type1, layout_type2>();
}

TYPED_TEST(CompileTestPairedLayoutTypes, have_same_rank) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_have_same_rank<layout_type1, layout_type2>();
}

TYPED_TEST(CompileTestPairedViewTypes, are_operatable_views) {
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
