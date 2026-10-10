#include "KokkosFFT_Batched_Base_Types.hpp"
#include <concepts>
#include <gtest/gtest.h>

namespace {
using KokkosFFT::Batched::AxisTag;

void test_axis_tag() {
  using MyAxes = AxisTag<0, 2, 4>;
  static_assert(MyAxes::head::value == 0, "Head should be 0");
  static_assert(std::same_as<MyAxes::tail, AxisTag<2, 4>>,
                "Tail should be AxisTag<2, 4>");
  static_assert(std::same_as<MyAxes::heads, AxisTag<0, 2>>,
                "Heads should be AxisTag<0, 2>");
  static_assert(MyAxes::last_v == 4, "Last should be 4");
  static_assert(MyAxes::rank == 3, "Rank should be 3");

  using OneAxis = AxisTag<1>;
  static_assert(std::same_as<OneAxis::heads, AxisTag<>>,
                "Heads of a single axis should be empty");
  static_assert(OneAxis::last_v == 1, "Last should be 1");
  static_assert(OneAxis::rank == 1, "Rank should be 1");
}
}  // namespace

TEST(CompileTestBaseTypes, AxisTag) { test_axis_tag(); }
