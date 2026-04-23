// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <utility>
#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_Mapping.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_Transpose.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

// Int like types
using int_types = ::testing::Types<int, std::size_t>;

using layout_types =
    ::testing::Types<std::pair<Kokkos::LayoutLeft, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutLeft, Kokkos::LayoutRight>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestIsTransposeNeeded : public ::testing::Test {
  using value_type = T;
};

template <typename T>
struct TestTranspose1D : public ::testing::Test {
  using layout_type1 = typename T::first_type;
  using layout_type2 = typename T::second_type;
};

template <typename T>
struct TestTranspose2D : public ::testing::Test {
  using layout_type1 = typename T::first_type;
  using layout_type2 = typename T::second_type;
};

template <typename T>
struct TestTranspose3D : public ::testing::Test {
  using layout_type1 = typename T::first_type;
  using layout_type2 = typename T::second_type;
};

/// \brief Helper function to test md_unary_operation for 1D-8D
/// \tparam IntType The integer type for extents
/// \tparam DIM The rank of the extents
///
/// \return An extents for a given rank, with values {3, 5, 3, 5, ...}
template <typename IntType, std::size_t DIM>
auto get_extents() {
  std::array<IntType, DIM> extents{};
  for (std::size_t i = 0; i < extents.size(); i++) {
    extents.at(i) = i % 2 == 0 ? 3 : 5;
  }
  return extents;
}

/// \brief Helper function to compute the output extents for transpose based on
/// the input extents and map
/// \tparam ContainerType The type of the map container
/// \tparam IntType The integer type for extents
/// \tparam DIM The rank of the extents
///
/// \param[in] extents The input extents
/// \param[in] map The map for permutation
/// \param[in] bounds_check Whether to perturb the output extents to trigger
/// out-of-bounds access
/// \return A permuted extents based on the input extents and map.
/// If bounds_check is true, the output extents are perturbed by +/- 1
/// to trigger out-of-bounds access
template <typename ContainerType, typename IntType, std::size_t DIM>
auto get_out_extents(const std::array<IntType, DIM>& extents,
                     const ContainerType& map, bool bounds_check) {
  auto out_extents = KokkosFFT::Impl::compute_mapped_extents(extents, map);

  if (bounds_check) {
    for (std::size_t i = 0; i < out_extents.size(); i++) {
      // Perturb the output extents to trigger out-of-bounds access if we
      // accidentally use the input extents instead of the output extents.
      IntType perturbation = i % 2 == 0 ? 1 : -1;
      out_extents.at(i) += perturbation;
    }
  }
  return out_extents;
}

/// \brief Helper function to create a reference after transpose, with given
/// source and destination indices
/// \tparam InViewType The type of the input view
/// \tparam OutViewType The type of the output view
/// \tparam DIM The rank of the Views
/// \tparam Is The index sequence for unpacking
///
/// \param[in] x The input view
/// \param[out] xT The output view
/// \param[in] src_idx The source indices
/// \param[in] dst_idx The destination indices
template <typename InViewType, typename OutViewType, std::size_t DIM,
          std::size_t... Is>
void make_transposed_internal(const InViewType& x, const OutViewType& xT,
                              const std::array<std::size_t, DIM>& src_idx,
                              const std::array<std::size_t, DIM>& dst_idx,
                              std::index_sequence<Is...>) {
  xT(dst_idx[Is]...) = x(src_idx[Is]...);
}

/// \brief Helper function to create a reference after transpose
/// \tparam InViewType The type of the input view
/// \tparam OutViewType The type of the output view
/// \tparam DIM The rank of the Views
///
/// \param[in] x The input view
/// \param[out] xT The output view permuted according to map
/// \param[in] map The map for permutation
template <typename InViewType, typename OutViewType, std::size_t DIM>
void make_transposed(const InViewType& x, const OutViewType& xT,
                     const KokkosFFT::axis_type<DIM>& map) {
  static_assert(InViewType::rank() == DIM && OutViewType::rank() == DIM,
                "make_transposed: Rank of Views must be equal to Rank of "
                "transpose axes.");
  auto h_x  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  auto h_xT = Kokkos::create_mirror_view(xT);

  const std::size_t n0 = InViewType::rank() > 0 ? h_x.extent(0) : 1;
  const std::size_t n1 = InViewType::rank() > 1 ? h_x.extent(1) : 1;
  const std::size_t n2 = InViewType::rank() > 2 ? h_x.extent(2) : 1;
  const std::size_t n3 = InViewType::rank() > 3 ? h_x.extent(3) : 1;
  const std::size_t n4 = InViewType::rank() > 4 ? h_x.extent(4) : 1;
  const std::size_t n5 = InViewType::rank() > 5 ? h_x.extent(5) : 1;
  const std::size_t n6 = InViewType::rank() > 6 ? h_x.extent(6) : 1;
  const std::size_t n7 = InViewType::rank() > 7 ? h_x.extent(7) : 1;

  for (std::size_t i0 = 0; i0 < n0; i0++) {
    for (std::size_t i1 = 0; i1 < n1; i1++) {
      for (std::size_t i2 = 0; i2 < n2; i2++) {
        for (std::size_t i3 = 0; i3 < n3; i3++) {
          for (std::size_t i4 = 0; i4 < n4; i4++) {
            for (std::size_t i5 = 0; i5 < n5; i5++) {
              for (std::size_t i6 = 0; i6 < n6; i6++) {
                for (std::size_t i7 = 0; i7 < n7; i7++) {
                  std::array<std::size_t, 8> indices{i0, i1, i2, i3,
                                                     i4, i5, i6, i7};
                  std::array<std::size_t, DIM> src{}, dst{};
                  bool in_bound = true;
                  for (std::size_t i = 0; i < DIM; ++i) {
                    src.at(i) = indices.at(i);
                  }
                  for (std::size_t i = 0; i < DIM; ++i) {
                    dst.at(i) = src.at(map.at(i));
                    in_bound &= dst.at(i) < h_xT.extent(i);
                  }
                  if (in_bound) {
                    make_transposed_internal(h_x, h_xT, src, dst,
                                             std::make_index_sequence<DIM>{});
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(xT, h_xT);
}

template <typename IndexType>
void test_is_transpose_needed() {
  std::array<IndexType, 1> map1D{0};
  EXPECT_FALSE(KokkosFFT::Impl::is_transpose_needed(map1D));

  std::array<IndexType, 2> map2D{0, 1}, map2D_axis0{1, 0};
  EXPECT_FALSE(KokkosFFT::Impl::is_transpose_needed(map2D));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map2D_axis0));

  std::array<IndexType, 3> map3D{0, 1, 2}, map3D_021{0, 2, 1},
      map3D_102{1, 0, 2}, map3D_120{1, 2, 0}, map3D_201{2, 0, 1},
      map3D_210{2, 1, 0};

  EXPECT_FALSE(KokkosFFT::Impl::is_transpose_needed(map3D));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_021));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_102));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_120));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_201));
  EXPECT_TRUE(KokkosFFT::Impl::is_transpose_needed(map3D_210));
}

// Tests for transpose
// 1D Transpose
template <typename LayoutType1, typename LayoutType2, std::size_t DIM>
void test_transpose_1d(bool bounds_check) {
  using view_data_type = KokkosFFT::Impl::add_pointer_n_t<double, DIM>;

  using ViewLayout1type =
      Kokkos::View<view_data_type, LayoutType1, execution_space>;
  using ViewLayout2type =
      Kokkos::View<view_data_type, LayoutType2, execution_space>;

  auto in_extents = get_extents<int, DIM>();
  auto in_layout  = KokkosFFT::Impl::create_layout<LayoutType1>(in_extents);

  ViewLayout1type x("x", in_layout);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, 1.0);
  exec.fence();

  for (std::size_t axis0 = 0; axis0 < DIM; axis0++) {
    auto [map, map_inv] =
        KokkosFFT::Impl::get_map_axes(x, static_cast<int>(axis0));
    auto out_extents = get_out_extents(in_extents, map, bounds_check);
    auto out_layout  = KokkosFFT::Impl::create_layout<LayoutType2>(out_extents);
    ViewLayout2type xt("xt", out_layout), xt_ref("xt_ref", out_layout);
    make_transposed(x, xt_ref, map);

    KokkosFFT::Impl::transpose(exec, x, xt, map, bounds_check);
    EXPECT_TRUE(allclose(exec, xt, xt_ref, 1.e-5, 1.e-12));

    // Inverse (transpose of transpose is identical to the original)
    ViewLayout1type x_inv("x_inv", in_layout),
        x_inv_ref("x_inv_ref", in_layout);
    if (bounds_check) {
      // With bounds_check, we may discard some of the input data,
      // so the inverse is not identical to the original
      make_transposed(xt_ref, x_inv_ref, map_inv);
    } else {
      Kokkos::deep_copy(x_inv_ref, x);
    }

    KokkosFFT::Impl::transpose(exec, xt, x_inv, map_inv, bounds_check);
    EXPECT_TRUE(allclose(exec, x_inv, x_inv_ref, 1.e-5, 1.e-12));
    exec.fence();
  }
}

// 2D Transpose
template <typename LayoutType1, typename LayoutType2, std::size_t DIM>
void test_transpose_2d(bool bounds_check) {
  using view_data_type = KokkosFFT::Impl::add_pointer_n_t<double, DIM>;

  using ViewLayout1type =
      Kokkos::View<view_data_type, LayoutType1, execution_space>;
  using ViewLayout2type =
      Kokkos::View<view_data_type, LayoutType2, execution_space>;

  auto in_extents = get_extents<int, DIM>();
  auto in_layout  = KokkosFFT::Impl::create_layout<LayoutType1>(in_extents);

  ViewLayout1type x("x", in_layout);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, 1.0);
  exec.fence();

  for (std::size_t axis0 = 0; axis0 < DIM; axis0++) {
    for (std::size_t axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      KokkosFFT::axis_type<2> axes{static_cast<int>(axis0),
                                   static_cast<int>(axis1)};

      auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
      auto out_extents    = get_out_extents(in_extents, map, bounds_check);
      auto out_layout =
          KokkosFFT::Impl::create_layout<LayoutType2>(out_extents);
      ViewLayout2type xt("xt", out_layout), xt_ref("xt_ref", out_layout);
      make_transposed(x, xt_ref, map);

      KokkosFFT::Impl::transpose(exec, x, xt, map, bounds_check);
      EXPECT_TRUE(allclose(exec, xt, xt_ref, 1.e-5, 1.e-12));

      // Inverse (transpose of transpose is identical to the original)
      ViewLayout1type x_inv("x_inv", in_layout),
          x_inv_ref("x_inv_ref", in_layout);
      if (bounds_check) {
        // With bounds_check, we may discard some of the input data,
        // so the inverse is not identical to the original
        make_transposed(xt_ref, x_inv_ref, map_inv);
      } else {
        Kokkos::deep_copy(x_inv_ref, x);
      }

      KokkosFFT::Impl::transpose(exec, xt, x_inv, map_inv, bounds_check);
      EXPECT_TRUE(allclose(exec, x_inv, x_inv_ref, 1.e-5, 1.e-12));
      exec.fence();
    }
  }
}

// 3D Transpose
template <typename LayoutType1, typename LayoutType2, std::size_t DIM>
void test_transpose_3d(bool bounds_check) {
  using view_data_type = KokkosFFT::Impl::add_pointer_n_t<double, DIM>;

  using ViewLayout1type =
      Kokkos::View<view_data_type, LayoutType1, execution_space>;
  using ViewLayout2type =
      Kokkos::View<view_data_type, LayoutType2, execution_space>;

  auto in_extents = get_extents<int, DIM>();
  auto in_layout  = KokkosFFT::Impl::create_layout<LayoutType1>(in_extents);

  ViewLayout1type x("x", in_layout);

  execution_space exec;
  Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
  Kokkos::fill_random(exec, x, random_pool, 1.0);
  exec.fence();

  for (std::size_t axis0 = 0; axis0 < DIM; axis0++) {
    for (std::size_t axis1 = 0; axis1 < DIM; axis1++) {
      for (std::size_t axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        KokkosFFT::axis_type<3> axes{static_cast<int>(axis0),
                                     static_cast<int>(axis1),
                                     static_cast<int>(axis2)};

        auto [map, map_inv] = KokkosFFT::Impl::get_map_axes(x, axes);
        auto out_extents    = get_out_extents(in_extents, map, bounds_check);
        auto out_layout =
            KokkosFFT::Impl::create_layout<LayoutType2>(out_extents);
        ViewLayout2type xt("xt", out_layout), xt_ref("xt_ref", out_layout);
        make_transposed(x, xt_ref, map);

        KokkosFFT::Impl::transpose(exec, x, xt, map, bounds_check);
        EXPECT_TRUE(allclose(exec, xt, xt_ref, 1.e-5, 1.e-12));

        // Inverse (transpose of transpose is identical to the original)
        ViewLayout1type x_inv("x_inv", in_layout),
            x_inv_ref("x_inv_ref", in_layout);
        if (bounds_check) {
          // With bounds_check, we may discard some of the input data,
          // so the inverse is not identical to the original
          make_transposed(xt_ref, x_inv_ref, map_inv);
        } else {
          Kokkos::deep_copy(x_inv_ref, x);
        }

        KokkosFFT::Impl::transpose(exec, xt, x_inv, map_inv, bounds_check);
        EXPECT_TRUE(allclose(exec, x_inv, x_inv_ref, 1.e-5, 1.e-12));
        exec.fence();
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(TestIsTransposeNeeded, int_types);

TYPED_TEST_SUITE(TestTranspose1D, layout_types);
TYPED_TEST_SUITE(TestTranspose2D, layout_types);
TYPED_TEST_SUITE(TestTranspose3D, layout_types);

TYPED_TEST(TestIsTransposeNeeded, is_transpose_needed) {
  using value_type = typename TestFixture::value_type;
  test_is_transpose_needed<value_type>();
}

TYPED_TEST(TestTranspose1D, 1DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 1>(false);
}

TYPED_TEST(TestTranspose1D, 1DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 1>(true);
}

TYPED_TEST(TestTranspose1D, 2DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 2>(false);
}

TYPED_TEST(TestTranspose1D, 2DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 2>(true);
}

TYPED_TEST(TestTranspose1D, 3DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 3>(false);
}

TYPED_TEST(TestTranspose1D, 3DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 3>(true);
}

TYPED_TEST(TestTranspose1D, 4DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 4>(false);
}

TYPED_TEST(TestTranspose1D, 4DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 4>(true);
}

TYPED_TEST(TestTranspose1D, 5DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 5>(false);
}

TYPED_TEST(TestTranspose1D, 5DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 5>(true);
}

TYPED_TEST(TestTranspose1D, 6DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 6>(false);
}

TYPED_TEST(TestTranspose1D, 6DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 6>(true);
}

TYPED_TEST(TestTranspose1D, 7DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 7>(false);
}

TYPED_TEST(TestTranspose1D, 7DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 7>(true);
}

TYPED_TEST(TestTranspose1D, 8DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 8>(false);
}

TYPED_TEST(TestTranspose1D, 8DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_1d<layout_type1, layout_type2, 8>(true);
}

TYPED_TEST(TestTranspose2D, 2DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 2>(false);
}

TYPED_TEST(TestTranspose2D, 2DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 2>(true);
}

TYPED_TEST(TestTranspose2D, 3DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 3>(false);
}

TYPED_TEST(TestTranspose2D, 3DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 3>(true);
}

TYPED_TEST(TestTranspose2D, 4DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 4>(false);
}

TYPED_TEST(TestTranspose2D, 4DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 4>(true);
}

TYPED_TEST(TestTranspose2D, 5DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 5>(false);
}

TYPED_TEST(TestTranspose2D, 5DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 5>(true);
}

TYPED_TEST(TestTranspose2D, 6DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 6>(false);
}

TYPED_TEST(TestTranspose2D, 6DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 6>(true);
}

TYPED_TEST(TestTranspose2D, 7DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 7>(false);
}

TYPED_TEST(TestTranspose2D, 7DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 7>(true);
}

TYPED_TEST(TestTranspose2D, 8DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 8>(false);
}

TYPED_TEST(TestTranspose2D, 8DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_2d<layout_type1, layout_type2, 8>(true);
}

TYPED_TEST(TestTranspose3D, 3DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 3>(false);
}

TYPED_TEST(TestTranspose3D, 3DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 3>(true);
}

TYPED_TEST(TestTranspose3D, 4DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 4>(false);
}

TYPED_TEST(TestTranspose3D, 4DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 4>(true);
}

TYPED_TEST(TestTranspose3D, 5DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 5>(false);
}

TYPED_TEST(TestTranspose3D, 5DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 5>(true);
}

TYPED_TEST(TestTranspose3D, 6DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 6>(false);
}

TYPED_TEST(TestTranspose3D, 6DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 6>(true);
}

TYPED_TEST(TestTranspose3D, 7DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 7>(false);
}

TYPED_TEST(TestTranspose3D, 7DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 7>(true);
}

TYPED_TEST(TestTranspose3D, 8DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 8>(false);
}

TYPED_TEST(TestTranspose3D, 8DView_with_bounds_check) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_transpose_3d<layout_type1, layout_type2, 8>(true);
}
