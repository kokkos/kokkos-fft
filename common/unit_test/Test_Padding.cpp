// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <random>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <Kokkos_Random.hpp>

#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_Layout.hpp"
#include "KokkosFFT_Padding.hpp"
#include "KokkosFFT_Testing_Allclose.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;

template <std::size_t DIM>
using shape_type = KokkosFFT::shape_type<DIM>;

template <std::size_t DIM>
using axes_type = KokkosFFT::axis_type<DIM>;

using test_types = ::testing::Types<double, Kokkos::complex<double>>;

using layout_types =
    ::testing::Types<std::pair<Kokkos::LayoutLeft, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutLeft, Kokkos::LayoutRight>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutLeft>,
                     std::pair<Kokkos::LayoutRight, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestModifiedShape1D : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct TestModifiedShape2D : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct TestModifiedShape3D : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct TestPadding : public ::testing::Test {
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

/// \brief Helper function to create a set of random trials for testing padding
/// and reshaping \tparam IntType The integer type for extents \tparam DIM The
/// rank of the extents
///
/// \return A vector of random extents for a given rank, with values {-1, 0, 1}
template <typename IntType, std::size_t DIM>
auto get_trials(int num_trials) {
  // Random number generator
  std::mt19937 gen(12345);

  // Distribution: {-1, 0, 1}
  std::uniform_int_distribution<int> dist(-1, 1);

  std::vector<std::array<IntType, DIM>> trials(num_trials);
  for (auto& trial : trials) {
    for (auto& diff : trial) {
      diff = dist(gen);
    }
  }
  return trials;
}

/// \brief Helper function to compute the output extents for transpose based on
/// the input extents and map
/// \tparam ContainerType The type of the map container
/// \tparam IntType The integer type for extents
/// \tparam DIM The rank of the extents
///
/// \param[in] extents The input extents
/// \param[in] perturbation The perturbation for padding or cropping
/// \return A perturbed extents based on the input extents and perturbation.
template <typename IntType, typename SignedIntType, std::size_t DIM>
auto get_out_extents(const std::array<IntType, DIM>& extents,
                     const std::array<SignedIntType, DIM>& perturbation) {
  auto out_extents = extents;

  for (std::size_t i = 0; i < out_extents.size(); i++) {
    // Perturb the output extents to try padding or cropping
    out_extents.at(i) += static_cast<IntType>(perturbation.at(i));
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
/// \param[in] in The input view
/// \param[out] out The output view
/// \param[in] src_idx The source indices
/// \param[in] dst_idx The destination indices
template <typename InViewType, typename OutViewType, std::size_t DIM,
          std::size_t... Is>
void make_padded_internal(const InViewType& in, const OutViewType& out,
                          const std::array<std::size_t, DIM>& indices,
                          std::index_sequence<Is...>) {
  out(indices[Is]...) = in(indices[Is]...);
}

/// \brief Helper function to create a reference after transpose
/// \tparam InViewType The type of the input view
/// \tparam OutViewType The type of the output view
/// \tparam DIM The rank of the Views
///
/// \param[in] in The input view
/// \param[out] out The output view after padding
template <typename InViewType, typename OutViewType>
void make_padded(const InViewType& in, const OutViewType& out) {
  static_assert(InViewType::rank() == OutViewType::rank(),
                "make_padded: Rank of Input and Output Views must be equal");
  auto h_in  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, in);
  auto h_out = Kokkos::create_mirror_view(out);

  const std::size_t n0 = InViewType::rank() > 0 ? h_in.extent(0) : 1;
  const std::size_t n1 = InViewType::rank() > 1 ? h_in.extent(1) : 1;
  const std::size_t n2 = InViewType::rank() > 2 ? h_in.extent(2) : 1;
  const std::size_t n3 = InViewType::rank() > 3 ? h_in.extent(3) : 1;
  const std::size_t n4 = InViewType::rank() > 4 ? h_in.extent(4) : 1;
  const std::size_t n5 = InViewType::rank() > 5 ? h_in.extent(5) : 1;
  const std::size_t n6 = InViewType::rank() > 6 ? h_in.extent(6) : 1;
  const std::size_t n7 = InViewType::rank() > 7 ? h_in.extent(7) : 1;

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
                  bool in_bound = true;
                  for (std::size_t i = 0; i < InViewType::rank(); ++i) {
                    in_bound &= indices.at(i) < h_out.extent(i);
                  }
                  if (in_bound) {
                    make_padded_internal(
                        h_in, h_out, indices,
                        std::make_index_sequence<InViewType::rank()>{});
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  Kokkos::deep_copy(out, h_out);
}

template <typename T, int DIM>
void test_modified_shape_1d() {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;

  using in_data_type  = KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using out_data_type = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using InViewType =
      Kokkos::View<in_data_type, Kokkos::LayoutRight, execution_space>;
  using OutViewType =
      Kokkos::View<out_data_type, Kokkos::LayoutRight, execution_space>;

  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;

  auto out_extents = get_extents<std::size_t, DIM>();
  auto out_layout =
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents);

  OutViewType x_out("x_out", out_layout);

  auto default_extents = out_extents;
  for (int axis0 = -1; axis0 < DIM; axis0++) {
    auto in_extents                  = default_extents;
    int non_negative_axis            = axis0 < 0 ? axis0 + DIM : axis0;
    in_extents.at(non_negative_axis) = KokkosFFT::Impl::extent_after_transform(
        x_out.extent(non_negative_axis), is_C2R);

    auto in_layout =
        KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents);
    InViewType x_in("x_in", in_layout);
    for (int i0 = -1; i0 <= 1; i0++) {
      auto ref_extents   = default_extents;
      std::size_t n0_new = x_in.extent_int(non_negative_axis) + i0;
      ref_extents.at(non_negative_axis) =
          KokkosFFT::Impl::extent_after_transform(n0_new, is_C2R);
      shape_type<1> new_extents = {n0_new};
      auto modified_extents     = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, new_extents, axes_type<1>{axis0});
      EXPECT_TRUE(modified_extents == ref_extents);
    }
  }
}

template <typename T, int DIM>
void test_modified_shape_2d() {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;

  using in_data_type  = KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using out_data_type = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using InViewType =
      Kokkos::View<in_data_type, Kokkos::LayoutRight, execution_space>;
  using OutViewType =
      Kokkos::View<out_data_type, Kokkos::LayoutRight, execution_space>;

  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;

  auto out_extents = get_extents<std::size_t, DIM>();
  auto out_layout =
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents);

  OutViewType x_out("x_out", out_layout);

  auto default_extents = out_extents;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      auto in_extents = default_extents;
      in_extents.at(axis1) =
          KokkosFFT::Impl::extent_after_transform(x_out.extent(axis1), is_C2R);

      auto in_layout =
          KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents);
      InViewType x_in("x_in", in_layout);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          auto ref_extents      = default_extents;
          std::size_t n0_new    = x_in.extent_int(axis0) + i0;
          std::size_t n1_new    = x_in.extent_int(axis1) + i1;
          ref_extents.at(axis0) = n0_new;
          ref_extents.at(axis1) =
              KokkosFFT::Impl::extent_after_transform(n1_new, is_C2R);
          shape_type<2> new_extents = {n0_new, n1_new};
          auto modified_extents     = KokkosFFT::Impl::get_modified_shape(
              x_in, x_out, new_extents, axes);
          EXPECT_TRUE(modified_extents == ref_extents);
        }
      }
    }
  }
}

template <typename T, int DIM>
void test_modified_shape_3d() {
  using float_type   = KokkosFFT::Impl::base_floating_point_type<T>;
  using complex_type = Kokkos::complex<float_type>;

  using in_data_type  = KokkosFFT::Impl::add_pointer_n_t<complex_type, DIM>;
  using out_data_type = KokkosFFT::Impl::add_pointer_n_t<T, DIM>;
  using InViewType =
      Kokkos::View<in_data_type, Kokkos::LayoutRight, execution_space>;
  using OutViewType =
      Kokkos::View<out_data_type, Kokkos::LayoutRight, execution_space>;

  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;

  auto out_extents = get_extents<std::size_t, DIM>();
  auto out_layout =
      KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(out_extents);

  OutViewType x_out("x_out", out_layout);

  auto default_extents = out_extents;
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        auto in_extents      = default_extents;
        in_extents.at(axis2) = KokkosFFT::Impl::extent_after_transform(
            x_out.extent(axis2), is_C2R);

        auto in_layout =
            KokkosFFT::Impl::create_layout<Kokkos::LayoutRight>(in_extents);
        InViewType x_in("x_in", in_layout);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              auto ref_extents   = default_extents;
              std::size_t n0_new = x_in.extent_int(axis0) + i0;
              std::size_t n1_new = x_in.extent_int(axis1) + i1;
              std::size_t n2_new = x_in.extent_int(axis2) + i2;

              ref_extents.at(axis0) = n0_new;
              ref_extents.at(axis1) = n1_new;
              ref_extents.at(axis2) =
                  KokkosFFT::Impl::extent_after_transform(n2_new, is_C2R);

              shape_type<3> new_extents = {n0_new, n1_new, n2_new};

              auto modified_extents = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_extents, axes);
              EXPECT_TRUE(modified_extents == ref_extents);
            }
          }
        }
      }
    }
  }
}

template <typename LayoutType1, typename LayoutType2, std::size_t DIM>
void test_padding() {
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

  const int nb_trials = 16;
  auto trials         = get_trials<int, DIM>(nb_trials);

  for (auto& trial : trials) {
    auto out_extents = get_out_extents(in_extents, trial);
    auto out_layout  = KokkosFFT::Impl::create_layout<LayoutType2>(out_extents);
    ViewLayout2type x_padded("x_padded", out_layout),
        x_padded_ref("x_padded_ref", out_layout);
    make_padded(x, x_padded_ref);
    KokkosFFT::Impl::crop_or_pad(exec, x, x_padded);
    exec.fence();
    EXPECT_THAT(x_padded,
                KokkosFFT::Testing::allclose(x_padded_ref, 1.e-5, 1.e-12));
  }
}

}  // namespace

TYPED_TEST_SUITE(TestModifiedShape1D, test_types);
TYPED_TEST_SUITE(TestModifiedShape2D, test_types);
TYPED_TEST_SUITE(TestModifiedShape3D, test_types);

TYPED_TEST_SUITE(TestPadding, layout_types);

TYPED_TEST(TestModifiedShape1D, 1DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 1>();
}

TYPED_TEST(TestModifiedShape1D, 2DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 2>();
}

TYPED_TEST(TestModifiedShape1D, 3DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 3>();
}

TYPED_TEST(TestModifiedShape1D, 4DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 4>();
}

TYPED_TEST(TestModifiedShape1D, 5DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 5>();
}

TYPED_TEST(TestModifiedShape1D, 6DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 6>();
}

TYPED_TEST(TestModifiedShape1D, 7DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 7>();
}

TYPED_TEST(TestModifiedShape1D, 8DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_1d<float_type, 8>();
}

TYPED_TEST(TestModifiedShape2D, 2DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 2>();
}

TYPED_TEST(TestModifiedShape2D, 3DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 3>();
}

TYPED_TEST(TestModifiedShape2D, 4DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 4>();
}

TYPED_TEST(TestModifiedShape2D, 5DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 5>();
}

TYPED_TEST(TestModifiedShape2D, 6DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 6>();
}

TYPED_TEST(TestModifiedShape2D, 7DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 7>();
}

TYPED_TEST(TestModifiedShape2D, 8DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_2d<float_type, 8>();
}

TYPED_TEST(TestModifiedShape3D, 3DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_3d<float_type, 3>();
}

TYPED_TEST(TestModifiedShape3D, 4DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_3d<float_type, 4>();
}

TYPED_TEST(TestModifiedShape3D, 5DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_3d<float_type, 5>();
}

TYPED_TEST(TestModifiedShape3D, 6DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_3d<float_type, 6>();
}

TYPED_TEST(TestModifiedShape3D, 7DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_3d<float_type, 7>();
}

TYPED_TEST(TestModifiedShape3D, 8DView) {
  using float_type = typename TestFixture::float_type;
  test_modified_shape_3d<float_type, 8>();
}

TYPED_TEST(TestPadding, 1DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 1>();
}

TYPED_TEST(TestPadding, 2DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 2>();
}

TYPED_TEST(TestPadding, 3DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 3>();
}

TYPED_TEST(TestPadding, 4DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 4>();
}

TYPED_TEST(TestPadding, 5DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 5>();
}

TYPED_TEST(TestPadding, 6DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 6>();
}

TYPED_TEST(TestPadding, 7DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 7>();
}

TYPED_TEST(TestPadding, 8DView) {
  using layout_type1 = typename TestFixture::layout_type1;
  using layout_type2 = typename TestFixture::layout_type2;

  test_padding<layout_type1, layout_type2, 8>();
}
