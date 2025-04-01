// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include <random>
#include "KokkosFFT_padding.hpp"
#include "Test_Utils.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, execution_space>;
template <typename T>
using View4D = Kokkos::View<T****, execution_space>;
template <typename T>
using View5D = Kokkos::View<T*****, execution_space>;
template <typename T>
using View6D = Kokkos::View<T******, execution_space>;
template <typename T>
using View7D = Kokkos::View<T*******, execution_space>;
template <typename T>
using View8D = Kokkos::View<T********, execution_space>;

template <std::size_t DIM>
using shape_type = KokkosFFT::shape_type<DIM>;

template <std::size_t DIM>
using axes_type = KokkosFFT::axis_type<DIM>;

using test_types = ::testing::Types<double, Kokkos::complex<double>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct GetModifiedShape1D : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct GetModifiedShape2D : public ::testing::Test {
  using float_type = T;
};

template <typename T>
struct GetModifiedShape3D : public ::testing::Test {
  using float_type = T;
};

auto get_c2r_shape(std::size_t len, bool is_C2R) {
  return is_C2R ? (len / 2 + 1) : len;
}

template <typename T>
void test_reshape1D_1DView() {
  const int len = 30, len_pad = 32, len_crop = 28;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;

  View1D<T> x_out("x_out", len);
  View1D<Kokkos::complex<double>> x_in("x_in", get_c2r_shape(len, is_C2R));

  auto shape = KokkosFFT::Impl::get_modified_shape(
      x_in, x_out, shape_type<1>{len}, axes_type<1>{-1});
  auto shape_pad = KokkosFFT::Impl::get_modified_shape(
      x_in, x_out, shape_type<1>{len_pad}, axes_type<1>{-1});
  auto shape_crop = KokkosFFT::Impl::get_modified_shape(
      x_in, x_out, shape_type<1>{len_crop}, axes_type<1>{-1});

  shape_type<1> ref_shape      = {get_c2r_shape(len, is_C2R)};
  shape_type<1> ref_shape_pad  = {get_c2r_shape(len_pad, is_C2R)};
  shape_type<1> ref_shape_crop = {get_c2r_shape(len_crop, is_C2R)};

  EXPECT_TRUE(shape == ref_shape);
  EXPECT_TRUE(shape_pad == ref_shape_pad);
  EXPECT_TRUE(shape_crop == ref_shape_crop);
}

template <typename T>
void test_reshape1D_2DView() {
  const int n0 = 30, n1 = 15;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;

  View2D<T> x_out("x_out", n0, n1);
  constexpr int DIM = 2;

  shape_type<2> default_shape({n0, n1});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<2> in_shape = default_shape;
    in_shape.at(axis0)     = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1]          = in_shape;
    View2D<Kokkos::complex<double>> x_in("x_in", s0, s1);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<2> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape1D_3DView() {
  const int n0 = 30, n1 = 15, n2 = 8;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;

  View3D<T> x_out("x_out", n0, n1, n2);
  constexpr int DIM = 3;

  shape_type<3> default_shape({n0, n1, n2});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<3> in_shape = default_shape;
    in_shape.at(axis0)     = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1, s2]      = in_shape;
    View3D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<3> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape1D_4DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View4D<T> x_out("x_out", n0, n1, n2, n3);

  constexpr int DIM = 4;
  shape_type<4> default_shape({n0, n1, n2, n3});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<4> in_shape = default_shape;
    in_shape.at(axis0)     = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1, s2, s3]  = in_shape;
    View4D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<4> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape1D_5DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View5D<T> x_out("x_out", n0, n1, n2, n3, n4);

  constexpr int DIM = 5;
  shape_type<5> default_shape({n0, n1, n2, n3, n4});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<5> in_shape    = default_shape;
    in_shape.at(axis0)        = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1, s2, s3, s4] = in_shape;
    View5D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<5> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape1D_6DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View6D<T> x_out("x_out", n0, n1, n2, n3, n4, n5);

  constexpr int DIM = 6;
  shape_type<6> default_shape({n0, n1, n2, n3, n4, n5});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<6> in_shape        = default_shape;
    in_shape.at(axis0)            = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1, s2, s3, s4, s5] = in_shape;
    View6D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<6> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape1D_7DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4, n6 = 7;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View7D<T> x_out("x_out", n0, n1, n2, n3, n4, n5, n6);

  constexpr int DIM = 7;
  shape_type<7> default_shape({n0, n1, n2, n3, n4, n5, n6});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<7> in_shape = default_shape;
    in_shape.at(axis0)     = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1, s2, s3, s4, s5, s6] = in_shape;
    View7D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5, s6);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<7> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape1D_8DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4, n6 = 7, n7 = 9;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View8D<T> x_out("x_out", n0, n1, n2, n3, n4, n5, n6, n7);

  constexpr int DIM = 8;
  shape_type<8> default_shape({n0, n1, n2, n3, n4, n5, n6, n7});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    shape_type<8> in_shape = default_shape;
    in_shape.at(axis0)     = get_c2r_shape(x_out.extent(axis0), is_C2R);
    auto [s0, s1, s2, s3, s4, s5, s6, s7] = in_shape;
    View8D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5, s6,
                                         s7);
    for (int i0 = -1; i0 <= 1; i0++) {
      shape_type<8> ref_shape = default_shape;
      auto n_new              = x_in.extent(axis0) + i0;
      ref_shape.at(axis0)     = get_c2r_shape(n_new, is_C2R);

      auto modified_shape = KokkosFFT::Impl::get_modified_shape(
          x_in, x_out, shape_type<1>{n_new}, axes_type<1>{axis0});

      EXPECT_TRUE(modified_shape == ref_shape);
    }
  }
}

template <typename T>
void test_reshape2D_2DView() {
  const int n0 = 30, n1 = 15;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View2D<T> x_out("x_out", n0, n1);
  constexpr int DIM = 2;
  shape_type<2> default_shape({n0, n1});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<2> in_shape = default_shape;
      in_shape.at(axis1)     = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1]          = in_shape;
      View2D<Kokkos::complex<double>> x_in("x_in", s0, s1);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<2> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape2D_3DView() {
  const int n0 = 30, n1 = 15, n2 = 8;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View3D<T> x_out("x_out", n0, n1, n2);
  constexpr int DIM = 3;
  shape_type<3> default_shape({n0, n1, n2});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<3> in_shape = default_shape;
      in_shape.at(axis1)     = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1, s2]      = in_shape;
      View3D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<3> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape2D_4DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View4D<T> x_out("x_out", n0, n1, n2, n3);
  constexpr int DIM = 4;
  shape_type<4> default_shape({n0, n1, n2, n3});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<4> in_shape = default_shape;
      in_shape.at(axis1)     = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1, s2, s3]  = in_shape;
      View4D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<4> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape2D_5DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View5D<T> x_out("x_out", n0, n1, n2, n3, n4);
  constexpr int DIM = 5;
  shape_type<5> default_shape({n0, n1, n2, n3, n4});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<5> in_shape    = default_shape;
      in_shape.at(axis1)        = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1, s2, s3, s4] = in_shape;
      View5D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<5> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape2D_6DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View6D<T> x_out("x_out", n0, n1, n2, n3, n4, n5);
  constexpr int DIM = 6;
  shape_type<6> default_shape({n0, n1, n2, n3, n4, n5});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<6> in_shape = default_shape;
      in_shape.at(axis1)     = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1, s2, s3, s4, s5] = in_shape;
      View6D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<6> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape2D_7DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4, n6 = 7;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View7D<T> x_out("x_out", n0, n1, n2, n3, n4, n5, n6);
  constexpr int DIM = 7;
  shape_type<7> default_shape({n0, n1, n2, n3, n4, n5, n6});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<7> in_shape = default_shape;
      in_shape.at(axis1)     = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1, s2, s3, s4, s5, s6] = in_shape;
      View7D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5, s6);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<7> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape2D_8DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4, n6 = 7, n7 = 9;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View8D<T> x_out("x_out", n0, n1, n2, n3, n4, n5, n6, n7);
  constexpr int DIM = 8;
  shape_type<8> default_shape({n0, n1, n2, n3, n4, n5, n6, n7});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      if (axis0 == axis1) continue;
      axes_type<2> axes({axis0, axis1});
      shape_type<8> in_shape = default_shape;
      in_shape.at(axis1)     = get_c2r_shape(x_out.extent(axis1), is_C2R);
      auto [s0, s1, s2, s3, s4, s5, s6, s7] = in_shape;
      View8D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5, s6,
                                           s7);
      for (int i0 = -1; i0 <= 1; i0++) {
        for (int i1 = -1; i1 <= 1; i1++) {
          shape_type<8> ref_shape = default_shape;
          auto n0_new             = x_in.extent(axis0) + i0;
          auto n1_new             = x_in.extent(axis1) + i1;
          ref_shape.at(axis0)     = n0_new;
          ref_shape.at(axis1)     = get_c2r_shape(n1_new, is_C2R);

          shape_type<2> new_shape = {n0_new, n1_new};

          auto modified_shape =
              KokkosFFT::Impl::get_modified_shape(x_in, x_out, new_shape, axes);
          EXPECT_TRUE(modified_shape == ref_shape);
        }
      }
    }
  }
}

template <typename T>
void test_reshape3D_3DView() {
  const int n0 = 30, n1 = 15, n2 = 8;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View3D<T> x_out("x_out", n0, n1, n2);

  constexpr int DIM = 3;
  shape_type<3> default_shape({n0, n1, n2});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        shape_type<3> in_shape = default_shape;
        in_shape.at(axis2)     = get_c2r_shape(x_out.extent(axis2), is_C2R);
        auto [s0, s1, s2]      = in_shape;
        View3D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              shape_type<3> ref_shape = default_shape;
              auto n0_new             = x_in.extent(axis0) + i0;
              auto n1_new             = x_in.extent(axis1) + i1;
              auto n2_new             = x_in.extent(axis2) + i2;

              ref_shape.at(axis0) = n0_new;
              ref_shape.at(axis1) = n1_new;
              ref_shape.at(axis2) = get_c2r_shape(n2_new, is_C2R);

              shape_type<3> new_shape = {n0_new, n1_new, n2_new};
              auto modified_shape     = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_shape, axes);

              EXPECT_TRUE(modified_shape == ref_shape);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void test_reshape3D_4DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View4D<T> x_out("x_out", n0, n1, n2, n3);

  constexpr int DIM = 4;
  shape_type<4> default_shape({n0, n1, n2, n3});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        shape_type<4> in_shape = default_shape;
        in_shape.at(axis2)     = get_c2r_shape(x_out.extent(axis2), is_C2R);
        auto [s0, s1, s2, s3]  = in_shape;
        View4D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              shape_type<4> ref_shape = default_shape;
              auto n0_new             = x_in.extent(axis0) + i0;
              auto n1_new             = x_in.extent(axis1) + i1;
              auto n2_new             = x_in.extent(axis2) + i2;

              ref_shape.at(axis0) = n0_new;
              ref_shape.at(axis1) = n1_new;
              ref_shape.at(axis2) = get_c2r_shape(n2_new, is_C2R);

              shape_type<3> new_shape = {n0_new, n1_new, n2_new};

              auto modified_shape = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_shape, axes);
              EXPECT_TRUE(modified_shape == ref_shape);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void test_reshape3D_5DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View5D<T> x_out("x_out", n0, n1, n2, n3, n4);
  constexpr int DIM = 5;
  shape_type<5> default_shape({n0, n1, n2, n3, n4});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        shape_type<5> in_shape    = default_shape;
        in_shape.at(axis2)        = get_c2r_shape(x_out.extent(axis2), is_C2R);
        auto [s0, s1, s2, s3, s4] = in_shape;
        View5D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              shape_type<5> ref_shape = default_shape;
              auto n0_new             = x_in.extent(axis0) + i0;
              auto n1_new             = x_in.extent(axis1) + i1;
              auto n2_new             = x_in.extent(axis2) + i2;

              ref_shape.at(axis0) = n0_new;
              ref_shape.at(axis1) = n1_new;
              ref_shape.at(axis2) = get_c2r_shape(n2_new, is_C2R);

              shape_type<3> new_shape = {n0_new, n1_new, n2_new};

              auto modified_shape = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_shape, axes);
              EXPECT_TRUE(modified_shape == ref_shape);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void test_reshape3D_6DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View6D<T> x_out("x_out", n0, n1, n2, n3, n4, n5);
  constexpr int DIM = 6;
  shape_type<6> default_shape({n0, n1, n2, n3, n4, n5});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        shape_type<6> in_shape = default_shape;
        in_shape.at(axis2)     = get_c2r_shape(x_out.extent(axis2), is_C2R);
        auto [s0, s1, s2, s3, s4, s5] = in_shape;
        View6D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              shape_type<6> ref_shape = default_shape;
              auto n0_new             = x_in.extent(axis0) + i0;
              auto n1_new             = x_in.extent(axis1) + i1;
              auto n2_new             = x_in.extent(axis2) + i2;

              ref_shape.at(axis0) = n0_new;
              ref_shape.at(axis1) = n1_new;
              ref_shape.at(axis2) = get_c2r_shape(n2_new, is_C2R);

              shape_type<3> new_shape = {n0_new, n1_new, n2_new};

              auto modified_shape = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_shape, axes);
              EXPECT_TRUE(modified_shape == ref_shape);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void test_reshape3D_7DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4, n6 = 7;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View7D<T> x_out("x_out", n0, n1, n2, n3, n4, n5, n6);
  constexpr int DIM = 7;
  shape_type<7> default_shape({n0, n1, n2, n3, n4, n5, n6});

  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        shape_type<7> in_shape = default_shape;
        in_shape.at(axis2)     = get_c2r_shape(x_out.extent(axis2), is_C2R);
        auto [s0, s1, s2, s3, s4, s5, s6] = in_shape;
        View7D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5,
                                             s6);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              shape_type<7> ref_shape = default_shape;
              auto n0_new             = x_in.extent(axis0) + i0;
              auto n1_new             = x_in.extent(axis1) + i1;
              auto n2_new             = x_in.extent(axis2) + i2;

              ref_shape.at(axis0) = n0_new;
              ref_shape.at(axis1) = n1_new;
              ref_shape.at(axis2) = get_c2r_shape(n2_new, is_C2R);

              shape_type<3> new_shape = {n0_new, n1_new, n2_new};

              auto modified_shape = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_shape, axes);
              EXPECT_TRUE(modified_shape == ref_shape);
            }
          }
        }
      }
    }
  }
}

template <typename T>
void test_reshape3D_8DView() {
  const int n0 = 5, n1 = 11, n2 = 10, n3 = 8, n4 = 3, n5 = 4, n6 = 7, n7 = 9;
  bool is_C2R = !KokkosFFT::Impl::is_complex_v<T>;
  View8D<T> x_out("x_out", n0, n1, n2, n3, n4, n5, n6, n7);
  constexpr int DIM = 8;
  shape_type<8> default_shape({n0, n1, n2, n3, n4, n5, n6, n7});
  for (int axis0 = 0; axis0 < DIM; axis0++) {
    for (int axis1 = 0; axis1 < DIM; axis1++) {
      for (int axis2 = 0; axis2 < DIM; axis2++) {
        if (axis0 == axis1 || axis0 == axis2 || axis1 == axis2) continue;
        axes_type<3> axes({axis0, axis1, axis2});
        shape_type<8> in_shape = default_shape;
        in_shape.at(axis2)     = get_c2r_shape(x_out.extent(axis2), is_C2R);
        auto [s0, s1, s2, s3, s4, s5, s6, s7] = in_shape;
        View8D<Kokkos::complex<double>> x_in("x_in", s0, s1, s2, s3, s4, s5, s6,
                                             s7);
        for (int i0 = -1; i0 <= 1; i0++) {
          for (int i1 = -1; i1 <= 1; i1++) {
            for (int i2 = -1; i2 <= 1; i2++) {
              shape_type<8> ref_shape = default_shape;
              auto n0_new             = x_in.extent(axis0) + i0;
              auto n1_new             = x_in.extent(axis1) + i1;
              auto n2_new             = x_in.extent(axis2) + i2;

              ref_shape.at(axis0) = n0_new;
              ref_shape.at(axis1) = n1_new;
              ref_shape.at(axis2) = get_c2r_shape(n2_new, is_C2R);

              shape_type<3> new_shape = {n0_new, n1_new, n2_new};

              auto modified_shape = KokkosFFT::Impl::get_modified_shape(
                  x_in, x_out, new_shape, axes);
              EXPECT_TRUE(modified_shape == ref_shape);
            }
          }
        }
      }
    }
  }
}

}  // namespace

TYPED_TEST_SUITE(GetModifiedShape1D, test_types);
TYPED_TEST_SUITE(GetModifiedShape2D, test_types);
TYPED_TEST_SUITE(GetModifiedShape3D, test_types);

TYPED_TEST(GetModifiedShape1D, 1DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_1DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 2DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_2DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 3DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_3DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 4DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_4DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 5DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_5DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 6DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_6DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 7DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_7DView<float_type>();
}

TYPED_TEST(GetModifiedShape1D, 8DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape1D_8DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 2DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_2DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 3DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_3DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 4DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_4DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 5DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_5DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 6DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_6DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 7DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_7DView<float_type>();
}

TYPED_TEST(GetModifiedShape2D, 8DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape2D_8DView<float_type>();
}

TYPED_TEST(GetModifiedShape3D, 3DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape3D_3DView<float_type>();
}

TYPED_TEST(GetModifiedShape3D, 4DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape3D_4DView<float_type>();
}

TYPED_TEST(GetModifiedShape3D, 5DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape3D_5DView<float_type>();
}

TYPED_TEST(GetModifiedShape3D, 6DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape3D_6DView<float_type>();
}

TYPED_TEST(GetModifiedShape3D, 7DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape3D_7DView<float_type>();
}

TYPED_TEST(GetModifiedShape3D, 8DView) {
  using float_type = typename TestFixture::float_type;
  test_reshape3D_8DView<float_type>();
}

TEST(IsCropOrPadNeeded, 1DView) {
  const int len = 30, len_pad = 32, len_crop = 28;
  View1D<double> x("x", len);

  EXPECT_FALSE(KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_type<1>{len}));
  EXPECT_TRUE(
      KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_type<1>{len_pad}));
  EXPECT_TRUE(
      KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_type<1>{len_crop}));
}

TEST(IsCropOrPadNeeded, 2DView) {
  const int n0 = 30, n1 = 15;
  View2D<double> x("x", n0, n1);

  for (int i0 = -1; i0 <= 1; i0++) {
    for (int i1 = -1; i1 <= 1; i1++) {
      std::size_t n0_new = static_cast<std::size_t>(n0 + i0);
      std::size_t n1_new = static_cast<std::size_t>(n1 + i1);

      shape_type<2> shape_new = {n0_new, n1_new};
      if (i0 == 0 && i1 == 0) {
        EXPECT_FALSE(KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_new));
      } else {
        EXPECT_TRUE(KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_new));
      }
    }
  }
}

TEST(IsCropOrPadNeeded, 3DView) {
  const int n0 = 30, n1 = 15, n2 = 8;
  View3D<double> x("x", n0, n1, n2);

  for (int i0 = -1; i0 <= 1; i0++) {
    for (int i1 = -1; i1 <= 1; i1++) {
      for (int i2 = -1; i2 <= 1; i2++) {
        std::size_t n0_new = static_cast<std::size_t>(n0 + i0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + i1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + i2);

        shape_type<3> shape_new = {n0_new, n1_new, n2_new};
        if (i0 == 0 && i1 == 0 && i2 == 0) {
          EXPECT_FALSE(KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_new));
        } else {
          EXPECT_TRUE(KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_new));
        }
      }
    }
  }
}

TEST(CropOrPad1D, 1DView) {
  const int len = 30, len_pad = 32, len_crop = 28;

  View1D<double> x("x", len);
  View1D<double> x_out("x_out", len), x_out_pad("x_out_pad", len_pad),
      x_out_crop("x_out_crop", len_crop);
  View1D<double> ref_x("ref_x", len), ref_x_pad("ref_x_pad", len_pad),
      ref_x_crop("ref_x_crop", len_crop);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_x, x);

  // Copying the first len elements, others are initialized with zeros
  auto range0        = std::pair<int, int>(0, len);
  auto sub_ref_x_pad = Kokkos::subview(ref_x_pad, range0);
  Kokkos::deep_copy(sub_ref_x_pad, x);

  // Copying the cropped part
  auto range1 = std::pair<int, int>(0, len_crop);
  auto sub_x  = Kokkos::subview(x, range1);
  Kokkos::deep_copy(ref_x_crop, sub_x);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_pad);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_crop);

  EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_pad, ref_x_pad, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_crop, ref_x_crop, 1.e-5, 1.e-12));
}

TEST(CropOrPad1D, 2DView) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;

  View2D<double> x("x", n0, n1);
  View2D<double> x_out("x_out", n0, n1),
      x_out_pad_axis0("x_out_pad_axis0", n0_pad, n1),
      x_out_pad_axis1("x_out_pad_axis1", n0, n1_pad),
      x_out_crop_axis0("x_out_crop_axis0", n0_crop, n1),
      x_out_crop_axis1("x_out_crop_axis1", n0, n1_crop);
  View2D<double> ref_x("ref_x", n0, n1),
      ref_x_pad_axis0("ref_x_pad_axis0", n0_pad, n1),
      ref_x_crop_axis0("ref_x_crop_axis0", n0_crop, n1);
  View2D<double> ref_x_pad_axis1("ref_x_pad_axis1", n0, n1_pad),
      ref_x_crop_axis1("ref_x_crop_axis1", n0, n1_crop);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_x, x);

  auto h_x                = Kokkos::create_mirror_view(x);
  auto h_ref_x_pad_axis0  = Kokkos::create_mirror_view(ref_x_pad_axis0);
  auto h_ref_x_crop_axis0 = Kokkos::create_mirror_view(ref_x_crop_axis0);
  auto h_ref_x_pad_axis1  = Kokkos::create_mirror_view(ref_x_pad_axis1);
  auto h_ref_x_crop_axis1 = Kokkos::create_mirror_view(ref_x_crop_axis1);
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_ref_x_pad_axis0, ref_x_pad_axis0);
  Kokkos::deep_copy(h_ref_x_crop_axis0, ref_x_crop_axis0);
  Kokkos::deep_copy(h_ref_x_pad_axis1, ref_x_pad_axis1);
  Kokkos::deep_copy(h_ref_x_crop_axis1, ref_x_crop_axis1);

  // Pad or crop along axis 0
  for (int i1 = 0; i1 < n1; i1++) {
    // Copying the first n0 elements, others are initialized with zeros
    for (int i0 = 0; i0 < n0; i0++) {
      h_ref_x_pad_axis0(i0, i1) = h_x(i0, i1);
    }

    // Copying the cropped part
    for (int i0 = 0; i0 < n0_crop; i0++) {
      h_ref_x_crop_axis0(i0, i1) = h_x(i0, i1);
    }
  }

  // Pad or crop along axis 1
  for (int i0 = 0; i0 < n0; i0++) {
    // Copying the first n0 elements, others are initialized with zeros
    for (int i1 = 0; i1 < n1; i1++) {
      h_ref_x_pad_axis1(i0, i1) = h_x(i0, i1);
    }

    // Copying the cropped part
    for (int i1 = 0; i1 < n1_crop; i1++) {
      h_ref_x_crop_axis1(i0, i1) = h_x(i0, i1);
    }
  }
  Kokkos::deep_copy(ref_x_pad_axis0, h_ref_x_pad_axis0);
  Kokkos::deep_copy(ref_x_crop_axis0, h_ref_x_crop_axis0);
  Kokkos::deep_copy(ref_x_pad_axis1, h_ref_x_pad_axis1);
  Kokkos::deep_copy(ref_x_crop_axis1, h_ref_x_crop_axis1);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_pad_axis0);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_crop_axis0);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_pad_axis1);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_crop_axis1);

  EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_pad_axis0, ref_x_pad_axis0,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_crop_axis0, ref_x_crop_axis0,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_pad_axis1, ref_x_pad_axis1,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_crop_axis1, ref_x_crop_axis1,
                       1.e-5, 1.e-12));
}

TEST(CropOrPad1D, 3DView) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;
  const int n2 = 8, n2_pad = 10, n2_crop = 6;

  View3D<double> x("x", n0, n1, n2);
  View3D<double> x_out("x_out", n0, n1, n2),
      x_out_pad_axis0("x_out_pad_axis0", n0_pad, n1, n2),
      x_out_crop_axis0("x_out_crop_axis0", n0_crop, n1, n2),
      x_out_pad_axis1("x_out_pad_axis1", n0, n1_pad, n2),
      x_out_crop_axis1("x_out_crop_axis1", n0, n1_crop, n2),
      x_out_pad_axis2("x_out_pad_axis2", n0, n1, n2_pad),
      x_out_crop_axis2("x_out_crop_axis2", n0, n1, n2_crop);
  View3D<double> ref_x("ref_x", n0, n1, n2),
      ref_x_pad_axis0("ref_x_pad_axis0", n0_pad, n1, n2),
      ref_x_crop_axis0("ref_x_crop_axis0", n0_crop, n1, n2);
  View3D<double> ref_x_pad_axis1("ref_x_pad_axis1", n0, n1_pad, n2),
      ref_x_crop_axis1("ref_x_cro_axis1", n0, n1_crop, n2);
  View3D<double> ref_x_pad_axis2("ref_x_pad_axis2", n0, n1, n2_pad),
      ref_x_crop_axis2("ref_x_cro_axis2", n0, n1, n2_crop);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_x, x);

  auto h_x                = Kokkos::create_mirror_view(x);
  auto h_ref_x_pad_axis0  = Kokkos::create_mirror_view(ref_x_pad_axis0);
  auto h_ref_x_crop_axis0 = Kokkos::create_mirror_view(ref_x_crop_axis0);
  auto h_ref_x_pad_axis1  = Kokkos::create_mirror_view(ref_x_pad_axis1);
  auto h_ref_x_crop_axis1 = Kokkos::create_mirror_view(ref_x_crop_axis1);
  auto h_ref_x_pad_axis2  = Kokkos::create_mirror_view(ref_x_pad_axis2);
  auto h_ref_x_crop_axis2 = Kokkos::create_mirror_view(ref_x_crop_axis2);
  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_ref_x_pad_axis0, ref_x_pad_axis0);
  Kokkos::deep_copy(h_ref_x_crop_axis0, ref_x_crop_axis0);
  Kokkos::deep_copy(h_ref_x_pad_axis1, ref_x_pad_axis1);
  Kokkos::deep_copy(h_ref_x_crop_axis1, ref_x_crop_axis1);
  Kokkos::deep_copy(h_ref_x_pad_axis2, ref_x_pad_axis2);
  Kokkos::deep_copy(h_ref_x_crop_axis2, ref_x_crop_axis2);

  // Pad or crop along axis 0
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      // Copying the first n0 elements, others are initialized with zeros
      for (int i0 = 0; i0 < n0; i0++) {
        h_ref_x_pad_axis0(i0, i1, i2) = h_x(i0, i1, i2);
      }

      // Copying the cropped part
      for (int i0 = 0; i0 < n0_crop; i0++) {
        h_ref_x_crop_axis0(i0, i1, i2) = h_x(i0, i1, i2);
      }
    }
  }

  // Pad or crop along axis 1
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i0 = 0; i0 < n0; i0++) {
      // Copying the first n0 elements, others are initialized with zeros
      for (int i1 = 0; i1 < n1; i1++) {
        h_ref_x_pad_axis1(i0, i1, i2) = h_x(i0, i1, i2);
      }

      // Copying the cropped part
      for (int i1 = 0; i1 < n1_crop; i1++) {
        h_ref_x_crop_axis1(i0, i1, i2) = h_x(i0, i1, i2);
      }
    }
  }

  // Pad or crop along axis 2
  for (int i1 = 0; i1 < n1; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      // Copying the first n0 elements, others are initialized with zeros
      for (int i2 = 0; i2 < n2; i2++) {
        h_ref_x_pad_axis2(i0, i1, i2) = h_x(i0, i1, i2);
      }

      // Copying the cropped part
      for (int i2 = 0; i2 < n2_crop; i2++) {
        h_ref_x_crop_axis2(i0, i1, i2) = h_x(i0, i1, i2);
      }
    }
  }

  Kokkos::deep_copy(ref_x_pad_axis0, h_ref_x_pad_axis0);
  Kokkos::deep_copy(ref_x_crop_axis0, h_ref_x_crop_axis0);
  Kokkos::deep_copy(ref_x_pad_axis1, h_ref_x_pad_axis1);
  Kokkos::deep_copy(ref_x_crop_axis1, h_ref_x_crop_axis1);
  Kokkos::deep_copy(ref_x_pad_axis2, h_ref_x_pad_axis2);
  Kokkos::deep_copy(ref_x_crop_axis2, h_ref_x_crop_axis2);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_pad_axis0);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_crop_axis0);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_pad_axis1);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_crop_axis1);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_pad_axis2);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_crop_axis2);

  EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_pad_axis0, ref_x_pad_axis0,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_crop_axis0, ref_x_crop_axis0,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_pad_axis1, ref_x_pad_axis1,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_crop_axis1, ref_x_crop_axis1,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_pad_axis2, ref_x_pad_axis2,
                       1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(execution_space(), x_out_crop_axis2, ref_x_crop_axis2,
                       1.e-5, 1.e-12));
}

TEST(CropOrPad2D, 2DView) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;

  View2D<double> x("x", n0, n1);
  View2D<double> x_out("x_out", n0, n1), x_out_0_1p("x_out_0_1p", n0, n1_pad),
      x_out_0_1c("x_out_0_1c", n0, n1_crop),
      x_out_0p_1("x_out_0p_1", n0_pad, n1),
      x_out_0p_1p("x_out_0p_1p", n0_pad, n1_pad),
      x_out_0p_1c("x_out_0p_1c", n0_pad, n1_crop),
      x_out_0c_1("x_out_0c_1", n0_crop, n1),
      x_out_0c_1p("x_out_0c_1p", n0_crop, n1_pad),
      x_out_0c_1c("x_out_0c_1c", n0_crop, n1_crop);
  View2D<double> ref_x("ref_x", n0, n1), ref_x_0_1p("ref_x_0_1p", n0, n1_pad),
      ref_x_0_1c("ref_x_0_1c", n0, n1_crop);
  View2D<double> ref_x_0p_1("ref_x_0p_1", n0_pad, n1),
      ref_x_0p_1p("ref_x_0p_1p", n0_pad, n1_pad),
      ref_x_0p_1c("ref_x_0p_1c", n0_pad, n1_crop);
  View2D<double> ref_x_0c_1("ref_x_0c_1", n0_crop, n1),
      ref_x_0c_1p("ref_x_0c_1p", n0_crop, n1_pad),
      ref_x_0c_1c("ref_x_0c_1c", n0_crop, n1_crop);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_x, x);

  auto h_x          = Kokkos::create_mirror_view(x);
  auto h_ref_x_0_1p = Kokkos::create_mirror_view(ref_x_0_1p);
  auto h_ref_x_0_1c = Kokkos::create_mirror_view(ref_x_0_1c);

  auto h_ref_x_0p_1  = Kokkos::create_mirror_view(ref_x_0p_1);
  auto h_ref_x_0p_1p = Kokkos::create_mirror_view(ref_x_0p_1p);
  auto h_ref_x_0p_1c = Kokkos::create_mirror_view(ref_x_0p_1c);
  auto h_ref_x_0c_1  = Kokkos::create_mirror_view(ref_x_0c_1);
  auto h_ref_x_0c_1p = Kokkos::create_mirror_view(ref_x_0c_1p);
  auto h_ref_x_0c_1c = Kokkos::create_mirror_view(ref_x_0c_1c);

  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_ref_x_0_1p, ref_x_0_1p);
  Kokkos::deep_copy(h_ref_x_0_1c, ref_x_0_1c);
  Kokkos::deep_copy(h_ref_x_0p_1, ref_x_0p_1);
  Kokkos::deep_copy(h_ref_x_0p_1p, ref_x_0p_1p);
  Kokkos::deep_copy(h_ref_x_0p_1c, ref_x_0p_1c);
  Kokkos::deep_copy(h_ref_x_0c_1, ref_x_0c_1);
  Kokkos::deep_copy(h_ref_x_0c_1p, ref_x_0c_1p);
  Kokkos::deep_copy(h_ref_x_0c_1c, ref_x_0c_1c);

  // Along axis 0
  for (int i1 = 0; i1 < n1; i1++) {
    // Copying the first n0 elements, others are initialized with zeros
    for (int i0 = 0; i0 < n0; i0++) {
      h_ref_x_0_1p(i0, i1)  = h_x(i0, i1);
      h_ref_x_0p_1(i0, i1)  = h_x(i0, i1);
      h_ref_x_0p_1p(i0, i1) = h_x(i0, i1);
    }

    // Copying the cropped part
    for (int i0 = 0; i0 < n0_crop; i0++) {
      h_ref_x_0c_1(i0, i1)  = h_x(i0, i1);
      h_ref_x_0c_1p(i0, i1) = h_x(i0, i1);
    }
  }

  // Crop Along axis 1
  for (int i1 = 0; i1 < n1_crop; i1++) {
    for (int i0 = 0; i0 < n0; i0++) {
      h_ref_x_0_1c(i0, i1)  = h_x(i0, i1);
      h_ref_x_0p_1c(i0, i1) = h_x(i0, i1);
    }

    // Copying the cropped part
    for (int i0 = 0; i0 < n0_crop; i0++) {
      h_ref_x_0c_1c(i0, i1) = h_x(i0, i1);
    }
  }

  Kokkos::deep_copy(ref_x_0_1p, h_ref_x_0_1p);
  Kokkos::deep_copy(ref_x_0_1c, h_ref_x_0_1c);
  Kokkos::deep_copy(ref_x_0p_1, h_ref_x_0p_1);
  Kokkos::deep_copy(ref_x_0p_1p, h_ref_x_0p_1p);
  Kokkos::deep_copy(ref_x_0p_1c, h_ref_x_0p_1c);
  Kokkos::deep_copy(ref_x_0c_1, h_ref_x_0c_1);
  Kokkos::deep_copy(ref_x_0c_1p, h_ref_x_0c_1p);
  Kokkos::deep_copy(ref_x_0c_1c, h_ref_x_0c_1c);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0_1p);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0_1c);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0p_1);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0p_1p);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0p_1c);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0c_1);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0c_1p);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0c_1c);

  EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0_1p, ref_x_0_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0_1c, ref_x_0_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0p_1, ref_x_0p_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0p_1p, ref_x_0p_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0p_1c, ref_x_0p_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0c_1, ref_x_0c_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0c_1p, ref_x_0c_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0c_1c, ref_x_0c_1c, 1.e-5, 1.e-12));
}

TEST(CropOrPad2D, 3DView) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;
  const int n2 = 8;

  View3D<double> x("x", n0, n1, n2);
  View3D<double> x_out("x_out", n0, n1, n2),
      x_out_0_1p("x_out_0_1p", n0, n1_pad, n2),
      x_out_0_1c("x_out_0_1c", n0, n1_crop, n2),
      x_out_0p_1("x_out_0p_1", n0_pad, n1, n2),
      x_out_0p_1p("x_out_0p_1p", n0_pad, n1_pad, n2),
      x_out_0p_1c("x_out_0p_1c", n0_pad, n1_crop, n2),
      x_out_0c_1("x_out_0c_1", n0_crop, n1, n2),
      x_out_0c_1p("x_out_0c_1p", n0_crop, n1_pad, n2),
      x_out_0c_1c("x_out_0c_1c", n0_crop, n1_crop, n2);
  View3D<double> ref_x("ref_x", n0, n1, n2),
      ref_x_0_1p("ref_x_0_1p", n0, n1_pad, n2),
      ref_x_0_1c("ref_x_0_1c", n0, n1_crop, n2);
  View3D<double> ref_x_0p_1("ref_x_0p_1", n0_pad, n1, n2),
      ref_x_0p_1p("ref_x_0p_1p", n0_pad, n1_pad, n2),
      ref_x_0p_1c("ref_x_0p_1c", n0_pad, n1_crop, n2);
  View3D<double> ref_x_0c_1("ref_x_0c_1", n0_crop, n1, n2),
      ref_x_0c_1p("ref_x_0c_1p", n0_crop, n1_pad, n2),
      ref_x_0c_1c("ref_x_0c_1c", n0_crop, n1_crop, n2);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  Kokkos::deep_copy(ref_x, x);

  auto h_x          = Kokkos::create_mirror_view(x);
  auto h_ref_x_0_1p = Kokkos::create_mirror_view(ref_x_0_1p);
  auto h_ref_x_0_1c = Kokkos::create_mirror_view(ref_x_0_1c);

  auto h_ref_x_0p_1  = Kokkos::create_mirror_view(ref_x_0p_1);
  auto h_ref_x_0p_1p = Kokkos::create_mirror_view(ref_x_0p_1p);
  auto h_ref_x_0p_1c = Kokkos::create_mirror_view(ref_x_0p_1c);
  auto h_ref_x_0c_1  = Kokkos::create_mirror_view(ref_x_0c_1);
  auto h_ref_x_0c_1p = Kokkos::create_mirror_view(ref_x_0c_1p);
  auto h_ref_x_0c_1c = Kokkos::create_mirror_view(ref_x_0c_1c);

  Kokkos::deep_copy(h_x, x);
  Kokkos::deep_copy(h_ref_x_0_1p, ref_x_0_1p);
  Kokkos::deep_copy(h_ref_x_0_1c, ref_x_0_1c);
  Kokkos::deep_copy(h_ref_x_0p_1, ref_x_0p_1);
  Kokkos::deep_copy(h_ref_x_0p_1p, ref_x_0p_1p);
  Kokkos::deep_copy(h_ref_x_0p_1c, ref_x_0p_1c);
  Kokkos::deep_copy(h_ref_x_0c_1, ref_x_0c_1);
  Kokkos::deep_copy(h_ref_x_0c_1p, ref_x_0c_1p);
  Kokkos::deep_copy(h_ref_x_0c_1c, ref_x_0c_1c);

  // Along axis 0
  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      // Copying the first n0 elements, others are initialized with zeros
      for (int i0 = 0; i0 < n0; i0++) {
        h_ref_x_0_1p(i0, i1, i2)  = h_x(i0, i1, i2);
        h_ref_x_0p_1(i0, i1, i2)  = h_x(i0, i1, i2);
        h_ref_x_0p_1p(i0, i1, i2) = h_x(i0, i1, i2);
      }

      // Copying the cropped part
      for (int i0 = 0; i0 < n0_crop; i0++) {
        h_ref_x_0c_1(i0, i1, i2)  = h_x(i0, i1, i2);
        h_ref_x_0c_1p(i0, i1, i2) = h_x(i0, i1, i2);
      }
    }

    // Crop Along axis 1
    for (int i1 = 0; i1 < n1_crop; i1++) {
      for (int i0 = 0; i0 < n0; i0++) {
        h_ref_x_0_1c(i0, i1, i2)  = h_x(i0, i1, i2);
        h_ref_x_0p_1c(i0, i1, i2) = h_x(i0, i1, i2);
      }

      // Copying the cropped part
      for (int i0 = 0; i0 < n0_crop; i0++) {
        h_ref_x_0c_1c(i0, i1, i2) = h_x(i0, i1, i2);
      }
    }
  }

  Kokkos::deep_copy(ref_x_0_1p, h_ref_x_0_1p);
  Kokkos::deep_copy(ref_x_0_1c, h_ref_x_0_1c);
  Kokkos::deep_copy(ref_x_0p_1, h_ref_x_0p_1);
  Kokkos::deep_copy(ref_x_0p_1p, h_ref_x_0p_1p);
  Kokkos::deep_copy(ref_x_0p_1c, h_ref_x_0p_1c);
  Kokkos::deep_copy(ref_x_0c_1, h_ref_x_0c_1);
  Kokkos::deep_copy(ref_x_0c_1p, h_ref_x_0c_1p);
  Kokkos::deep_copy(ref_x_0c_1c, h_ref_x_0c_1c);

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0_1p);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0_1c);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0p_1);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0p_1p);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0p_1c);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0c_1);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0c_1p);
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out_0c_1c);

  EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0_1p, ref_x_0_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0_1c, ref_x_0_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0p_1, ref_x_0p_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0p_1p, ref_x_0p_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0p_1c, ref_x_0p_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0c_1, ref_x_0c_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0c_1p, ref_x_0c_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(
      allclose(execution_space(), x_out_0c_1c, ref_x_0c_1c, 1.e-5, 1.e-12));
}

TEST(CropOrPad3D, 3DView) {
  const int n0 = 30, n1 = 15, n2 = 8;
  View3D<double> x("x", n0, n1, n2);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int d0 = -1; d0 <= 1; d0++) {
    for (int d1 = -1; d1 <= 1; d1++) {
      for (int d2 = -1; d2 <= 1; d2++) {
        std::size_t n0_new = static_cast<std::size_t>(n0 + d0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + d1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + d2);

        View3D<double> x_out("x_out", n0_new, n1_new, n2_new);
        View3D<double> ref_x("ref_x", n0_new, n1_new, n2_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        for (int i2 = 0; i2 < n2; i2++) {
          for (int i1 = 0; i1 < n1; i1++) {
            for (int i0 = 0; i0 < n0; i0++) {
              if (static_cast<std::size_t>(i0) >= h_ref_x.extent(0) ||
                  static_cast<std::size_t>(i1) >= h_ref_x.extent(1) ||
                  static_cast<std::size_t>(i2) >= h_ref_x.extent(2))
                continue;
              h_ref_x(i0, i1, i2) = h_x(i0, i1, i2);
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
        EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}

TEST(CropOrPad4D, 4DView) {
  const int n0 = 30, n1 = 15, n2 = 8, n3 = 7;
  View4D<double> x("x", n0, n1, n2, n3);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto rand_engine = std::mt19937(0);
  auto rand_dist   = std::uniform_int_distribution<int>(-1, 1);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int d0 = -1; d0 <= 1; d0++) {
    for (int d1 = -1; d1 <= 1; d1++) {
      for (int d2 = -1; d2 <= 1; d2++) {
        int d3             = rand_dist(rand_engine);
        std::size_t n0_new = static_cast<std::size_t>(n0 + d0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + d1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + d2);
        std::size_t n3_new = static_cast<std::size_t>(n3 + d3);

        View4D<double> x_out("x_out", n0_new, n1_new, n2_new, n3_new);
        View4D<double> ref_x("ref_x", n0_new, n1_new, n2_new, n3_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        // TODO: stop loop early
        for (int i3 = 0; i3 < n3; i3++) {
          for (int i2 = 0; i2 < n2; i2++) {
            for (int i1 = 0; i1 < n1; i1++) {
              for (int i0 = 0; i0 < n0; i0++) {
                if (static_cast<std::size_t>(i0) >= h_ref_x.extent(0) ||
                    static_cast<std::size_t>(i1) >= h_ref_x.extent(1) ||
                    static_cast<std::size_t>(i2) >= h_ref_x.extent(2) ||
                    static_cast<std::size_t>(i3) >= h_ref_x.extent(3))
                  continue;
                h_ref_x(i0, i1, i2, i3) = h_x(i0, i1, i2, i3);
              }
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
        EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}

TEST(CropOrPad5D, 5DView) {
  const int n0 = 30, n1 = 15, n2 = 8, n3 = 7, n4 = 3;
  View5D<double> x("x", n0, n1, n2, n3, n4);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto rand_engine = std::mt19937(0);
  auto rand_dist   = std::uniform_int_distribution<int>(-1, 1);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int d0 = -1; d0 <= 1; d0++) {
    for (int d1 = -1; d1 <= 1; d1++) {
      for (int d2 = -1; d2 <= 1; d2++) {
        int d3             = rand_dist(rand_engine);
        int d4             = rand_dist(rand_engine);
        std::size_t n0_new = static_cast<std::size_t>(n0 + d0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + d1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + d2);
        std::size_t n3_new = static_cast<std::size_t>(n3 + d3);
        std::size_t n4_new = static_cast<std::size_t>(n4 + d4);

        View5D<double> x_out("x_out", n0_new, n1_new, n2_new, n3_new, n4_new);
        View5D<double> ref_x("ref_x", n0_new, n1_new, n2_new, n3_new, n4_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        for (int i4 = 0; i4 < n4; i4++) {
          for (int i3 = 0; i3 < n3; i3++) {
            for (int i2 = 0; i2 < n2; i2++) {
              for (int i1 = 0; i1 < n1; i1++) {
                for (int i0 = 0; i0 < n0; i0++) {
                  if (static_cast<std::size_t>(i0) >= h_ref_x.extent(0) ||
                      static_cast<std::size_t>(i1) >= h_ref_x.extent(1) ||
                      static_cast<std::size_t>(i2) >= h_ref_x.extent(2) ||
                      static_cast<std::size_t>(i3) >= h_ref_x.extent(3) ||
                      static_cast<std::size_t>(i4) >= h_ref_x.extent(4))
                    continue;
                  h_ref_x(i0, i1, i2, i3, i4) = h_x(i0, i1, i2, i3, i4);
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
        EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}

TEST(CropOrPad6D, 6DView) {
  const int n0 = 10, n1 = 15, n2 = 8, n3 = 7, n4 = 3, n5 = 4;
  View6D<double> x("x", n0, n1, n2, n3, n4, n5);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto rand_engine = std::mt19937(0);
  auto rand_dist   = std::uniform_int_distribution<int>(-1, 1);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int d0 = -1; d0 <= 1; d0++) {
    for (int d1 = -1; d1 <= 1; d1++) {
      for (int d2 = -1; d2 <= 1; d2++) {
        int d3             = rand_dist(rand_engine);
        int d4             = rand_dist(rand_engine);
        int d5             = rand_dist(rand_engine);
        std::size_t n0_new = static_cast<std::size_t>(n0 + d0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + d1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + d2);
        std::size_t n3_new = static_cast<std::size_t>(n3 + d3);
        std::size_t n4_new = static_cast<std::size_t>(n4 + d4);
        std::size_t n5_new = static_cast<std::size_t>(n5 + d5);

        View6D<double> x_out("x_out", n0_new, n1_new, n2_new, n3_new, n4_new,
                             n5_new);
        View6D<double> ref_x("ref_x", n0_new, n1_new, n2_new, n3_new, n4_new,
                             n5_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        for (int i5 = 0; i5 < n5; i5++) {
          for (int i4 = 0; i4 < n4; i4++) {
            for (int i3 = 0; i3 < n3; i3++) {
              for (int i2 = 0; i2 < n2; i2++) {
                for (int i1 = 0; i1 < n1; i1++) {
                  for (int i0 = 0; i0 < n0; i0++) {
                    if (static_cast<std::size_t>(i0) >= h_ref_x.extent(0) ||
                        static_cast<std::size_t>(i1) >= h_ref_x.extent(1) ||
                        static_cast<std::size_t>(i2) >= h_ref_x.extent(2) ||
                        static_cast<std::size_t>(i3) >= h_ref_x.extent(3) ||
                        static_cast<std::size_t>(i4) >= h_ref_x.extent(4) ||
                        static_cast<std::size_t>(i5) >= h_ref_x.extent(5))
                      continue;
                    h_ref_x(i0, i1, i2, i3, i4, i5) =
                        h_x(i0, i1, i2, i3, i4, i5);
                  }
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
        EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}

TEST(CropOrPad7D, 7DView) {
  const int n0 = 10, n1 = 15, n2 = 8, n3 = 7, n4 = 3, n5 = 4, n6 = 5;
  View7D<double> x("x", n0, n1, n2, n3, n4, n5, n6);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto rand_engine = std::mt19937(0);
  auto rand_dist   = std::uniform_int_distribution<int>(-1, 1);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int d0 = -1; d0 <= 1; d0++) {
    for (int d1 = -1; d1 <= 1; d1++) {
      for (int d2 = -1; d2 <= 1; d2++) {
        int d3             = rand_dist(rand_engine);
        int d4             = rand_dist(rand_engine);
        int d5             = rand_dist(rand_engine);
        int d6             = rand_dist(rand_engine);
        std::size_t n0_new = static_cast<std::size_t>(n0 + d0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + d1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + d2);
        std::size_t n3_new = static_cast<std::size_t>(n3 + d3);
        std::size_t n4_new = static_cast<std::size_t>(n4 + d4);
        std::size_t n5_new = static_cast<std::size_t>(n5 + d5);
        std::size_t n6_new = static_cast<std::size_t>(n6 + d6);

        View7D<double> x_out("x_out", n0_new, n1_new, n2_new, n3_new, n4_new,
                             n5_new, n6_new);
        View7D<double> ref_x("ref_x", n0_new, n1_new, n2_new, n3_new, n4_new,
                             n5_new, n6_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        for (int i6 = 0; i6 < n6; i6++) {
          for (int i5 = 0; i5 < n5; i5++) {
            for (int i4 = 0; i4 < n4; i4++) {
              for (int i3 = 0; i3 < n3; i3++) {
                for (int i2 = 0; i2 < n2; i2++) {
                  for (int i1 = 0; i1 < n1; i1++) {
                    for (int i0 = 0; i0 < n0; i0++) {
                      if (static_cast<std::size_t>(i0) >= h_ref_x.extent(0) ||
                          static_cast<std::size_t>(i1) >= h_ref_x.extent(1) ||
                          static_cast<std::size_t>(i2) >= h_ref_x.extent(2) ||
                          static_cast<std::size_t>(i3) >= h_ref_x.extent(3) ||
                          static_cast<std::size_t>(i4) >= h_ref_x.extent(4) ||
                          static_cast<std::size_t>(i5) >= h_ref_x.extent(5) ||
                          static_cast<std::size_t>(i6) >= h_ref_x.extent(6))
                        continue;
                      h_ref_x(i0, i1, i2, i3, i4, i5, i6) =
                          h_x(i0, i1, i2, i3, i4, i5, i6);
                    }
                  }
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
        EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}

TEST(CropOrPad8D, 8DView) {
  const int n0 = 10, n1 = 15, n2 = 8, n3 = 7, n4 = 3, n5 = 4, n6 = 5, n7 = 3;
  View8D<double> x("x", n0, n1, n2, n3, n4, n5, n6, n7);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto rand_engine = std::mt19937(0);
  auto rand_dist   = std::uniform_int_distribution<int>(-1, 1);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int d0 = -1; d0 <= 1; d0++) {
    for (int d1 = -1; d1 <= 1; d1++) {
      for (int d2 = -1; d2 <= 1; d2++) {
        int d3             = rand_dist(rand_engine);
        int d4             = rand_dist(rand_engine);
        int d5             = rand_dist(rand_engine);
        int d6             = rand_dist(rand_engine);
        int d7             = rand_dist(rand_engine);
        std::size_t n0_new = static_cast<std::size_t>(n0 + d0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + d1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + d2);
        std::size_t n3_new = static_cast<std::size_t>(n3 + d3);
        std::size_t n4_new = static_cast<std::size_t>(n4 + d4);
        std::size_t n5_new = static_cast<std::size_t>(n5 + d5);
        std::size_t n6_new = static_cast<std::size_t>(n6 + d6);
        std::size_t n7_new = static_cast<std::size_t>(n7 + d7);

        View8D<double> x_out("x_out", n0_new, n1_new, n2_new, n3_new, n4_new,
                             n5_new, n6_new, n7_new);
        View8D<double> ref_x("ref_x", n0_new, n1_new, n2_new, n3_new, n4_new,
                             n5_new, n6_new, n7_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        for (int i7 = 0; i7 < n7; i7++) {
          for (int i6 = 0; i6 < n6; i6++) {
            for (int i5 = 0; i5 < n5; i5++) {
              for (int i4 = 0; i4 < n4; i4++) {
                for (int i3 = 0; i3 < n3; i3++) {
                  for (int i2 = 0; i2 < n2; i2++) {
                    for (int i1 = 0; i1 < n1; i1++) {
                      for (int i0 = 0; i0 < n0; i0++) {
                        if (static_cast<std::size_t>(i0) >= h_ref_x.extent(0) ||
                            static_cast<std::size_t>(i1) >= h_ref_x.extent(1) ||
                            static_cast<std::size_t>(i2) >= h_ref_x.extent(2) ||
                            static_cast<std::size_t>(i3) >= h_ref_x.extent(3) ||
                            static_cast<std::size_t>(i4) >= h_ref_x.extent(4) ||
                            static_cast<std::size_t>(i5) >= h_ref_x.extent(5) ||
                            static_cast<std::size_t>(i6) >= h_ref_x.extent(6) ||
                            static_cast<std::size_t>(i7) >= h_ref_x.extent(7))
                          continue;
                        h_ref_x(i0, i1, i2, i3, i4, i5, i6, i7) =
                            h_x(i0, i1, i2, i3, i4, i5, i6, i7);
                      }
                    }
                  }
                }
              }
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, x_out);
        EXPECT_TRUE(allclose(execution_space(), x_out, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}
