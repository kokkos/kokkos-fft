#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include <random>
#include "KokkosFFT_padding.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

template <std::size_t DIM>
using shape_type = std::array<std::size_t, DIM>;

TEST(ModifyShape1D, View1D) {
  const int len = 30, len_pad = 32, len_crop = 28;

  View1D<double> x("x", len);

  auto shape = KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{len});
  auto shape_pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{len_pad});
  auto shape_crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{len_crop});

  shape_type<1> ref_shape      = {len};
  shape_type<1> ref_shape_pad  = {len_pad};
  shape_type<1> ref_shape_crop = {len_crop};

  EXPECT_TRUE(shape == ref_shape);
  EXPECT_TRUE(shape_pad == ref_shape_pad);
  EXPECT_TRUE(shape_crop == ref_shape_crop);
}

TEST(ModifyShape1D, View2D) {
  const int n0 = 30, n0_pad = 32, n0_crop = 28;
  const int n1 = 15;

  View2D<double> x("x", n0, n1);

  auto shape = KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{n0});
  auto shape_pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{n0_pad});
  auto shape_crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{n0_crop});

  shape_type<2> ref_shape      = {n0, n1};
  shape_type<2> ref_shape_pad  = {n0_pad, n1};
  shape_type<2> ref_shape_crop = {n0_crop, n1};

  EXPECT_TRUE(shape == ref_shape);
  EXPECT_TRUE(shape_pad == ref_shape_pad);
  EXPECT_TRUE(shape_crop == ref_shape_crop);
}

TEST(ModifyShape1D, View3D) {
  const int n0 = 30, n0_pad = 32, n0_crop = 28;
  const int n1 = 15, n2 = 8;

  View3D<double> x("x", n0, n1, n2);

  auto shape = KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{n0});
  auto shape_pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{n0_pad});
  auto shape_crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<1>{n0_crop});

  shape_type<3> ref_shape      = {n0, n1, n2};
  shape_type<3> ref_shape_pad  = {n0_pad, n1, n2};
  shape_type<3> ref_shape_crop = {n0_crop, n1, n2};

  EXPECT_TRUE(shape == ref_shape);
  EXPECT_TRUE(shape_pad == ref_shape_pad);
  EXPECT_TRUE(shape_crop == ref_shape_crop);
}

TEST(ModifyShape2D, View2D) {
  const int n0 = 30, n0_pad = 32, n0_crop = 28;
  const int n1 = 15, n1_pad = 16, n1_crop = 14;

  View2D<double> x("x", n0, n1);

  auto shape_n0_n1 =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0, n1});
  auto shape_n0_n1pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0, n1_pad});
  auto shape_n0_n1crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0, n1_crop});
  auto shape_n0pad_n1 =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_pad, n1});
  auto shape_n0pad_n1pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_pad, n1_pad});
  auto shape_n0pad_n1crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_pad, n1_crop});
  auto shape_n0crop_n1 =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_crop, n1});
  auto shape_n0crop_n1pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_crop, n1_pad});
  auto shape_n0crop_n1crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_crop, n1_crop});

  shape_type<2> ref_shape_n0_n1         = {n0, n1};
  shape_type<2> ref_shape_n0_n1pad      = {n0, n1_pad};
  shape_type<2> ref_shape_n0_n1crop     = {n0, n1_crop};
  shape_type<2> ref_shape_n0pad_n1      = {n0_pad, n1};
  shape_type<2> ref_shape_n0pad_n1pad   = {n0_pad, n1_pad};
  shape_type<2> ref_shape_n0pad_n1crop  = {n0_pad, n1_crop};
  shape_type<2> ref_shape_n0crop_n1     = {n0_crop, n1};
  shape_type<2> ref_shape_n0crop_n1pad  = {n0_crop, n1_pad};
  shape_type<2> ref_shape_n0crop_n1crop = {n0_crop, n1_crop};

  EXPECT_TRUE(shape_n0_n1 == ref_shape_n0_n1);
  EXPECT_TRUE(shape_n0_n1pad == ref_shape_n0_n1pad);
  EXPECT_TRUE(shape_n0_n1crop == ref_shape_n0_n1crop);
  EXPECT_TRUE(shape_n0pad_n1 == ref_shape_n0pad_n1);
  EXPECT_TRUE(shape_n0pad_n1pad == ref_shape_n0pad_n1pad);
  EXPECT_TRUE(shape_n0pad_n1crop == ref_shape_n0pad_n1crop);
  EXPECT_TRUE(shape_n0crop_n1 == ref_shape_n0crop_n1);
  EXPECT_TRUE(shape_n0crop_n1pad == ref_shape_n0crop_n1pad);
  EXPECT_TRUE(shape_n0crop_n1crop == ref_shape_n0crop_n1crop);
}

TEST(ModifyShape2D, View3D) {
  const int n0 = 30, n0_pad = 32, n0_crop = 28;
  const int n1 = 15, n1_pad = 16, n1_crop = 14;
  const int n2 = 8;

  View3D<double> x("x", n0, n1, n2);

  auto shape_n0_n1 =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0, n1});
  auto shape_n0_n1pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0, n1_pad});
  auto shape_n0_n1crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0, n1_crop});
  auto shape_n0pad_n1 =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_pad, n1});
  auto shape_n0pad_n1pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_pad, n1_pad});
  auto shape_n0pad_n1crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_pad, n1_crop});
  auto shape_n0crop_n1 =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_crop, n1});
  auto shape_n0crop_n1pad =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_crop, n1_pad});
  auto shape_n0crop_n1crop =
      KokkosFFT::Impl::get_modified_shape(x, shape_type<2>{n0_crop, n1_crop});

  shape_type<3> ref_shape_n0_n1         = {n0, n1, n2};
  shape_type<3> ref_shape_n0_n1pad      = {n0, n1_pad, n2};
  shape_type<3> ref_shape_n0_n1crop     = {n0, n1_crop, n2};
  shape_type<3> ref_shape_n0pad_n1      = {n0_pad, n1, n2};
  shape_type<3> ref_shape_n0pad_n1pad   = {n0_pad, n1_pad, n2};
  shape_type<3> ref_shape_n0pad_n1crop  = {n0_pad, n1_crop, n2};
  shape_type<3> ref_shape_n0crop_n1     = {n0_crop, n1, n2};
  shape_type<3> ref_shape_n0crop_n1pad  = {n0_crop, n1_pad, n2};
  shape_type<3> ref_shape_n0crop_n1crop = {n0_crop, n1_crop, n2};

  EXPECT_TRUE(shape_n0_n1 == ref_shape_n0_n1);
  EXPECT_TRUE(shape_n0_n1pad == ref_shape_n0_n1pad);
  EXPECT_TRUE(shape_n0_n1crop == ref_shape_n0_n1crop);
  EXPECT_TRUE(shape_n0pad_n1 == ref_shape_n0pad_n1);
  EXPECT_TRUE(shape_n0pad_n1pad == ref_shape_n0pad_n1pad);
  EXPECT_TRUE(shape_n0pad_n1crop == ref_shape_n0pad_n1crop);
  EXPECT_TRUE(shape_n0crop_n1 == ref_shape_n0crop_n1);
  EXPECT_TRUE(shape_n0crop_n1pad == ref_shape_n0crop_n1pad);
  EXPECT_TRUE(shape_n0crop_n1crop == ref_shape_n0crop_n1crop);
}

TEST(ModifyShape3D, View3D) {
  const int n0 = 30, n1 = 15, n2 = 8;
  View3D<double> x("x", n0, n1, n2);

  for (int i0 = -1; i0 <= 1; i0++) {
    for (int i1 = -1; i1 <= 1; i1++) {
      for (int i2 = -1; i2 <= 1; i2++) {
        std::size_t n0_new = static_cast<std::size_t>(n0 + i0);
        std::size_t n1_new = static_cast<std::size_t>(n1 + i1);
        std::size_t n2_new = static_cast<std::size_t>(n2 + i2);

        shape_type<3> ref_shape = {n0_new, n1_new, n2_new};
        auto shape = KokkosFFT::Impl::get_modified_shape(x, ref_shape);

        EXPECT_TRUE(shape == ref_shape);
      }
    }
  }
}

TEST(IsCropOrPadNeeded, View1D) {
  const int len = 30, len_pad = 32, len_crop = 28;
  View1D<double> x("x", len);

  EXPECT_FALSE(KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_type<1>{len}));
  EXPECT_TRUE(
      KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_type<1>{len_pad}));
  EXPECT_TRUE(
      KokkosFFT::Impl::is_crop_or_pad_needed(x, shape_type<1>{len_crop}));
}

TEST(IsCropOrPadNeeded, View2D) {
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

TEST(IsCropOrPadNeeded, View3D) {
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

TEST(CropOrPad1D, View1D) {
  const int len = 30, len_pad = 32, len_crop = 28;

  View1D<double> x("x", len);
  View1D<double> _x, _x_pad, _x_crop;
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

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x, shape_type<1>{len});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_pad,
                               shape_type<1>{len_pad});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_crop,
                               shape_type<1>{len_crop});

  EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_pad, ref_x_pad, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_crop, ref_x_crop, 1.e-5, 1.e-12));
}

TEST(CropOrPad1D, View2D) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;

  View2D<double> x("x", n0, n1);
  View2D<double> _x, _x_pad_axis0, _x_pad_axis1, _x_crop_axis0, _x_crop_axis1;
  View2D<double> ref_x("ref_x", n0, n1),
      ref_x_pad_axis0("ref_x_pad_axis0", n0_pad, n1),
      ref_x_crop_axis0("ref_x_crop_axis0", n0_crop, n1);
  View2D<double> ref_x_pad_axis1("ref_x_pad_axis1", n0, n1_pad),
      ref_x_crop_axis1("ref_x_cro_axis1", n0, n1_crop);

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

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x, shape_type<2>{n0, n1});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_pad_axis0,
                               shape_type<2>{n0_pad, n1});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_crop_axis0,
                               shape_type<2>{n0_crop, n1});

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_pad_axis1,
                               shape_type<2>{n0, n1_pad});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_crop_axis1,
                               shape_type<2>{n0, n1_crop});

  EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_pad_axis0, ref_x_pad_axis0, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_crop_axis0, ref_x_crop_axis0, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_pad_axis1, ref_x_pad_axis1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_crop_axis1, ref_x_crop_axis1, 1.e-5, 1.e-12));
}

TEST(CropOrPad1D, View3D) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;
  const int n2 = 8, n2_pad = 10, n2_crop = 6;

  View3D<double> x("x", n0, n1, n2);
  View3D<double> _x, _x_pad_axis0, _x_crop_axis0, _x_pad_axis1, _x_crop_axis1,
      _x_pad_axis2, _x_crop_axis2;
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

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x,
                               shape_type<3>{n0, n1, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_pad_axis0,
                               shape_type<3>{n0_pad, n1, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_crop_axis0,
                               shape_type<3>{n0_crop, n1, n2});

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_pad_axis1,
                               shape_type<3>{n0, n1_pad, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_crop_axis1,
                               shape_type<3>{n0, n1_crop, n2});

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_pad_axis2,
                               shape_type<3>{n0, n1, n2_pad});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_crop_axis2,
                               shape_type<3>{n0, n1, n2_crop});

  EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_pad_axis0, ref_x_pad_axis0, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_crop_axis0, ref_x_crop_axis0, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_pad_axis1, ref_x_pad_axis1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_crop_axis1, ref_x_crop_axis1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_pad_axis2, ref_x_pad_axis2, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_crop_axis2, ref_x_crop_axis2, 1.e-5, 1.e-12));
}

TEST(CropOrPad2D, View2D) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;

  View2D<double> x("x", n0, n1);
  View2D<double> _x, _x_0_1p, _x_0_1c, _x_0p_1, _x_0p_1p, _x_0p_1c, _x_0c_1,
      _x_0c_1p, _x_0c_1c;
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

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x, shape_type<2>{n0, n1});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0_1p,
                               shape_type<2>{n0, n1_pad});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0_1c,
                               shape_type<2>{n0, n1_crop});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0p_1,
                               shape_type<2>{n0_pad, n1});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0p_1p,
                               shape_type<2>{n0_pad, n1_pad});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0p_1c,
                               shape_type<2>{n0_pad, n1_crop});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0c_1,
                               shape_type<2>{n0_crop, n1});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0c_1p,
                               shape_type<2>{n0_crop, n1_pad});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0c_1c,
                               shape_type<2>{n0_crop, n1_crop});

  EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0_1p, ref_x_0_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0_1c, ref_x_0_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0p_1, ref_x_0p_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0p_1p, ref_x_0p_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0p_1c, ref_x_0p_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0c_1, ref_x_0c_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0c_1p, ref_x_0c_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0c_1c, ref_x_0c_1c, 1.e-5, 1.e-12));
}

TEST(CropOrPad2D, View3D) {
  const int n0 = 12, n0_pad = 14, n0_crop = 10;
  const int n1 = 5, n1_pad = 7, n1_crop = 3;
  const int n2 = 8;

  View3D<double> x("x", n0, n1, n2);
  View3D<double> _x, _x_0_1p, _x_0_1c, _x_0p_1, _x_0p_1p, _x_0p_1c, _x_0c_1,
      _x_0c_1p, _x_0c_1c;
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

  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x,
                               shape_type<3>{n0, n1, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0_1p,
                               shape_type<3>{n0, n1_pad, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0_1c,
                               shape_type<3>{n0, n1_crop, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0p_1,
                               shape_type<3>{n0_pad, n1, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0p_1p,
                               shape_type<3>{n0_pad, n1_pad, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0p_1c,
                               shape_type<3>{n0_pad, n1_crop, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0c_1,
                               shape_type<3>{n0_crop, n1, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0c_1p,
                               shape_type<3>{n0_crop, n1_pad, n2});
  KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x_0c_1c,
                               shape_type<3>{n0_crop, n1_crop, n2});

  EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0_1p, ref_x_0_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0_1c, ref_x_0_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0p_1, ref_x_0p_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0p_1p, ref_x_0p_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0p_1c, ref_x_0p_1c, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0c_1, ref_x_0c_1, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0c_1p, ref_x_0c_1p, 1.e-5, 1.e-12));
  EXPECT_TRUE(allclose(_x_0c_1c, ref_x_0c_1c, 1.e-5, 1.e-12));
}

TEST(CropOrPad3D, View3D) {
  const int n0 = 30, n1 = 15, n2 = 8;
  View3D<double> x("x", n0, n1, n2);

  Kokkos::Random_XorShift64_Pool<> random_pool(12345);
  Kokkos::fill_random(x, random_pool, 1.0);

  auto h_x = Kokkos::create_mirror_view(x);
  Kokkos::deep_copy(h_x, x);
  for (int i0 = -1; i0 <= 1; i0++) {
    for (int i1 = -1; i1 <= 1; i1++) {
      for (int i2 = -1; i2 <= 1; i2++) {
        std::size_t n0_new      = static_cast<std::size_t>(n0 + i0);
        std::size_t n1_new      = static_cast<std::size_t>(n1 + i1);
        std::size_t n2_new      = static_cast<std::size_t>(n2 + i2);
        shape_type<3> shape_new = {n0_new, n1_new, n2_new};

        View3D<double> _x;
        View3D<double> ref_x("ref_x", n0_new, n1_new, n2_new);

        auto h_ref_x = Kokkos::create_mirror_view(ref_x);
        for (int i2 = 0; i2 < n2; i2++) {
          for (int i1 = 0; i1 < n1; i1++) {
            for (int i0 = 0; i0 < n0; i0++) {
              if (i0 >= h_ref_x.extent(0) || i1 >= h_ref_x.extent(1) ||
                  i2 >= h_ref_x.extent(2))
                continue;
              h_ref_x(i0, i1, i2) = h_x(i0, i1, i2);
            }
          }
        }

        Kokkos::deep_copy(ref_x, h_ref_x);
        KokkosFFT::Impl::crop_or_pad(execution_space(), x, _x, shape_new);
        EXPECT_TRUE(allclose(_x, ref_x, 1.e-5, 1.e-12));
      }
    }
  }
}