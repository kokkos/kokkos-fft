#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_transpose.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

template <std::size_t DIM>
using axes_type = std::array<int, DIM>;

TEST(MapAxes, 1D) {
    const int len = 30;
    View1D<double> x("x", len);

    auto map_axis = KokkosFFT::get_map_axes(x, /*axis=*/0);
    auto map_axes = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({0}));

    std::vector<int> ref_map_axis(1, 0), ref_map_axes(1, 0);

    EXPECT_TRUE( map_axis == ref_map_axis );
    EXPECT_TRUE( map_axes == ref_map_axes );
}

TEST(MapAxes, 2DLeft) {
    const int n0 = 3, n1 = 5;
    LeftView2D<double> x("x", n0, n1);
    
    auto map_axis_0        = KokkosFFT::get_map_axes(x, /*axis=*/0);
    auto map_axis_1        = KokkosFFT::get_map_axes(x, /*axis=*/1);
    auto map_axis_minus1   = KokkosFFT::get_map_axes(x, /*axis=*/-1);
    auto map_axes_0        = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({0}));
    auto map_axes_1        = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({1}));
    auto map_axes_minus1   = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({-1}));
    auto map_axes_0_minus1 = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({0, -1}));
    auto map_axes_minus1_0 = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({-1, 0}));
    auto map_axes_0_1      = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({0, 1}));
    auto map_axes_1_0      = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({1, 0}));

    std::vector<int> ref_map_axis_0({0, 1});
    std::vector<int> ref_map_axis_1({1, 0});
    std::vector<int> ref_map_axis_minus1({1, 0});
    std::vector<int> ref_map_axes_0({0, 1});
    std::vector<int> ref_map_axes_1({1, 0});
    std::vector<int> ref_map_axes_minus1({1, 0});
    std::vector<int> ref_map_axes_0_minus1({0, 1});
    std::vector<int> ref_map_axes_minus1_0({1, 0});
    std::vector<int> ref_map_axes_0_1({0, 1});
    std::vector<int> ref_map_axes_1_0({1, 0});

    EXPECT_TRUE( map_axis_0 == ref_map_axis_0 );
    EXPECT_TRUE( map_axis_1 == ref_map_axis_1 );
    EXPECT_TRUE( map_axis_minus1 == ref_map_axis_minus1 );
    EXPECT_TRUE( map_axes_0 == ref_map_axes_0 );
    EXPECT_TRUE( map_axes_1 == ref_map_axes_1 );
    EXPECT_TRUE( map_axes_minus1 == ref_map_axes_minus1 );
    EXPECT_TRUE( map_axes_0_minus1 == ref_map_axes_0_minus1 );
    EXPECT_TRUE( map_axes_minus1_0 == ref_map_axes_minus1_0 );
    EXPECT_TRUE( map_axes_0_1 == ref_map_axes_0_1 );
    EXPECT_TRUE( map_axes_1_0 == ref_map_axes_1_0 );
}

TEST(MapAxes, 2DRight) {
    const int n0 = 3, n1 = 5;
    RightView2D<double> x("x", n0, n1);
    
    auto map_axis_0        = KokkosFFT::get_map_axes(x, /*axis=*/0);
    auto map_axis_1        = KokkosFFT::get_map_axes(x, /*axis=*/1);
    auto map_axis_minus1   = KokkosFFT::get_map_axes(x, /*axis=*/-1);
    auto map_axes_0        = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({0}));
    auto map_axes_1        = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({1}));
    auto map_axes_minus1   = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<1>({-1}));
    auto map_axes_0_minus1 = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({0, -1}));
    auto map_axes_minus1_0 = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({-1, 0}));
    auto map_axes_0_1      = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({0, 1}));
    auto map_axes_1_0      = KokkosFFT::get_map_axes(x, /*axes=*/ axes_type<2>({1, 0}));

    std::vector<int> ref_map_axis_0({1, 0});
    std::vector<int> ref_map_axis_1({0, 1});
    std::vector<int> ref_map_axis_minus1({0, 1});
    std::vector<int> ref_map_axes_0({1, 0});
    std::vector<int> ref_map_axes_1({0, 1});
    std::vector<int> ref_map_axes_minus1({0, 1});
    std::vector<int> ref_map_axes_0_minus1({1, 0});
    std::vector<int> ref_map_axes_minus1_0({0, 1});
    std::vector<int> ref_map_axes_0_1({1, 0});
    std::vector<int> ref_map_axes_1_0({0, 1});

    EXPECT_TRUE( map_axis_0 == ref_map_axis_0 );
    EXPECT_TRUE( map_axis_1 == ref_map_axis_1 );
    EXPECT_TRUE( map_axis_minus1 == ref_map_axis_minus1 );
    EXPECT_TRUE( map_axes_0 == ref_map_axes_0 );
    EXPECT_TRUE( map_axes_1 == ref_map_axes_1 );
    EXPECT_TRUE( map_axes_minus1 == ref_map_axes_minus1 );
    EXPECT_TRUE( map_axes_0_minus1 == ref_map_axes_0_minus1 );
    EXPECT_TRUE( map_axes_minus1_0 == ref_map_axes_minus1_0 );
    EXPECT_TRUE( map_axes_0_1 == ref_map_axes_0_1 );
    EXPECT_TRUE( map_axes_1_0 == ref_map_axes_1_0 );
}

/*
TEST(MapAxes, 3DLeft) {
    const int n0 = 3, n1 = 5, n2 = 8;
    LeftView3D<double> x("x", n0, n1, n2);
    
    auto map_axis_0        = KokkosFFT::get_map_axes(x, 0);
    auto map_axis_1        = KokkosFFT::get_map_axes(x, 1);
    auto map_axis_2        = KokkosFFT::get_map_axes(x, 2);
    auto map_axis_minus1   = KokkosFFT::get_map_axes(x, -1);
    auto map_axis_minus2   = KokkosFFT::get_map_axes(x, -2);

    auto map_axes_0        = KokkosFFT::get_map_axes(x, axes_type<1>({0}));
    auto map_axes_1        = KokkosFFT::get_map_axes(x, axes_type<1>({1}));
    auto map_axes_2        = KokkosFFT::get_map_axes(x, axes_type<1>({2}));
    auto map_axes_minus1   = KokkosFFT::get_map_axes(x, axes_type<1>({-1}));
    auto map_axes_minus2   = KokkosFFT::get_map_axes(x, axes_type<1>({-2}));

    auto map_axes_0_1      = KokkosFFT::get_map_axes(x, axes_type<2>({0, 1}));
    auto map_axes_0_2      = KokkosFFT::get_map_axes(x, axes_type<2>({0, 2}));
    auto map_axes_0_minus1 = KokkosFFT::get_map_axes(x, axes_type<2>({0, -1}));
    auto map_axes_0_minus2 = KokkosFFT::get_map_axes(x, axes_type<2>({0, -2}));
    auto map_axes_1_0      = KokkosFFT::get_map_axes(x, axes_type<2>({1, 0}));
    auto map_axes_1_2      = KokkosFFT::get_map_axes(x, axes_type<2>({1, 2}));
    auto map_axes_1_minus1 = KokkosFFT::get_map_axes(x, axes_type<2>({1, -1}));
    auto map_axes_2_0      = KokkosFFT::get_map_axes(x, axes_type<2>({2, 0}));
    auto map_axes_2_1      = KokkosFFT::get_map_axes(x, axes_type<2>({2, 1}));
    auto map_axes_2_minus2 = KokkosFFT::get_map_axes(x, axes_type<2>({2, -2}));
    auto map_axes_minus1_0 = KokkosFFT::get_map_axes(x, axes_type<2>({-1, 0}));
    auto map_axes_minus1_1 = KokkosFFT::get_map_axes(x, axes_type<2>({-1, 1}));
    auto map_axes_minus1_minus2 = KokkosFFT::get_map_axes(x, axes_type<2>({-1, -2}));
    auto map_axes_minus2_0 = KokkosFFT::get_map_axes(x, axes_type<2>({-2, 0}));
    auto map_axes_minus2_2 = KokkosFFT::get_map_axes(x, axes_type<2>({-2, 2}));
    auto map_axes_minus2_minus1 = KokkosFFT::get_map_axes(x, axes_type<2>({-2, -1}));

    auto map_axes_0_1_2           = KokkosFFT::get_map_axes(x, axes_type<3>({0, 1, 2}));
    auto map_axes_0_1_minus1      = KokkosFFT::get_map_axes(x, axes_type<3>({0, 1, -1}));
    auto map_axes_0_2_1           = KokkosFFT::get_map_axes(x, axes_type<3>({0, 2, 1}));
    auto map_axes_0_2_minus2      = KokkosFFT::get_map_axes(x, axes_type<3>({0, 2, -2}));
    auto map_axes_0_minus1_1      = KokkosFFT::get_map_axes(x, axes_type<3>({0, -1, 1}));
    auto map_axes_0_minus1_minus2 = KokkosFFT::get_map_axes(x, axes_type<3>({0, -1, -2}));
    auto map_axes_0_minus2_2      = KokkosFFT::get_map_axes(x, axes_type<3>({0, -2, 2}));
    auto map_axes_0_minus2_minus1 = KokkosFFT::get_map_axes(x, axes_type<3>({0, -2, -1}));
    
    auto map_axes_1_0_2           = KokkosFFT::get_map_axes(x, axes_type<3>({1, 0, 2}));
    auto map_axes_1_0_minus1      = KokkosFFT::get_map_axes(x, axes_type<3>({1, 0, -1}));
    auto map_axes_1_2_0           = KokkosFFT::get_map_axes(x, axes_type<3>({1, 2, 0}));
    auto map_axes_1_minus1_0      = KokkosFFT::get_map_axes(x, axes_type<3>({1, -1, 0}));
    auto map_axes_2_0_1           = KokkosFFT::get_map_axes(x, axes_type<3>({2, 0, 1}));
    auto map_axes_2_0_minus2      = KokkosFFT::get_map_axes(x, axes_type<3>({2, 0, -2}));
    auto map_axes_2_1_0           = KokkosFFT::get_map_axes(x, axes_type<3>({2, 1, 0}));
    auto map_axes_2_minus2_0      = KokkosFFT::get_map_axes(x, axes_type<3>({2, -2, 0}));
    auto map_axes_minus1_0_1      = KokkosFFT::get_map_axes(x, axes_type<3>({-1, 0, 1}));
    auto map_axes_minus1_0_minus2 = KokkosFFT::get_map_axes(x, axes_type<3>({-1, 0, -2}));
    auto map_axes_minus1_1_0      = KokkosFFT::get_map_axes(x, axes_type<3>({-1, 1, 0}));
    auto map_axes_minus1_minus2_0 = KokkosFFT::get_map_axes(x, axes_type<3>({-1, -2, 0}));   
    auto map_axes_minus2_0_2      = KokkosFFT::get_map_axes(x, axes_type<3>({-2, 0, 2}));
    auto map_axes_minus2_0_minus1 = KokkosFFT::get_map_axes(x, axes_type<3>({-2, 0, -1}));
    auto map_axes_minus2_2_0      = KokkosFFT::get_map_axes(x, axes_type<3>({-2, 2, 0}));
    auto map_axes_minus2_minus1_0 = KokkosFFT::get_map_axes(x, axes_type<3>({-2, -1, 0}));   
    
    std::vector<int> ref_map_axis_0({0, 1, 2});
    std::vector<int> ref_map_axis_1({1, 0, 2});
    std::vector<int> ref_map_axis_2({2, 1, 0});
    std::vector<int> ref_map_axis_minus1({2, 1, 0});
    std::vector<int> ref_map_axis_minus2({1, 0, 2});

    std::vector<int> ref_map_axes_0({0, 1, 2});
    std::vector<int> ref_map_axes_1({1, 0, 2});
    std::vector<int> ref_map_axes_2({2, 1, 0});
    std::vector<int> ref_map_axes_minus1({2, 1, 0});
    std::vector<int> ref_map_axes_minus2({1, 0, 2});

    std::vector<int> ref_map_axes_0_1({0, 1, 2});
    std::vector<int> ref_map_axes_0_2({0, 2, 1});
    std::vector<int> ref_map_axes_0_minus1({0, 2, 1});
    std::vector<int> ref_map_axes_0_minus2({0, 1, 2});
    std::vector<int> ref_map_axes_1_0({1, 0, 2});
    std::vector<int> ref_map_axes_1_2({1, 2, 0});
    std::vector<int> ref_map_axes_1_minus1({1, 2, 0});
    std::vector<int> ref_map_axes_2_0({2, 0, 1});
    std::vector<int> ref_map_axes_2_1({2, 1, 0});
    std::vector<int> ref_map_axes_2_minus2({2, 1, 0});
    std::vector<int> ref_map_axes_minus1_0({2, 0, 1});
    std::vector<int> ref_map_axes_minus1_1({2, 1, 0});
    std::vector<int> ref_map_axes_minus1_minus2({2, 1, 0});
    std::vector<int> ref_map_axes_minus2_0({1, 0, 2});
    std::vector<int> ref_map_axes_minus2_2({1, 2, 0});
    std::vector<int> ref_map_axes_minus2_minus1({1, 2, 0});

    std::vector<int> ref_map_axes_0_1_2({0, 1, 2});
    std::vector<int> ref_map_axes_0_1_minus1({0, 1, 2});
    std::vector<int> ref_map_axes_0_2_1({0, 2, 1});
    std::vector<int> ref_map_axes_0_2_minus2({0, 2, 1});
    std::vector<int> ref_map_axes_0_minus1_1({0, 2, 1});
    std::vector<int> ref_map_axes_0_minus1_minus2({0, 2, 1});
    std::vector<int> ref_map_axes_0_minus2_2({0, 1, 2});
    std::vector<int> ref_map_axes_0_minus2_minus1({0, 1, 2});

    std::vector<int> ref_map_axes_1_0_2({1, 0, 2});
    std::vector<int> ref_map_axes_1_0_minus1({0, 1, 2});
    std::vector<int> ref_map_axes_1_2_0({1, 2, 0});
    std::vector<int> ref_map_axes_1_minus1_0({1, 2, 0});
    std::vector<int> ref_map_axes_2_0_1({2, 0, 1});
    std::vector<int> ref_map_axes_2_0_minus2({2, 0, 1});
    std::vector<int> ref_map_axes_2_1_0({2, 1, 0});
    std::vector<int> ref_map_axes_2_minus2_0({2, 1, 0});
    std::vector<int> ref_map_axes_minus1_0_1({2, 0, 1});
    std::vector<int> ref_map_axes_minus1_0_minus2({2, 0, 1});
    std::vector<int> ref_map_axes_minus1_1_0({2, 1, 0});
    std::vector<int> ref_map_axes_minus1_minus2_0({2, 1, 0});
    std::vector<int> ref_map_axes_minus2_0_2({1, 0, 2});
    std::vector<int> ref_map_axes_minus2_0_minus1({1, 0, 2});
    std::vector<int> ref_map_axes_minus2_2_0({1, 2, 0});
    std::vector<int> ref_map_axes_minus2_minus1_0({1, 2, 0});

    EXPECT_TRUE( map_axis_0 == ref_map_axis_0 );
    EXPECT_TRUE( map_axis_1 == ref_map_axis_1 );
    EXPECT_TRUE( map_axis_2 == ref_map_axis_2 );
    EXPECT_TRUE( map_axis_minus1 == ref_map_axis_minus1 );
    EXPECT_TRUE( map_axis_minus2 == ref_map_axis_minus2 );

    EXPECT_TRUE( map_axes_0 == ref_map_axes_0 );
    EXPECT_TRUE( map_axes_1 == ref_map_axes_1 );
    EXPECT_TRUE( map_axes_2 == ref_map_axes_2 );
    EXPECT_TRUE( map_axes_minus1 == ref_map_axes_minus1 );
    EXPECT_TRUE( map_axes_minus2 == ref_map_axes_minus2 );

    EXPECT_TRUE( map_axes_0_1 == ref_map_axes_0_1 );
    EXPECT_TRUE( map_axes_0_2 == ref_map_axes_0_2 );
    EXPECT_TRUE( map_axes_0_minus1 == ref_map_axes_0_minus1 );
    EXPECT_TRUE( map_axes_0_minus2 == ref_map_axes_0_minus2 );
    EXPECT_TRUE( map_axes_1_0 == ref_map_axes_1_0 );
    EXPECT_TRUE( map_axes_1_2 == ref_map_axes_1_2 );
    EXPECT_TRUE( map_axes_1_minus1 == ref_map_axes_1_minus1 );
    EXPECT_TRUE( map_axes_2_0 == ref_map_axes_2_0 );
    EXPECT_TRUE( map_axes_2_1 == ref_map_axes_2_1 );
    EXPECT_TRUE( map_axes_2_minus2 == ref_map_axes_2_minus2 );
    EXPECT_TRUE( map_axes_minus1_0 == ref_map_axes_minus1_0 );
    EXPECT_TRUE( map_axes_minus1_1 == ref_map_axes_minus1_1 );
    EXPECT_TRUE( map_axes_minus1_minus2 == ref_map_axes_minus1_minus2 );
    EXPECT_TRUE( map_axes_minus2_0 == ref_map_axes_minus2_0 );
    EXPECT_TRUE( map_axes_minus2_2 == ref_map_axes_minus2_2 );
    EXPECT_TRUE( map_axes_minus2_minus1 == ref_map_axes_minus2_minus1 );

    EXPECT_TRUE( map_axes_0_1_2 == ref_map_axes_0_1_2 );
    EXPECT_TRUE( map_axes_0_1_minus1 == ref_map_axes_0_1_minus1 );
    EXPECT_TRUE( map_axes_0_2_1 == ref_map_axes_0_2_1 );
    EXPECT_TRUE( map_axes_0_2_minus2 == ref_map_axes_0_2_minus2 );
    EXPECT_TRUE( map_axes_0_minus1_1 == ref_map_axes_0_minus1_1 );
    EXPECT_TRUE( map_axes_0_minus1_minus2 == ref_map_axes_0_minus1_minus2 );
    EXPECT_TRUE( map_axes_0_minus2_2 == ref_map_axes_0_minus2_2 );
    EXPECT_TRUE( map_axes_0_minus2_minus1 == ref_map_axes_0_minus2_minus1 );

    EXPECT_TRUE( map_axes_1_0_2 == ref_map_axes_1_0_2 );
    EXPECT_TRUE( map_axes_1_0_minus1 == ref_map_axes_1_0_minus1 );
    EXPECT_TRUE( map_axes_1_2_0 == ref_map_axes_1_2_0 );
    EXPECT_TRUE( map_axes_1_minus1_0 == ref_map_axes_1_minus1_0 );
    EXPECT_TRUE( map_axes_2_1_0 == ref_map_axes_2_1_0 );
    EXPECT_TRUE( map_axes_2_minus2_0 == ref_map_axes_2_minus2_0 );
    EXPECT_TRUE( map_axes_minus1_0_1 == ref_map_axes_minus1_0_1 );
    EXPECT_TRUE( map_axes_minus1_0_minus2 == ref_map_axes_minus1_0_minus2 );
    EXPECT_TRUE( map_axes_minus1_1_0 == ref_map_axes_minus1_1_0 );
    EXPECT_TRUE( map_axes_minus1_minus2_0 == ref_map_axes_minus1_minus2_0 );
    EXPECT_TRUE( map_axes_minus2_0_2 == ref_map_axes_minus2_0_2 );
    EXPECT_TRUE( map_axes_minus2_0_minus1 == ref_map_axes_minus2_0_minus1 );
    EXPECT_TRUE( map_axes_minus2_2_0 == ref_map_axes_minus2_2_0 );
    EXPECT_TRUE( map_axes_minus2_minus1_0 == ref_map_axes_minus2_minus1_0 );
}
*/

/* For the moment, disabled. Compilation issue on NVIDIA GPUs
TEST(Transpose, 1D) {
    const int len = 30;
    View1D<double> x("x", len), ref("ref", len);
    View1D<double> xt;

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    Kokkos::deep_copy(ref, x);

    Kokkos::fence();
    KokkosFFT::transpose(x, xt);

    EXPECT_TRUE( allclose(xt, ref, 1.e-5, 1.e-12) );
}

TEST(Transpose, 2DLeft) {
    const int n0 = 3, n1 = 5;
    LeftView2D<double> x("x", n0, n1), ref_axis0("ref_axis0", n0, n1);
    LeftView2D<double> xt_axis0, xt_axis1, xt_axis_minus1; // views are allocated internally
    LeftView2D<double> ref_axis1("ref_axis1", n1, n0), ref_axis_minus1("ref_axis_minus1", n1, n0); 

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    Kokkos::deep_copy(ref_axis0, x);

    // Transposed views
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_ref_axis1 = Kokkos::create_mirror_view(ref_axis1);
    Kokkos::deep_copy(h_x, x);

    for(int i0=0; i0<h_x.extent(0); i0++) {
      for(int i1=0; i1<h_x.extent(1); i1++) {
        h_ref_axis1(i1, i0) = h_x(i0, i1);
      }
    }
    Kokkos::deep_copy(ref_axis1, h_ref_axis1);
    Kokkos::deep_copy(ref_axis_minus1, h_ref_axis1);
    Kokkos::fence();

    int axis = 0;
    KokkosFFT::transpose(x, xt_axis0, axis); // xt is identical to x

    EXPECT_TRUE( allclose(xt_axis0, ref_axis0, 1.e-5, 1.e-12) );

    axis = 1;
    KokkosFFT::transpose(x, xt_axis1, axis); // xt is the transpose of x

    EXPECT_TRUE( allclose(xt_axis1, ref_axis1, 1.e-5, 1.e-12) );

    axis = -1;
    KokkosFFT::transpose(x, xt_axis_minus1, axis); // xt is the transpose of x

    EXPECT_TRUE( allclose(xt_axis_minus1, ref_axis_minus1, 1.e-5, 1.e-12) );
}

TEST(Transpose, 2DRight) {
    const int n0 = 3, n1 = 5;
    RightView2D<double> x("x", n0, n1), ref_axis0("ref_axis0", n1, n0);
    RightView2D<double> xt_axis0, xt_axis1, xt_axis_minus1; // views are allocated internally
    RightView2D<double> ref_axis1("ref_axis1", n0, n1), ref_axis_minus1("ref_axis_minus1", n0, n1); 

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    // Identical to x
    Kokkos::deep_copy(ref_axis1, x);
    Kokkos::deep_copy(ref_axis_minus1, x);

    // Transposed views
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_ref_axis0 = Kokkos::create_mirror_view(ref_axis0);
    Kokkos::deep_copy(h_x, x);

    for(int i0=0; i0<h_x.extent(0); i0++) {
      for(int i1=0; i1<h_x.extent(1); i1++) {
        h_ref_axis0(i1, i0) = h_x(i0, i1);
      }
    }
    Kokkos::deep_copy(ref_axis0, h_ref_axis0);
    Kokkos::fence();

    int axis = 0;
    KokkosFFT::transpose(x, xt_axis0, axis); // xt is the transpose of x

    EXPECT_TRUE( allclose(xt_axis0, ref_axis0, 1.e-5, 1.e-12) );

    axis = 1;
    KokkosFFT::transpose(x, xt_axis1, axis); // xt is identical to x

    EXPECT_TRUE( allclose(xt_axis1, ref_axis1, 1.e-5, 1.e-12) );

    axis = -1;
    KokkosFFT::transpose(x, xt_axis_minus1, axis); // xt is identical to x

    EXPECT_TRUE( allclose(xt_axis_minus1, ref_axis_minus1, 1.e-5, 1.e-12) );
}

TEST(Transpose, 3DLeft) {
    const int n0 = 3, n1 = 5, n2 = 8;
    LeftView3D<double> x("x", n0, n1, n2);
    LeftView3D<double> xt_axis0, xt_axis1, xt_axis2, xt_axis_minus1, xt_axis_minus2; // views are allocated internally
    LeftView3D<double> ref_axis0("ref_axis0", n0, n1, n2), ref_axis1("ref_axis1", n1, n0, n2), ref_axis2("ref_axis2", n2, n1, n0);
    LeftView3D<double> ref_axis_minus1("ref_axis_minus1", n2, n1, n0), ref_axis_minus2("ref_axis_minus2", n1, n0, n2); 

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    Kokkos::deep_copy(ref_axis0, x);

    // Transposed views
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_ref_axis1 = Kokkos::create_mirror_view(ref_axis1);
    auto h_ref_axis2 = Kokkos::create_mirror_view(ref_axis2);
    Kokkos::deep_copy(h_x, x);

    for(int i0=0; i0<h_x.extent(0); i0++) {
      for(int i1=0; i1<h_x.extent(1); i1++) {
        for(int i2=0; i2<h_x.extent(2); i2++) {
          h_ref_axis1(i1, i0, i2) = h_x(i0, i1, i2);
          h_ref_axis2(i2, i1, i0) = h_x(i0, i1, i2);
        }
      }
    }
    Kokkos::deep_copy(ref_axis1, h_ref_axis1);
    Kokkos::deep_copy(ref_axis_minus2, h_ref_axis1);
    Kokkos::deep_copy(ref_axis2, h_ref_axis2);
    Kokkos::deep_copy(ref_axis_minus1, h_ref_axis2);
    Kokkos::fence();

    int axis = 0;
    KokkosFFT::transpose(x, xt_axis0, axis); // xt is identical to x
    EXPECT_TRUE( allclose(xt_axis0, ref_axis0, 1.e-5, 1.e-12) );

    axis = 1;
    KokkosFFT::transpose(x, xt_axis1, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis1, ref_axis1, 1.e-5, 1.e-12) );

    axis = 2;
    KokkosFFT::transpose(x, xt_axis2, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis2, ref_axis2, 1.e-5, 1.e-12) );

    axis = -1;
    KokkosFFT::transpose(x, xt_axis_minus1, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis_minus1, ref_axis_minus1, 1.e-5, 1.e-12) );

    axis = -2;
    KokkosFFT::transpose(x, xt_axis_minus2, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis_minus2, ref_axis_minus2, 1.e-5, 1.e-12) );
}

TEST(Transpose, 3DRight) {
    const int n0 = 3, n1 = 5, n2 = 8;
    RightView3D<double> x("x", n0, n1, n2);
    RightView3D<double> xt_axis0, xt_axis1, xt_axis2, xt_axis_minus1, xt_axis_minus2; // views are allocated internally
    RightView3D<double> ref_axis0("ref_axis0", n2, n1, n0), ref_axis1("ref_axis1", n0, n2, n1), ref_axis2("ref_axis2", n0, n1, n2);
    RightView3D<double> ref_axis_minus1("ref_axis_minus1", n0, n1, n2), ref_axis_minus2("ref_axis_minus2", n0, n2, n1); 

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    Kokkos::deep_copy(ref_axis2, x);
    Kokkos::deep_copy(ref_axis_minus1, x);

    // Transposed views
    auto h_x = Kokkos::create_mirror_view(x);
    auto h_ref_axis0 = Kokkos::create_mirror_view(ref_axis0);
    auto h_ref_axis1 = Kokkos::create_mirror_view(ref_axis1);
    Kokkos::deep_copy(h_x, x);

    for(int i0=0; i0<h_x.extent(0); i0++) {
      for(int i1=0; i1<h_x.extent(1); i1++) {
        for(int i2=0; i2<h_x.extent(2); i2++) {
          h_ref_axis0(i2, i1, i0) = h_x(i0, i1, i2);
          h_ref_axis1(i0, i2, i1) = h_x(i0, i1, i2);
        }
      }
    }
    Kokkos::deep_copy(ref_axis0, h_ref_axis0);
    Kokkos::deep_copy(ref_axis_minus2, h_ref_axis1);
    Kokkos::deep_copy(ref_axis1, h_ref_axis1);
    Kokkos::fence();

    int axis = 0;
    KokkosFFT::transpose(x, xt_axis0, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis0, ref_axis0, 1.e-5, 1.e-12) );

    axis = 1;
    KokkosFFT::transpose(x, xt_axis1, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis1, ref_axis1, 1.e-5, 1.e-12) );
    
    axis = 2;
    KokkosFFT::transpose(x, xt_axis2, axis); // xt is identical to x
    EXPECT_TRUE( allclose(xt_axis2, ref_axis2, 1.e-5, 1.e-12) );

    axis = -1;
    KokkosFFT::transpose(x, xt_axis_minus1, axis); // xt is identical to x
    EXPECT_TRUE( allclose(xt_axis_minus1, ref_axis_minus1, 1.e-5, 1.e-12) );

    axis = -2;
    KokkosFFT::transpose(x, xt_axis_minus2, axis); // xt is the transpose of x
    EXPECT_TRUE( allclose(xt_axis_minus2, ref_axis_minus2, 1.e-5, 1.e-12) );
}
*/