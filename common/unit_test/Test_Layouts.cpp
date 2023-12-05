#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include <vector>
#include "KokkosFFT_layouts.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

TEST(Layouts, 1D) {
    const int n0 = 6;
    View1D<double> xr("xr", n0);
    View1D<Kokkos::complex<double>> xc("xc", n0/2+1);
    View1D<Kokkos::complex<double>> xcin("xcin", n0), xcout("xcout", n0);

    // R2C
    std::vector<int> ref_in_extents_r2c(1), ref_out_extents_r2c(1), ref_fft_extents_r2c(1);
    ref_in_extents_r2c.at(0) = n0;
    ref_out_extents_r2c.at(0) = n0/2+1;
    ref_fft_extents_r2c.at(0) = n0;

    auto [in_extents_r2c, out_extents_r2c, fft_extents_r2c] = KokkosFFT::get_extents(xr, xc, 0);
    EXPECT_TRUE( in_extents_r2c == ref_in_extents_r2c );
    EXPECT_TRUE( out_extents_r2c == ref_out_extents_r2c );
    EXPECT_TRUE( fft_extents_r2c == ref_fft_extents_r2c );

    // C2R
    std::vector<int> ref_in_extents_c2r(1), ref_out_extents_c2r(1), ref_fft_extents_c2r(1);
    ref_in_extents_c2r.at(0) = n0/2+1;
    ref_out_extents_c2r.at(0) = n0;
    ref_fft_extents_c2r.at(0) = n0;

    auto [in_extents_c2r, out_extents_c2r, fft_extents_c2r] = KokkosFFT::get_extents(xc, xr, 0);
    EXPECT_TRUE( in_extents_c2r == ref_in_extents_c2r );
    EXPECT_TRUE( out_extents_c2r == ref_out_extents_c2r );
    EXPECT_TRUE( fft_extents_c2r == ref_fft_extents_c2r );

    // C2C
    std::vector<int> ref_in_extents_c2c(1), ref_out_extents_c2c(1), ref_fft_extents_c2c(1);
    ref_in_extents_c2c.at(0) = n0;
    ref_out_extents_c2c.at(0) = n0;
    ref_fft_extents_c2c.at(0) = n0;
    auto [in_extents_c2c, out_extents_c2c, fft_extents_c2c] = KokkosFFT::get_extents(xcin, xcout, 0);
    EXPECT_TRUE( in_extents_c2c == ref_in_extents_c2c );
    EXPECT_TRUE( out_extents_c2c == ref_out_extents_c2c );
    EXPECT_TRUE( fft_extents_c2c == ref_fft_extents_c2c );
}

TEST(Layouts, 2DLeft) {
    const int n0 = 6, n1 = 10;

    LeftView2D<double> xr2("l_xr2", n0, n1);
    LeftView2D<Kokkos::complex<double>> xc2_axis0("xc2_axis0", n0/2+1, n1);
    LeftView2D<Kokkos::complex<double>> xc2_axis1("xc2_axis1", n0, n1/2+1);
    LeftView2D<Kokkos::complex<double>> xcin2("xcin2", n0, n1), xcout2("xcout2", n0, n1);

    // R2C
    std::size_t rank = 2;
    std::vector<int> ref_in_extents_r2c_axis0{n0, n1};
    std::vector<int> ref_in_extents_r2c_axis1{n1, n0};
    std::vector<int> ref_fft_extents_r2c_axis0{n0, n1};
    std::vector<int> ref_fft_extents_r2c_axis1{n1, n0};
    std::vector<int> ref_out_extents_r2c_axis0{n0/2+1, n1};
    std::vector<int> ref_out_extents_r2c_axis1{n1/2+1, n0};

    std::reverse( ref_in_extents_r2c_axis0.begin(), ref_in_extents_r2c_axis0.end() );
    std::reverse( ref_in_extents_r2c_axis1.begin(), ref_in_extents_r2c_axis1.end() );
    std::reverse( ref_fft_extents_r2c_axis0.begin(), ref_fft_extents_r2c_axis0.end() );
    std::reverse( ref_fft_extents_r2c_axis1.begin(), ref_fft_extents_r2c_axis1.end() );
    std::reverse( ref_out_extents_r2c_axis0.begin(), ref_out_extents_r2c_axis0.end() );
    std::reverse( ref_out_extents_r2c_axis1.begin(), ref_out_extents_r2c_axis1.end() );

    auto [in_extents_r2c_axis0, out_extents_r2c_axis0, fft_extents_r2c_axis0] = KokkosFFT::get_extents(xr2, xc2_axis0, 0);
    auto [in_extents_r2c_axis1, out_extents_r2c_axis1, fft_extents_r2c_axis1] = KokkosFFT::get_extents(xr2, xc2_axis1, 1);
    EXPECT_TRUE( in_extents_r2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( in_extents_r2c_axis1 == ref_in_extents_r2c_axis1 );

    EXPECT_TRUE( fft_extents_r2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_r2c_axis1 == ref_fft_extents_r2c_axis1 );

    EXPECT_TRUE( out_extents_r2c_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_r2c_axis1 == ref_out_extents_r2c_axis1 );

    // C2R
    auto [in_extents_c2r_axis0, out_extents_c2r_axis0, fft_extents_c2r_axis0] = KokkosFFT::get_extents(xc2_axis0, xr2, 0);
    auto [in_extents_c2r_axis1, out_extents_c2r_axis1, fft_extents_c2r_axis1] = KokkosFFT::get_extents(xc2_axis1, xr2, 1);
    EXPECT_TRUE( in_extents_c2r_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( in_extents_c2r_axis1 == ref_out_extents_r2c_axis1 );

    EXPECT_TRUE( fft_extents_c2r_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2r_axis1 == ref_fft_extents_r2c_axis1 );

    EXPECT_TRUE( out_extents_c2r_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2r_axis1 == ref_in_extents_r2c_axis1 );

    // C2C
    auto [in_extents_c2c_axis0, out_extents_c2c_axis0, fft_extents_c2c_axis0] = KokkosFFT::get_extents(xcin2, xcout2, 0);
    auto [in_extents_c2c_axis1, out_extents_c2c_axis1, fft_extents_c2c_axis1] = KokkosFFT::get_extents(xcin2, xcout2, 1);
    EXPECT_TRUE( in_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( in_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );

    EXPECT_TRUE( fft_extents_c2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2c_axis1 == ref_fft_extents_r2c_axis1 );

    EXPECT_TRUE( out_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
}

TEST(Layouts, 2DRight) {
    const int n0 = 6, n1 = 10;

    RightView2D<double> xr2("r_xr2", n0, n1);
    RightView2D<Kokkos::complex<double>> xc2_axis0("xc2_axis0", n0/2+1, n1);
    RightView2D<Kokkos::complex<double>> xc2_axis1("xc2_axis1", n0, n1/2+1);
    RightView2D<Kokkos::complex<double>> xcin2("xcin2", n0, n1), xcout2("xcout2", n0, n1);

    // R2C
    std::size_t rank = 2;
    std::vector<int> ref_in_extents_r2c_axis0{n1, n0};
    std::vector<int> ref_in_extents_r2c_axis1{n0, n1};
    std::vector<int> ref_fft_extents_r2c_axis0{n1, n0};
    std::vector<int> ref_fft_extents_r2c_axis1{n0, n1};
    std::vector<int> ref_out_extents_r2c_axis0{n1, n0/2+1};
    std::vector<int> ref_out_extents_r2c_axis1{n0, n1/2+1};

    auto [in_extents_r2c_axis0, out_extents_r2c_axis0, fft_extents_r2c_axis0] = KokkosFFT::get_extents(xr2, xc2_axis0, 0);
    auto [in_extents_r2c_axis1, out_extents_r2c_axis1, fft_extents_r2c_axis1] = KokkosFFT::get_extents(xr2, xc2_axis1, 1);
    EXPECT_TRUE( in_extents_r2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( in_extents_r2c_axis1 == ref_in_extents_r2c_axis1 );

    EXPECT_TRUE( fft_extents_r2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_r2c_axis1 == ref_fft_extents_r2c_axis1 );

    EXPECT_TRUE( out_extents_r2c_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_r2c_axis1 == ref_out_extents_r2c_axis1 );

    // C2R
    auto [in_extents_c2r_axis0, out_extents_c2r_axis0, fft_extents_c2r_axis0] = KokkosFFT::get_extents(xc2_axis0, xr2, 0);
    auto [in_extents_c2r_axis1, out_extents_c2r_axis1, fft_extents_c2r_axis1] = KokkosFFT::get_extents(xc2_axis1, xr2, 1);
    EXPECT_TRUE( in_extents_c2r_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( in_extents_c2r_axis1 == ref_out_extents_r2c_axis1 );

    EXPECT_TRUE( fft_extents_c2r_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2r_axis1 == ref_fft_extents_r2c_axis1 );

    EXPECT_TRUE( out_extents_c2r_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2r_axis1 == ref_in_extents_r2c_axis1 );

    // C2C
    auto [in_extents_c2c_axis0, out_extents_c2c_axis0, fft_extents_c2c_axis0] = KokkosFFT::get_extents(xcin2, xcout2, 0);
    auto [in_extents_c2c_axis1, out_extents_c2c_axis1, fft_extents_c2c_axis1] = KokkosFFT::get_extents(xcin2, xcout2, 1);
    EXPECT_TRUE( in_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( in_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );

    EXPECT_TRUE( fft_extents_c2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2c_axis1 == ref_fft_extents_r2c_axis1 );

    EXPECT_TRUE( out_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
}

TEST(Layouts, 1DBatchedFFT_2DLeftView) {
    const int n0 = 6, n1 = 10;

    LeftView2D<double> xr2("l_xr2", n0, n1);
    LeftView2D<Kokkos::complex<double>> xc2_axis0("xc2_axis0", n0/2+1, n1);
    LeftView2D<Kokkos::complex<double>> xc2_axis1("xc2_axis1", n0, n1/2+1);
    LeftView2D<Kokkos::complex<double>> xcin2("xcin2", n0, n1), xcout2("xcout2", n0, n1);

    // Reference shapes
    std::vector<int> ref_in_extents_r2c_axis1{n1}, ref_in_extents_r2c_axis0{n0};
    std::vector<int> ref_fft_extents_r2c_axis1{n1}, ref_fft_extents_r2c_axis0{n0};
    std::vector<int> ref_out_extents_r2c_axis1{n1/2+1}, ref_out_extents_r2c_axis0{n0/2+1};
    int ref_howmany_r2c_axis0 = n1;
    int ref_howmany_r2c_axis1 = n0;

    // R2C
    auto [in_extents_r2c_axis0, out_extents_r2c_axis0, fft_extents_r2c_axis0, howmany_r2c_axis0] = KokkosFFT::get_extents_batched(xr2, xc2_axis0, 0);
    EXPECT_TRUE( in_extents_r2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_r2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_r2c_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_EQ( howmany_r2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_r2c_axis1, out_extents_r2c_axis1, fft_extents_r2c_axis1, howmany_r2c_axis1] = KokkosFFT::get_extents_batched(xr2, xc2_axis1, 1);
    EXPECT_TRUE( in_extents_r2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_r2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_r2c_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_EQ( howmany_r2c_axis1, ref_howmany_r2c_axis1 );

    // C2R
    auto [in_extents_c2r_axis0, out_extents_c2r_axis0, fft_extents_c2r_axis0, howmany_c2r_axis0] = KokkosFFT::get_extents_batched(xc2_axis0, xr2, 0);
    EXPECT_TRUE( in_extents_c2r_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2r_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2r_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2r_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2r_axis1, out_extents_c2r_axis1, fft_extents_c2r_axis1, howmany_c2r_axis1] = KokkosFFT::get_extents_batched(xc2_axis1, xr2, 1);
    EXPECT_TRUE( in_extents_c2r_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2r_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2r_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2r_axis1, ref_howmany_r2c_axis1 );

    // C2C
    auto [in_extents_c2c_axis0, out_extents_c2c_axis0, fft_extents_c2c_axis0, howmany_c2c_axis0] = KokkosFFT::get_extents_batched(xcin2, xcout2, 0);
    EXPECT_TRUE( in_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2c_axis1, out_extents_c2c_axis1, fft_extents_c2c_axis1, howmany_c2c_axis1] = KokkosFFT::get_extents_batched(xcin2, xcout2, 1);
    EXPECT_TRUE( in_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2c_axis1, ref_howmany_r2c_axis1 );
}

TEST(Layouts, 1DBatchedFFT_2DRightView) {
    const int n0 = 6, n1 = 10;

    RightView2D<double> xr2("r_xr2", n0, n1);
    RightView2D<Kokkos::complex<double>> xc2_axis0("xc2_axis0", n0/2+1, n1);
    RightView2D<Kokkos::complex<double>> xc2_axis1("xc2_axis1", n0, n1/2+1);
    RightView2D<Kokkos::complex<double>> xcin2("xcin2", n0, n1), xcout2("xcout2", n0, n1);

    // Reference shapes
    std::vector<int> ref_in_extents_r2c_axis1{n1}, ref_in_extents_r2c_axis0{n0};
    std::vector<int> ref_fft_extents_r2c_axis1{n1}, ref_fft_extents_r2c_axis0{n0};
    std::vector<int> ref_out_extents_r2c_axis1{n1/2+1}, ref_out_extents_r2c_axis0{n0/2+1};
    int ref_howmany_r2c_axis0 = n1;
    int ref_howmany_r2c_axis1 = n0;

    // R2C
    auto [in_extents_r2c_axis0, out_extents_r2c_axis0, fft_extents_r2c_axis0, howmany_r2c_axis0] = KokkosFFT::get_extents_batched(xr2, xc2_axis0, 0);
    EXPECT_TRUE( in_extents_r2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_r2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_r2c_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_EQ( howmany_r2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_r2c_axis1, out_extents_r2c_axis1, fft_extents_r2c_axis1, howmany_r2c_axis1] = KokkosFFT::get_extents_batched(xr2, xc2_axis1, 1);
    EXPECT_TRUE( in_extents_r2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_r2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_r2c_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_EQ( howmany_r2c_axis1, ref_howmany_r2c_axis1 );

    // C2R
    auto [in_extents_c2r_axis0, out_extents_c2r_axis0, fft_extents_c2r_axis0, howmany_c2r_axis0] = KokkosFFT::get_extents_batched(xc2_axis0, xr2, 0);
    EXPECT_TRUE( in_extents_c2r_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2r_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2r_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2r_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2r_axis1, out_extents_c2r_axis1, fft_extents_c2r_axis1, howmany_c2r_axis1] = KokkosFFT::get_extents_batched(xc2_axis1, xr2, 1);
    EXPECT_TRUE( in_extents_c2r_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2r_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2r_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2r_axis1, ref_howmany_r2c_axis1 );

    // C2C
    auto [in_extents_c2c_axis0, out_extents_c2c_axis0, fft_extents_c2c_axis0, howmany_c2c_axis0] = KokkosFFT::get_extents_batched(xcin2, xcout2, 0);
    EXPECT_TRUE( in_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2c_axis1, out_extents_c2c_axis1, fft_extents_c2c_axis1, howmany_c2c_axis1] = KokkosFFT::get_extents_batched(xcin2, xcout2, 1);
    EXPECT_TRUE( in_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2c_axis1, ref_howmany_r2c_axis1 );
}

TEST(Layouts, 1DBatchedFFT_3DLeftView) {
    const int n0 = 6, n1 = 10, n2 = 8;

    LeftView3D<double> xr3("r_xr3", n0, n1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis0("xc3_axis0", n0/2+1, n1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis1("xc3_axis1", n0, n1/2+1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis2("xc3_axis2", n0, n1, n2/2+1);
    LeftView3D<Kokkos::complex<double>> xcin3("xcin3", n0, n1, n2), xcout3("xcout3", n0, n1, n2);

    // Reference shapes
    std::vector<int> ref_in_extents_r2c_axis0{n0};
    std::vector<int> ref_fft_extents_r2c_axis0{n0};
    std::vector<int> ref_out_extents_r2c_axis0{n0/2+1};
    int ref_howmany_r2c_axis0 = n1 * n2;

    std::vector<int> ref_in_extents_r2c_axis1{n1};
    std::vector<int> ref_fft_extents_r2c_axis1{n1};
    std::vector<int> ref_out_extents_r2c_axis1{n1/2+1};
    int ref_howmany_r2c_axis1 = n0 * n2;

    std::vector<int> ref_in_extents_r2c_axis2{n2};
    std::vector<int> ref_fft_extents_r2c_axis2{n2};
    std::vector<int> ref_out_extents_r2c_axis2{n2/2+1};
    int ref_howmany_r2c_axis2 = n0 * n1;

    // R2C
    auto [in_extents_r2c_axis0, out_extents_r2c_axis0, fft_extents_r2c_axis0, howmany_r2c_axis0] = KokkosFFT::get_extents_batched(xr3, xc3_axis0, 0);
    EXPECT_TRUE( in_extents_r2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_r2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_r2c_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_EQ( howmany_r2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_r2c_axis1, out_extents_r2c_axis1, fft_extents_r2c_axis1, howmany_r2c_axis1] = KokkosFFT::get_extents_batched(xr3, xc3_axis1, 1);
    EXPECT_TRUE( in_extents_r2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_r2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_r2c_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_EQ( howmany_r2c_axis1, ref_howmany_r2c_axis1 );

    auto [in_extents_r2c_axis2, out_extents_r2c_axis2, fft_extents_r2c_axis2, howmany_r2c_axis2] = KokkosFFT::get_extents_batched(xr3, xc3_axis2, 2);
    EXPECT_TRUE( in_extents_r2c_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_TRUE( fft_extents_r2c_axis2 == ref_fft_extents_r2c_axis2 );
    EXPECT_TRUE( out_extents_r2c_axis2 == ref_out_extents_r2c_axis2 );
    EXPECT_EQ( howmany_r2c_axis2, ref_howmany_r2c_axis2 );

    // C2R
    auto [in_extents_c2r_axis0, out_extents_c2r_axis0, fft_extents_c2r_axis0, howmany_c2r_axis0] = KokkosFFT::get_extents_batched(xc3_axis0, xr3, 0);
    EXPECT_TRUE( in_extents_c2r_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2r_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2r_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2r_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2r_axis1, out_extents_c2r_axis1, fft_extents_c2r_axis1, howmany_c2r_axis1] = KokkosFFT::get_extents_batched(xc3_axis1, xr3, 1);
    EXPECT_TRUE( in_extents_c2r_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2r_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2r_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2r_axis1, ref_howmany_r2c_axis1 );

    auto [in_extents_c2r_axis2, out_extents_c2r_axis2, fft_extents_c2r_axis2, howmany_c2r_axis2] = KokkosFFT::get_extents_batched(xc3_axis2, xr3, 2);
    EXPECT_TRUE( in_extents_c2r_axis2 == ref_out_extents_r2c_axis2 );
    EXPECT_TRUE( fft_extents_c2r_axis2 == ref_fft_extents_r2c_axis2 );
    EXPECT_TRUE( out_extents_c2r_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_EQ( howmany_c2r_axis2, ref_howmany_r2c_axis2 );

    // C2C
    auto [in_extents_c2c_axis0, out_extents_c2c_axis0, fft_extents_c2c_axis0, howmany_c2c_axis0] = KokkosFFT::get_extents_batched(xcin3, xcout3, 0);
    EXPECT_TRUE( in_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2c_axis1, out_extents_c2c_axis1, fft_extents_c2c_axis1, howmany_c2c_axis1] = KokkosFFT::get_extents_batched(xcin3, xcout3, 1);
    EXPECT_TRUE( in_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2c_axis1, ref_howmany_r2c_axis1 );

    auto [in_extents_c2c_axis2, out_extents_c2c_axis2, fft_extents_c2c_axis2, howmany_c2c_axis2] = KokkosFFT::get_extents_batched(xcin3, xcout3, 2);
    EXPECT_TRUE( in_extents_c2c_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_TRUE( fft_extents_c2c_axis2 == ref_fft_extents_r2c_axis2 );
    EXPECT_TRUE( out_extents_c2c_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_EQ( howmany_c2c_axis2, ref_howmany_r2c_axis2 );
}

TEST(Layouts, 1DBatchedFFT_3DRightView) {
    const int n0 = 6, n1 = 10, n2 = 8;

    RightView3D<double> xr3("r_xr3", n0, n1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis0("xc3_axis0", n0/2+1, n1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis1("xc3_axis1", n0, n1/2+1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis2("xc3_axis2", n0, n1, n2/2+1);
    RightView3D<Kokkos::complex<double>> xcin3("xcin3", n0, n1, n2), xcout3("xcout3", n0, n1, n2);

    // Reference shapes
    std::vector<int> ref_in_extents_r2c_axis0{n0};
    std::vector<int> ref_fft_extents_r2c_axis0{n0};
    std::vector<int> ref_out_extents_r2c_axis0{n0/2+1};
    int ref_howmany_r2c_axis0 = n1 * n2;

    std::vector<int> ref_in_extents_r2c_axis1{n1};
    std::vector<int> ref_fft_extents_r2c_axis1{n1};
    std::vector<int> ref_out_extents_r2c_axis1{n1/2+1};
    int ref_howmany_r2c_axis1 = n0 * n2;

    std::vector<int> ref_in_extents_r2c_axis2{n2};
    std::vector<int> ref_fft_extents_r2c_axis2{n2};
    std::vector<int> ref_out_extents_r2c_axis2{n2/2+1};
    int ref_howmany_r2c_axis2 = n0 * n1;

    // R2C
    auto [in_extents_r2c_axis0, out_extents_r2c_axis0, fft_extents_r2c_axis0, howmany_r2c_axis0] = KokkosFFT::get_extents_batched(xr3, xc3_axis0, 0);
    EXPECT_TRUE( in_extents_r2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_r2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_r2c_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_EQ( howmany_r2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_r2c_axis1, out_extents_r2c_axis1, fft_extents_r2c_axis1, howmany_r2c_axis1] = KokkosFFT::get_extents_batched(xr3, xc3_axis1, 1);
    EXPECT_TRUE( in_extents_r2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_r2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_r2c_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_EQ( howmany_r2c_axis1, ref_howmany_r2c_axis1 );

    auto [in_extents_r2c_axis2, out_extents_r2c_axis2, fft_extents_r2c_axis2, howmany_r2c_axis2] = KokkosFFT::get_extents_batched(xr3, xc3_axis2, 2);
    EXPECT_TRUE( in_extents_r2c_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_TRUE( fft_extents_r2c_axis2 == ref_fft_extents_r2c_axis2 );
    EXPECT_TRUE( out_extents_r2c_axis2 == ref_out_extents_r2c_axis2 );
    EXPECT_EQ( howmany_r2c_axis2, ref_howmany_r2c_axis2 );

    // C2R
    auto [in_extents_c2r_axis0, out_extents_c2r_axis0, fft_extents_c2r_axis0, howmany_c2r_axis0] = KokkosFFT::get_extents_batched(xc3_axis0, xr3, 0);
    EXPECT_TRUE( in_extents_c2r_axis0 == ref_out_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2r_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2r_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2r_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2r_axis1, out_extents_c2r_axis1, fft_extents_c2r_axis1, howmany_c2r_axis1] = KokkosFFT::get_extents_batched(xc3_axis1, xr3, 1);
    EXPECT_TRUE( in_extents_c2r_axis1 == ref_out_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2r_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2r_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2r_axis1, ref_howmany_r2c_axis1 );

    auto [in_extents_c2r_axis2, out_extents_c2r_axis2, fft_extents_c2r_axis2, howmany_c2r_axis2] = KokkosFFT::get_extents_batched(xc3_axis2, xr3, 2);
    EXPECT_TRUE( in_extents_c2r_axis2 == ref_out_extents_r2c_axis2 );
    EXPECT_TRUE( fft_extents_c2r_axis2 == ref_fft_extents_r2c_axis2 );
    EXPECT_TRUE( out_extents_c2r_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_EQ( howmany_c2r_axis2, ref_howmany_r2c_axis2 );

    // C2C
    auto [in_extents_c2c_axis0, out_extents_c2c_axis0, fft_extents_c2c_axis0, howmany_c2c_axis0] = KokkosFFT::get_extents_batched(xcin3, xcout3, 0);
    EXPECT_TRUE( in_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_TRUE( fft_extents_c2c_axis0 == ref_fft_extents_r2c_axis0 );
    EXPECT_TRUE( out_extents_c2c_axis0 == ref_in_extents_r2c_axis0 );
    EXPECT_EQ( howmany_c2c_axis0, ref_howmany_r2c_axis0 );

    auto [in_extents_c2c_axis1, out_extents_c2c_axis1, fft_extents_c2c_axis1, howmany_c2c_axis1] = KokkosFFT::get_extents_batched(xcin3, xcout3, 1);
    EXPECT_TRUE( in_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_TRUE( fft_extents_c2c_axis1 == ref_fft_extents_r2c_axis1 );
    EXPECT_TRUE( out_extents_c2c_axis1 == ref_in_extents_r2c_axis1 );
    EXPECT_EQ( howmany_c2c_axis1, ref_howmany_r2c_axis1 );

    auto [in_extents_c2c_axis2, out_extents_c2c_axis2, fft_extents_c2c_axis2, howmany_c2c_axis2] = KokkosFFT::get_extents_batched(xcin3, xcout3, 2);
    EXPECT_TRUE( in_extents_c2c_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_TRUE( fft_extents_c2c_axis2 == ref_fft_extents_r2c_axis2 );
    EXPECT_TRUE( out_extents_c2c_axis2 == ref_in_extents_r2c_axis2 );
    EXPECT_EQ( howmany_c2c_axis2, ref_howmany_r2c_axis2 );
}

TEST(Layouts, 2DBatchedFFT_3DLeftView) {
    using axes_type = KokkosFFT::axis_type<2>;
    const int n0 = 6, n1 = 10, n2 = 8;

    LeftView3D<double> xr3("r_xr3", n0, n1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis_01("xc3_axis_01", n0, n1/2+1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis_02("xc3_axis_02", n0, n1, n2/2+1);
    LeftView3D<Kokkos::complex<double>> xc3_axis_10("xc3_axis_10", n0/2+1, n1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis_12("xc3_axis_12", n0, n1, n2/2+1);
    LeftView3D<Kokkos::complex<double>> xc3_axis_20("xc3_axis_20", n0/2+1, n1, n2);
    LeftView3D<Kokkos::complex<double>> xc3_axis_21("xc3_axis_21", n0, n1/2+1, n2);
    LeftView3D<Kokkos::complex<double>> xcin3("xcin3", n0, n1, n2), xcout3("xcout3", n0, n1, n2);

    // Reference shapes
    std::vector<int> ref_in_extents_r2c_axis_01{n0, n1};
    std::vector<int> ref_fft_extents_r2c_axis_01{n0, n1};
    std::vector<int> ref_out_extents_r2c_axis_01{n0, n1/2+1};
    int ref_howmany_r2c_axis_01 = n2;

    std::vector<int> ref_in_extents_r2c_axis_02{n0, n2};
    std::vector<int> ref_fft_extents_r2c_axis_02{n0, n2};
    std::vector<int> ref_out_extents_r2c_axis_02{n0, n2/2+1};
    int ref_howmany_r2c_axis_02 = n1;

    std::vector<int> ref_in_extents_r2c_axis_10{n1, n0};
    std::vector<int> ref_fft_extents_r2c_axis_10{n1, n0};
    std::vector<int> ref_out_extents_r2c_axis_10{n1, n0/2+1};
    int ref_howmany_r2c_axis_10 = n2;

    std::vector<int> ref_in_extents_r2c_axis_12{n1, n2};
    std::vector<int> ref_fft_extents_r2c_axis_12{n1, n2};
    std::vector<int> ref_out_extents_r2c_axis_12{n1, n2/2+1};
    int ref_howmany_r2c_axis_12 = n0;

    std::vector<int> ref_in_extents_r2c_axis_20{n2, n0};
    std::vector<int> ref_fft_extents_r2c_axis_20{n2, n0};
    std::vector<int> ref_out_extents_r2c_axis_20{n2, n0/2+1};
    int ref_howmany_r2c_axis_20 = n1;

    std::vector<int> ref_in_extents_r2c_axis_21{n2, n1};
    std::vector<int> ref_fft_extents_r2c_axis_21{n2, n1};
    std::vector<int> ref_out_extents_r2c_axis_21{n2, n1/2+1};
    int ref_howmany_r2c_axis_21 = n0;

    // R2C
    auto [in_extents_r2c_axis_01, out_extents_r2c_axis_01, fft_extents_r2c_axis_01, howmany_r2c_axis_01] = KokkosFFT::get_extents_batched(xr3, xc3_axis_01, axes_type({0, 1}));
    EXPECT_TRUE( in_extents_r2c_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_TRUE( fft_extents_r2c_axis_01 == ref_fft_extents_r2c_axis_01 );
    EXPECT_TRUE( out_extents_r2c_axis_01 == ref_out_extents_r2c_axis_01 );
    EXPECT_EQ( howmany_r2c_axis_01, ref_howmany_r2c_axis_01 );

    auto [in_extents_r2c_axis_02, out_extents_r2c_axis_02, fft_extents_r2c_axis_02, howmany_r2c_axis_02] = KokkosFFT::get_extents_batched(xr3, xc3_axis_02, axes_type({0, 2}));
    EXPECT_TRUE( in_extents_r2c_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_TRUE( fft_extents_r2c_axis_02 == ref_fft_extents_r2c_axis_02 );
    EXPECT_TRUE( out_extents_r2c_axis_02 == ref_out_extents_r2c_axis_02 );
    EXPECT_EQ( howmany_r2c_axis_02, ref_howmany_r2c_axis_02 );

    auto [in_extents_r2c_axis_10, out_extents_r2c_axis_10, fft_extents_r2c_axis_10, howmany_r2c_axis_10] = KokkosFFT::get_extents_batched(xr3, xc3_axis_10, axes_type({1, 0}));
    EXPECT_TRUE( in_extents_r2c_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_TRUE( fft_extents_r2c_axis_10 == ref_fft_extents_r2c_axis_10 );
    EXPECT_TRUE( out_extents_r2c_axis_10 == ref_out_extents_r2c_axis_10 );
    EXPECT_EQ( howmany_r2c_axis_10, ref_howmany_r2c_axis_10 );

    auto [in_extents_r2c_axis_12, out_extents_r2c_axis_12, fft_extents_r2c_axis_12, howmany_r2c_axis_12] = KokkosFFT::get_extents_batched(xr3, xc3_axis_12, axes_type({1, 2}));
    EXPECT_TRUE( in_extents_r2c_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_TRUE( fft_extents_r2c_axis_12 == ref_fft_extents_r2c_axis_12 );
    EXPECT_TRUE( out_extents_r2c_axis_12 == ref_out_extents_r2c_axis_12 );
    EXPECT_EQ( howmany_r2c_axis_12, ref_howmany_r2c_axis_12 );

    auto [in_extents_r2c_axis_20, out_extents_r2c_axis_20, fft_extents_r2c_axis_20, howmany_r2c_axis_20] = KokkosFFT::get_extents_batched(xr3, xc3_axis_20, axes_type({2, 0}));
    EXPECT_TRUE( in_extents_r2c_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_TRUE( fft_extents_r2c_axis_20 == ref_fft_extents_r2c_axis_20 );
    EXPECT_TRUE( out_extents_r2c_axis_20 == ref_out_extents_r2c_axis_20 );
    EXPECT_EQ( howmany_r2c_axis_20, ref_howmany_r2c_axis_20 );

    auto [in_extents_r2c_axis_21, out_extents_r2c_axis_21, fft_extents_r2c_axis_21, howmany_r2c_axis_21] = KokkosFFT::get_extents_batched(xr3, xc3_axis_21, axes_type({2, 1}));
    EXPECT_TRUE( in_extents_r2c_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_TRUE( fft_extents_r2c_axis_21 == ref_fft_extents_r2c_axis_21 );
    EXPECT_TRUE( out_extents_r2c_axis_21 == ref_out_extents_r2c_axis_21 );
    EXPECT_EQ( howmany_r2c_axis_21, ref_howmany_r2c_axis_21 );

    // C2R
    auto [in_extents_c2r_axis_01, out_extents_c2r_axis_01, fft_extents_c2r_axis_01, howmany_c2r_axis_01] = KokkosFFT::get_extents_batched(xc3_axis_01, xr3, axes_type({0, 1}));
    EXPECT_TRUE( in_extents_c2r_axis_01 == ref_out_extents_r2c_axis_01 );
    EXPECT_TRUE( fft_extents_c2r_axis_01 == ref_fft_extents_r2c_axis_01 );
    EXPECT_TRUE( out_extents_c2r_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_EQ( howmany_c2r_axis_01, ref_howmany_r2c_axis_01 );

    auto [in_extents_c2r_axis_02, out_extents_c2r_axis_02, fft_extents_c2r_axis_02, howmany_c2r_axis_02] = KokkosFFT::get_extents_batched(xc3_axis_02, xr3, axes_type({0, 2}));
    EXPECT_TRUE( in_extents_c2r_axis_02 == ref_out_extents_r2c_axis_02 );
    EXPECT_TRUE( fft_extents_c2r_axis_02 == ref_fft_extents_r2c_axis_02 );
    EXPECT_TRUE( out_extents_c2r_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_EQ( howmany_c2r_axis_02, ref_howmany_r2c_axis_02 );

    auto [in_extents_c2r_axis_10, out_extents_c2r_axis_10, fft_extents_c2r_axis_10, howmany_c2r_axis_10] = KokkosFFT::get_extents_batched(xc3_axis_10, xr3, axes_type({1, 0}));
    EXPECT_TRUE( in_extents_c2r_axis_10 == ref_out_extents_r2c_axis_10 );
    EXPECT_TRUE( fft_extents_c2r_axis_10 == ref_fft_extents_r2c_axis_10 );
    EXPECT_TRUE( out_extents_c2r_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_EQ( howmany_c2r_axis_10, ref_howmany_r2c_axis_10 );

    auto [in_extents_c2r_axis_12, out_extents_c2r_axis_12, fft_extents_c2r_axis_12, howmany_c2r_axis_12] = KokkosFFT::get_extents_batched(xc3_axis_12, xr3, axes_type({1, 2}));
    EXPECT_TRUE( in_extents_c2r_axis_12 == ref_out_extents_r2c_axis_12 );
    EXPECT_TRUE( fft_extents_c2r_axis_12 == ref_fft_extents_r2c_axis_12 );
    EXPECT_TRUE( out_extents_c2r_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_EQ( howmany_c2r_axis_12, ref_howmany_r2c_axis_12 );

    auto [in_extents_c2r_axis_20, out_extents_c2r_axis_20, fft_extents_c2r_axis_20, howmany_c2r_axis_20] = KokkosFFT::get_extents_batched(xc3_axis_20, xr3, axes_type({2, 0}));
    EXPECT_TRUE( in_extents_c2r_axis_20 == ref_out_extents_r2c_axis_20 );
    EXPECT_TRUE( fft_extents_c2r_axis_20 == ref_fft_extents_r2c_axis_20 );
    EXPECT_TRUE( out_extents_c2r_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_EQ( howmany_c2r_axis_20, ref_howmany_r2c_axis_20 );

    auto [in_extents_c2r_axis_21, out_extents_c2r_axis_21, fft_extents_c2r_axis_21, howmany_c2r_axis_21] = KokkosFFT::get_extents_batched(xc3_axis_21, xr3, axes_type({2, 1}));
    EXPECT_TRUE( in_extents_c2r_axis_21 == ref_out_extents_r2c_axis_21 );
    EXPECT_TRUE( fft_extents_c2r_axis_21 == ref_fft_extents_r2c_axis_21 );
    EXPECT_TRUE( out_extents_c2r_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_EQ( howmany_c2r_axis_21, ref_howmany_r2c_axis_21 );

    // C2C
    auto [in_extents_c2c_axis_01, out_extents_c2c_axis_01, fft_extents_c2c_axis_01, howmany_c2c_axis_01] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({0, 1}));
    EXPECT_TRUE( in_extents_c2c_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_TRUE( fft_extents_c2c_axis_01 == ref_fft_extents_r2c_axis_01 );
    EXPECT_TRUE( out_extents_c2c_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_EQ( howmany_c2c_axis_01, ref_howmany_r2c_axis_01 );

    auto [in_extents_c2c_axis_02, out_extents_c2c_axis_02, fft_extents_c2c_axis_02, howmany_c2c_axis_02] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({0, 2}));
    EXPECT_TRUE( in_extents_c2c_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_TRUE( fft_extents_c2c_axis_02 == ref_fft_extents_r2c_axis_02 );
    EXPECT_TRUE( out_extents_c2c_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_EQ( howmany_c2c_axis_02, ref_howmany_r2c_axis_02 );

    auto [in_extents_c2c_axis_10, out_extents_c2c_axis_10, fft_extents_c2c_axis_10, howmany_c2c_axis_10] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({1, 0}));
    EXPECT_TRUE( in_extents_c2c_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_TRUE( fft_extents_c2c_axis_10 == ref_fft_extents_r2c_axis_10 );
    EXPECT_TRUE( out_extents_c2c_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_EQ( howmany_c2c_axis_10, ref_howmany_r2c_axis_10 );

    auto [in_extents_c2c_axis_12, out_extents_c2c_axis_12, fft_extents_c2c_axis_12, howmany_c2c_axis_12] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({1, 2}));
    EXPECT_TRUE( in_extents_c2c_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_TRUE( fft_extents_c2c_axis_12 == ref_fft_extents_r2c_axis_12 );
    EXPECT_TRUE( out_extents_c2c_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_EQ( howmany_c2c_axis_12, ref_howmany_r2c_axis_12 );

    auto [in_extents_c2c_axis_20, out_extents_c2c_axis_20, fft_extents_c2c_axis_20, howmany_c2c_axis_20] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({2, 0}));
    EXPECT_TRUE( in_extents_c2c_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_TRUE( fft_extents_c2c_axis_20 == ref_fft_extents_r2c_axis_20 );
    EXPECT_TRUE( out_extents_c2c_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_EQ( howmany_c2c_axis_20, ref_howmany_r2c_axis_20 );

    auto [in_extents_c2c_axis_21, out_extents_c2c_axis_21, fft_extents_c2c_axis_21, howmany_c2c_axis_21] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({2, 1}));
    EXPECT_TRUE( in_extents_c2c_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_TRUE( fft_extents_c2c_axis_21 == ref_fft_extents_r2c_axis_21 );
    EXPECT_TRUE( out_extents_c2c_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_EQ( howmany_c2c_axis_21, ref_howmany_r2c_axis_21 );
}

TEST(Layouts, 2DBatchedFFT_3DRightView) {
    using axes_type = KokkosFFT::axis_type<2>;
    const int n0 = 6, n1 = 10, n2 = 8;

    RightView3D<double> xr3("r_xr3", n0, n1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis_01("xc3_axis_01", n0, n1/2+1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis_02("xc3_axis_02", n0, n1, n2/2+1);
    RightView3D<Kokkos::complex<double>> xc3_axis_10("xc3_axis_10", n0/2+1, n1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis_12("xc3_axis_12", n0, n1, n2/2+1);
    RightView3D<Kokkos::complex<double>> xc3_axis_20("xc3_axis_20", n0/2+1, n1, n2);
    RightView3D<Kokkos::complex<double>> xc3_axis_21("xc3_axis_21", n0, n1/2+1, n2);
    RightView3D<Kokkos::complex<double>> xcin3("xcin3", n0, n1, n2), xcout3("xcout3", n0, n1, n2);

    // Reference shapes
    std::vector<int> ref_in_extents_r2c_axis_01{n0, n1};
    std::vector<int> ref_fft_extents_r2c_axis_01{n0, n1};
    std::vector<int> ref_out_extents_r2c_axis_01{n0, n1/2+1};
    int ref_howmany_r2c_axis_01 = n2;

    std::vector<int> ref_in_extents_r2c_axis_02{n0, n2};
    std::vector<int> ref_fft_extents_r2c_axis_02{n0, n2};
    std::vector<int> ref_out_extents_r2c_axis_02{n0, n2/2+1};
    int ref_howmany_r2c_axis_02 = n1;

    std::vector<int> ref_in_extents_r2c_axis_10{n1, n0};
    std::vector<int> ref_fft_extents_r2c_axis_10{n1, n0};
    std::vector<int> ref_out_extents_r2c_axis_10{n1, n0/2+1};
    int ref_howmany_r2c_axis_10 = n2;

    std::vector<int> ref_in_extents_r2c_axis_12{n1, n2};
    std::vector<int> ref_fft_extents_r2c_axis_12{n1, n2};
    std::vector<int> ref_out_extents_r2c_axis_12{n1, n2/2+1};
    int ref_howmany_r2c_axis_12 = n0;

    std::vector<int> ref_in_extents_r2c_axis_20{n2, n0};
    std::vector<int> ref_fft_extents_r2c_axis_20{n2, n0};
    std::vector<int> ref_out_extents_r2c_axis_20{n2, n0/2+1};
    int ref_howmany_r2c_axis_20 = n1;

    std::vector<int> ref_in_extents_r2c_axis_21{n2, n1};
    std::vector<int> ref_fft_extents_r2c_axis_21{n2, n1};
    std::vector<int> ref_out_extents_r2c_axis_21{n2, n1/2+1};
    int ref_howmany_r2c_axis_21 = n0;

    // R2C
    auto [in_extents_r2c_axis_01, out_extents_r2c_axis_01, fft_extents_r2c_axis_01, howmany_r2c_axis_01] = KokkosFFT::get_extents_batched(xr3, xc3_axis_01, axes_type({0, 1}));
    EXPECT_TRUE( in_extents_r2c_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_TRUE( fft_extents_r2c_axis_01 == ref_fft_extents_r2c_axis_01 );
    EXPECT_TRUE( out_extents_r2c_axis_01 == ref_out_extents_r2c_axis_01 );
    EXPECT_EQ( howmany_r2c_axis_01, ref_howmany_r2c_axis_01 );

    auto [in_extents_r2c_axis_02, out_extents_r2c_axis_02, fft_extents_r2c_axis_02, howmany_r2c_axis_02] = KokkosFFT::get_extents_batched(xr3, xc3_axis_02, axes_type({0, 2}));
    EXPECT_TRUE( in_extents_r2c_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_TRUE( fft_extents_r2c_axis_02 == ref_fft_extents_r2c_axis_02 );
    EXPECT_TRUE( out_extents_r2c_axis_02 == ref_out_extents_r2c_axis_02 );
    EXPECT_EQ( howmany_r2c_axis_02, ref_howmany_r2c_axis_02 );

    auto [in_extents_r2c_axis_10, out_extents_r2c_axis_10, fft_extents_r2c_axis_10, howmany_r2c_axis_10] = KokkosFFT::get_extents_batched(xr3, xc3_axis_10, axes_type({1, 0}));
    EXPECT_TRUE( in_extents_r2c_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_TRUE( fft_extents_r2c_axis_10 == ref_fft_extents_r2c_axis_10 );
    EXPECT_TRUE( out_extents_r2c_axis_10 == ref_out_extents_r2c_axis_10 );
    EXPECT_EQ( howmany_r2c_axis_10, ref_howmany_r2c_axis_10 );

    auto [in_extents_r2c_axis_12, out_extents_r2c_axis_12, fft_extents_r2c_axis_12, howmany_r2c_axis_12] = KokkosFFT::get_extents_batched(xr3, xc3_axis_12, axes_type({1, 2}));
    EXPECT_TRUE( in_extents_r2c_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_TRUE( fft_extents_r2c_axis_12 == ref_fft_extents_r2c_axis_12 );
    EXPECT_TRUE( out_extents_r2c_axis_12 == ref_out_extents_r2c_axis_12 );
    EXPECT_EQ( howmany_r2c_axis_12, ref_howmany_r2c_axis_12 );

    auto [in_extents_r2c_axis_20, out_extents_r2c_axis_20, fft_extents_r2c_axis_20, howmany_r2c_axis_20] = KokkosFFT::get_extents_batched(xr3, xc3_axis_20, axes_type({2, 0}));
    EXPECT_TRUE( in_extents_r2c_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_TRUE( fft_extents_r2c_axis_20 == ref_fft_extents_r2c_axis_20 );
    EXPECT_TRUE( out_extents_r2c_axis_20 == ref_out_extents_r2c_axis_20 );
    EXPECT_EQ( howmany_r2c_axis_20, ref_howmany_r2c_axis_20 );

    auto [in_extents_r2c_axis_21, out_extents_r2c_axis_21, fft_extents_r2c_axis_21, howmany_r2c_axis_21] = KokkosFFT::get_extents_batched(xr3, xc3_axis_21, axes_type({2, 1}));
    EXPECT_TRUE( in_extents_r2c_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_TRUE( fft_extents_r2c_axis_21 == ref_fft_extents_r2c_axis_21 );
    EXPECT_TRUE( out_extents_r2c_axis_21 == ref_out_extents_r2c_axis_21 );
    EXPECT_EQ( howmany_r2c_axis_21, ref_howmany_r2c_axis_21 );

    // C2R
    auto [in_extents_c2r_axis_01, out_extents_c2r_axis_01, fft_extents_c2r_axis_01, howmany_c2r_axis_01] = KokkosFFT::get_extents_batched(xc3_axis_01, xr3, axes_type({0, 1}));
    EXPECT_TRUE( in_extents_c2r_axis_01 == ref_out_extents_r2c_axis_01 );
    EXPECT_TRUE( fft_extents_c2r_axis_01 == ref_fft_extents_r2c_axis_01 );
    EXPECT_TRUE( out_extents_c2r_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_EQ( howmany_c2r_axis_01, ref_howmany_r2c_axis_01 );

    auto [in_extents_c2r_axis_02, out_extents_c2r_axis_02, fft_extents_c2r_axis_02, howmany_c2r_axis_02] = KokkosFFT::get_extents_batched(xc3_axis_02, xr3, axes_type({0, 2}));
    EXPECT_TRUE( in_extents_c2r_axis_02 == ref_out_extents_r2c_axis_02 );
    EXPECT_TRUE( fft_extents_c2r_axis_02 == ref_fft_extents_r2c_axis_02 );
    EXPECT_TRUE( out_extents_c2r_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_EQ( howmany_c2r_axis_02, ref_howmany_r2c_axis_02 );

    auto [in_extents_c2r_axis_10, out_extents_c2r_axis_10, fft_extents_c2r_axis_10, howmany_c2r_axis_10] = KokkosFFT::get_extents_batched(xc3_axis_10, xr3, axes_type({1, 0}));
    EXPECT_TRUE( in_extents_c2r_axis_10 == ref_out_extents_r2c_axis_10 );
    EXPECT_TRUE( fft_extents_c2r_axis_10 == ref_fft_extents_r2c_axis_10 );
    EXPECT_TRUE( out_extents_c2r_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_EQ( howmany_c2r_axis_10, ref_howmany_r2c_axis_10 );

    auto [in_extents_c2r_axis_12, out_extents_c2r_axis_12, fft_extents_c2r_axis_12, howmany_c2r_axis_12] = KokkosFFT::get_extents_batched(xc3_axis_12, xr3, axes_type({1, 2}));
    EXPECT_TRUE( in_extents_c2r_axis_12 == ref_out_extents_r2c_axis_12 );
    EXPECT_TRUE( fft_extents_c2r_axis_12 == ref_fft_extents_r2c_axis_12 );
    EXPECT_TRUE( out_extents_c2r_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_EQ( howmany_c2r_axis_12, ref_howmany_r2c_axis_12 );

    auto [in_extents_c2r_axis_20, out_extents_c2r_axis_20, fft_extents_c2r_axis_20, howmany_c2r_axis_20] = KokkosFFT::get_extents_batched(xc3_axis_20, xr3, axes_type({2, 0}));
    EXPECT_TRUE( in_extents_c2r_axis_20 == ref_out_extents_r2c_axis_20 );
    EXPECT_TRUE( fft_extents_c2r_axis_20 == ref_fft_extents_r2c_axis_20 );
    EXPECT_TRUE( out_extents_c2r_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_EQ( howmany_c2r_axis_20, ref_howmany_r2c_axis_20 );

    auto [in_extents_c2r_axis_21, out_extents_c2r_axis_21, fft_extents_c2r_axis_21, howmany_c2r_axis_21] = KokkosFFT::get_extents_batched(xc3_axis_21, xr3, axes_type({2, 1}));
    EXPECT_TRUE( in_extents_c2r_axis_21 == ref_out_extents_r2c_axis_21 );
    EXPECT_TRUE( fft_extents_c2r_axis_21 == ref_fft_extents_r2c_axis_21 );
    EXPECT_TRUE( out_extents_c2r_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_EQ( howmany_c2r_axis_21, ref_howmany_r2c_axis_21 );

    // C2C
    auto [in_extents_c2c_axis_01, out_extents_c2c_axis_01, fft_extents_c2c_axis_01, howmany_c2c_axis_01] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({0, 1}));
    EXPECT_TRUE( in_extents_c2c_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_TRUE( fft_extents_c2c_axis_01 == ref_fft_extents_r2c_axis_01 );
    EXPECT_TRUE( out_extents_c2c_axis_01 == ref_in_extents_r2c_axis_01 );
    EXPECT_EQ( howmany_c2c_axis_01, ref_howmany_r2c_axis_01 );

    auto [in_extents_c2c_axis_02, out_extents_c2c_axis_02, fft_extents_c2c_axis_02, howmany_c2c_axis_02] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({0, 2}));
    EXPECT_TRUE( in_extents_c2c_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_TRUE( fft_extents_c2c_axis_02 == ref_fft_extents_r2c_axis_02 );
    EXPECT_TRUE( out_extents_c2c_axis_02 == ref_in_extents_r2c_axis_02 );
    EXPECT_EQ( howmany_c2c_axis_02, ref_howmany_r2c_axis_02 );

    auto [in_extents_c2c_axis_10, out_extents_c2c_axis_10, fft_extents_c2c_axis_10, howmany_c2c_axis_10] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({1, 0}));
    EXPECT_TRUE( in_extents_c2c_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_TRUE( fft_extents_c2c_axis_10 == ref_fft_extents_r2c_axis_10 );
    EXPECT_TRUE( out_extents_c2c_axis_10 == ref_in_extents_r2c_axis_10 );
    EXPECT_EQ( howmany_c2c_axis_10, ref_howmany_r2c_axis_10 );

    auto [in_extents_c2c_axis_12, out_extents_c2c_axis_12, fft_extents_c2c_axis_12, howmany_c2c_axis_12] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({1, 2}));
    EXPECT_TRUE( in_extents_c2c_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_TRUE( fft_extents_c2c_axis_12 == ref_fft_extents_r2c_axis_12 );
    EXPECT_TRUE( out_extents_c2c_axis_12 == ref_in_extents_r2c_axis_12 );
    EXPECT_EQ( howmany_c2c_axis_12, ref_howmany_r2c_axis_12 );

    auto [in_extents_c2c_axis_20, out_extents_c2c_axis_20, fft_extents_c2c_axis_20, howmany_c2c_axis_20] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({2, 0}));
    EXPECT_TRUE( in_extents_c2c_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_TRUE( fft_extents_c2c_axis_20 == ref_fft_extents_r2c_axis_20 );
    EXPECT_TRUE( out_extents_c2c_axis_20 == ref_in_extents_r2c_axis_20 );
    EXPECT_EQ( howmany_c2c_axis_20, ref_howmany_r2c_axis_20 );

    auto [in_extents_c2c_axis_21, out_extents_c2c_axis_21, fft_extents_c2c_axis_21, howmany_c2c_axis_21] = KokkosFFT::get_extents_batched(xcin3, xcout3, axes_type({2, 1}));
    EXPECT_TRUE( in_extents_c2c_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_TRUE( fft_extents_c2c_axis_21 == ref_fft_extents_r2c_axis_21 );
    EXPECT_TRUE( out_extents_c2c_axis_21 == ref_in_extents_r2c_axis_21 );
    EXPECT_EQ( howmany_c2c_axis_21, ref_howmany_r2c_axis_21 );
}