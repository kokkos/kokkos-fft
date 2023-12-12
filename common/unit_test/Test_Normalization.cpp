
#include <gtest/gtest.h>
#include <Kokkos_Random.hpp>
#include "KokkosFFT_normalization.hpp"
#include "Test_Types.hpp"
#include "Test_Utils.hpp"

TEST(Normalization, Forward) {
    const int len = 30;
    View1D<double> x("x", len), ref_f("ref_f", len), ref_b("ref_b", len);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    Kokkos::deep_copy(ref_f, x);
    Kokkos::deep_copy(ref_b, x);

    double coef = 1.0 / static_cast<double>(len);
    multiply(ref_f, coef);

    Kokkos::fence();

    // Backward FFT with Forward Normalization -> Do nothing
    KokkosFFT::normalize(x, KOKKOS_FFT_BACKWARD, KokkosFFT::FFT_Normalization::FORWARD, len);
    EXPECT_TRUE( allclose(x, ref_b, 1.e-5, 1.e-12) );

    // Forward FFT with Forward Normalization -> 1/N normalization
    KokkosFFT::normalize(x, KOKKOS_FFT_FORWARD, KokkosFFT::FFT_Normalization::FORWARD, len);
    EXPECT_TRUE( allclose(x, ref_f, 1.e-5, 1.e-12) );
}

TEST(Normalization, Backward) {
    const int len = 30;
    View1D<double> x("x", len), ref_f("ref_f", len), ref_b("ref_b", len);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::fill_random(x, random_pool, 1.0);

    Kokkos::deep_copy(ref_f, x);
    Kokkos::deep_copy(ref_b, x);

    double coef = 1.0 / static_cast<double>(len);
    multiply(ref_b, coef);

    Kokkos::fence();

    // Forward FFT with Backward Normalization -> Do nothing
    KokkosFFT::normalize(x, KOKKOS_FFT_FORWARD, KokkosFFT::FFT_Normalization::BACKWARD, len);
    EXPECT_TRUE( allclose(x, ref_f, 1.e-5, 1.e-12) );

    // Backward FFT with Backward Normalization -> 1/N normalization
    KokkosFFT::normalize(x, KOKKOS_FFT_BACKWARD, KokkosFFT::FFT_Normalization::BACKWARD, len);
    EXPECT_TRUE( allclose(x, ref_b, 1.e-5, 1.e-12) );
}

TEST(Normalization, Ortho) {
    const int len = 30;
    View1D<double> x_f("x_f", len), x_b("x_b", len);
    View1D<double> ref_f("ref_f", len), ref_b("ref_b", len);

    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::fill_random(x_f, random_pool, 1.0);

    Kokkos::deep_copy(x_b, x_f);
    Kokkos::deep_copy(ref_f, x_f);
    Kokkos::deep_copy(ref_b, x_f);

    double coef = 1.0 / Kokkos::sqrt( static_cast<double>(len) );
    multiply(ref_f, coef);
    multiply(ref_b, coef);

    Kokkos::fence();

    // Forward FFT with Ortho Normalization -> 1 / sqrt(N) normalization
    KokkosFFT::normalize(x_f, KOKKOS_FFT_FORWARD, KokkosFFT::FFT_Normalization::ORTHO, len);
    EXPECT_TRUE( allclose(x_f, ref_f, 1.e-5, 1.e-12) );

    // Backward FFT with Ortho Normalization -> 1 / sqrt(N) normalization
    KokkosFFT::normalize(x_b, KOKKOS_FFT_BACKWARD, KokkosFFT::FFT_Normalization::ORTHO, len);
    EXPECT_TRUE( allclose(x_b, ref_b, 1.e-5, 1.e-12) );
}