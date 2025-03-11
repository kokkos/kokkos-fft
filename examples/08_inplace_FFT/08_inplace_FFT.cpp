// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;

template <typename T>
using RightView2D = Kokkos::View<T **, Kokkos::LayoutRight, execution_space>;

int main(int argc, char *argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int n0 = 128, n1 = 128;
    const Kokkos::complex<double> z(1.0, 1.0);

    // Forward and backward complex to complex transform
    // Define a 2D complex view to handle data
    RightView2D<Kokkos::complex<double>> xc2c("xc2c", n0, n1);

    // Fill the input view with random data
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, xc2c, random_pool, z);

    KokkosFFT::fft2(exec, xc2c, xc2c);
    KokkosFFT::ifft2(exec, xc2c, xc2c);

    // Real to complex transform
    // Define a 2D complex view to handle data
    RightView2D<Kokkos::complex<double>> xr2c_hat("xr2c", n0, n1 / 2 + 1);

    // Create unmanaged views on the same data with the FFT shape,
    // that is (n0, n1) -> (n0, n1/2+1) R2C transform
    // The shape is incorrect from the view point of casting to real
    // For casting, the shape should be (n0, (n1/2+1) * 2)
    RightView2D<double> xr2c(reinterpret_cast<double *>(xr2c_hat.data()), n0,
                             n1),
        xr2c_padded(reinterpret_cast<double *>(xr2c_hat.data()), n0,
                    (n1 / 2 + 1) * 2);

    // Fill the input view with random data in real space through xr2c_padded
    auto sub_xr2c_padded =
        Kokkos::subview(xr2c_padded, Kokkos::ALL, Kokkos::make_pair(0, n1));
    Kokkos::fill_random(exec, sub_xr2c_padded, random_pool, 1.0);

    // Perform the real to complex transform
    // [Important] You must use xr2c to define the FFT shape correctly
    KokkosFFT::rfft2(exec, xr2c, xr2c_hat);

    // Complex to real transform
    // Define a 2D complex view to handle data
    RightView2D<Kokkos::complex<double>> xc2r("xc2r", n0, n1 / 2 + 1);

    // Create an unmanaged view on the same data with the FFT shape
    RightView2D<double> xc2r_hat(reinterpret_cast<double *>(xc2r.data()), n0,
                                 n1);

    // Fill the input view with random data in complex space
    Kokkos::fill_random(exec, xc2r, random_pool, z);

    // Perform the complex to real transform
    // [Important] You must use xc2r_hat to define the FFT shape correctly
    KokkosFFT::irfft2(exec, xc2r, xc2r_hat);

    exec.fence();
  }
  Kokkos::finalize();

  return 0;
}
