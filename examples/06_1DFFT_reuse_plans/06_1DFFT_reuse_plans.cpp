// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, execution_space>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int n0 = 128;
    const Kokkos::complex<double> I(1.0, 1.0);

    // 1D C2C FFT (Forward and Backward)
    View1D<Kokkos::complex<double> > xc2c("xc2c", n0);
    View1D<Kokkos::complex<double> > xc2c_hat("xc2c_hat", n0);
    View1D<Kokkos::complex<double> > xc2c_inv("xc2c_inv", n0);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, xc2c, random_pool, I);
    exec.fence();

    int axis = -1;
    KokkosFFT::Impl::Plan fft_plan(exec, xc2c, xc2c_hat,
                                   KokkosFFT::Direction::forward, axis);
    KokkosFFT::fft(exec, xc2c, xc2c_hat, fft_plan);
    exec.fence();

    KokkosFFT::Impl::Plan ifft_plan(exec, xc2c_hat, xc2c_inv,
                                    KokkosFFT::Direction::backward, axis);
    KokkosFFT::ifft(exec, xc2c_hat, xc2c_inv, ifft_plan);
    exec.fence();

    // 1D R2C FFT
    View1D<double> xr2c("xr2c", n0);
    View1D<Kokkos::complex<double> > xr2c_hat("xr2c_hat", n0 / 2 + 1);
    Kokkos::fill_random(exec, xr2c, random_pool, 1);
    exec.fence();

    KokkosFFT::Impl::Plan rfft_plan(exec, xr2c, xr2c_hat,
                                    KokkosFFT::Direction::forward, axis);
    KokkosFFT::rfft(exec, xr2c, xr2c_hat, rfft_plan);
    exec.fence();

    // 1D C2R FFT
    View1D<Kokkos::complex<double> > xc2r("xc2r_hat", n0 / 2 + 1);
    View1D<double> xc2r_hat("xc2r", n0);
    Kokkos::fill_random(exec, xc2r, random_pool, I);
    exec.fence();

    KokkosFFT::Impl::Plan irfft_plan(exec, xc2r, xc2r_hat,
                                     KokkosFFT::Direction::backward, axis);
    KokkosFFT::irfft(exec, xc2r, xc2r_hat, irfft_plan);
    exec.fence();
  }
  Kokkos::finalize();

  return 0;
}