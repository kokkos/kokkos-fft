// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
using execution_space = Kokkos::DefaultExecutionSpace;
#else
using execution_space = Kokkos::DefaultHostExecutionSpace;
#endif

using host_execution_space = Kokkos::DefaultHostExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, execution_space>;
template <typename T>
using HostView1D = Kokkos::View<T*, host_execution_space>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int n0 = 128;
    const Kokkos::complex<double> z(1.0, 1.0);

    // 1D C2C FFT (Forward and Backward)
    View1D<Kokkos::complex<double> > xc2c("xc2c", n0);
    View1D<Kokkos::complex<double> > xc2c_hat("xc2c_hat", n0);
    View1D<Kokkos::complex<double> > xc2c_inv("xc2c_inv", n0);

    Kokkos::Random_XorShift64_Pool<execution_space> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, xc2c, random_pool, z);

    KokkosFFT::fft(exec, xc2c, xc2c_hat);
    KokkosFFT::ifft(exec, xc2c_hat, xc2c_inv);

    // 1D R2C FFT
    View1D<double> xr2c("xr2c", n0);
    View1D<Kokkos::complex<double> > xr2c_hat("xr2c_hat", n0 / 2 + 1);
    Kokkos::fill_random(exec, xr2c, random_pool, 1);

    KokkosFFT::rfft(exec, xr2c, xr2c_hat);

    // 1D C2R FFT
    View1D<Kokkos::complex<double> > xc2r("xr2c_hat", n0 / 2 + 1);
    View1D<double> xc2r_hat("xc2r", n0);
    Kokkos::fill_random(exec, xc2r, random_pool, z);

    KokkosFFT::irfft(exec, xc2r, xc2r_hat);
    exec.fence();

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
    // FFTs on Host
    // 1D C2C FFT (Forward and Backward)
    HostView1D<Kokkos::complex<double> > h_xc2c("h_xc2c", n0);
    HostView1D<Kokkos::complex<double> > h_xc2c_hat("h_xc2c_hat", n0);
    HostView1D<Kokkos::complex<double> > h_xc2c_inv("h_xc2c_inv", n0);

    Kokkos::deep_copy(h_xc2c, xc2c);

    host_execution_space host_exec;

    KokkosFFT::fft(host_exec, h_xc2c, h_xc2c_hat);
    KokkosFFT::ifft(host_exec, h_xc2c_hat, h_xc2c_inv);
    host_exec.fence();

    // 1D R2C FFT
    HostView1D<double> h_xr2c("h_xr2c", n0);
    HostView1D<Kokkos::complex<double> > h_xr2c_hat("h_xr2c_hat", n0 / 2 + 1);

    Kokkos::deep_copy(h_xr2c, xr2c);
    KokkosFFT::rfft(host_exec, h_xr2c, h_xr2c_hat);
    host_exec.fence();

    // 1D C2R FFT
    HostView1D<Kokkos::complex<double> > h_xc2r("h_xr2c_hat", n0 / 2 + 1);
    HostView1D<double> h_xc2r_hat("h_xc2r", n0);

    Kokkos::deep_copy(h_xc2r, xc2r);
    KokkosFFT::irfft(host_exec, h_xc2r, h_xc2r_hat);
    host_exec.fence();
#endif
  }
  Kokkos::finalize();

  return 0;
}
