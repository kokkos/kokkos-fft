// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View3D = Kokkos::View<T***, execution_space>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int n0 = 128, n1 = 128, n2 = 16;
    const Kokkos::complex<double> I(1.0, 1.0);

    // 1D batched C2C FFT (Forward and Backward)
    View3D<Kokkos::complex<double> > xc2c("xc2c", n0, n1, n2);
    View3D<Kokkos::complex<double> > xc2c_hat("xc2c_hat", n0, n1, n2);
    View3D<Kokkos::complex<double> > xc2c_inv("xc2c_inv", n0, n1, n2);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    Kokkos::fill_random(xc2c, random_pool, I);

    KokkosFFT::fft(execution_space(), xc2c, xc2c_hat,
                   KokkosFFT::Normalization::backward, /*axis=*/-1);
    KokkosFFT::ifft(execution_space(), xc2c_hat, xc2c_inv,
                    KokkosFFT::Normalization::backward, /*axis=*/-1);

    // 1D batched R2C FFT
    View3D<double> xr2c("xr2c", n0, n1, n2);
    View3D<Kokkos::complex<double> > xr2c_hat("xr2c_hat", n0, n1, n2 / 2 + 1);
    Kokkos::fill_random(xr2c, random_pool, 1);

    KokkosFFT::rfft(execution_space(), xr2c, xr2c_hat,
                    KokkosFFT::Normalization::backward, /*axis=*/-1);

    // 1D batched C2R FFT
    View3D<Kokkos::complex<double> > xc2r("xr2c_hat", n0, n1, n2 / 2 + 1);
    View3D<double> xc2r_hat("xc2r", n0, n1, n2);
    Kokkos::fill_random(xc2r, random_pool, I);

    KokkosFFT::irfft(execution_space(), xc2r, xc2r_hat,
                     KokkosFFT::Normalization::backward, /*axis=*/-1);
  }
  Kokkos::finalize();

  return 0;
}