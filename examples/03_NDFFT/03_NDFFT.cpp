// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View3D = Kokkos::View<T***, execution_space>;
template <std::size_t DIM>
using axis_type = KokkosFFT::axis_type<DIM>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int n0 = 128, n1 = 128, n2 = 16;
    const Kokkos::complex<double> z(1.0, 1.0);

    // 3D C2C FFT (Forward and Backward)
    View3D<Kokkos::complex<double> > xc2c("xc2c", n0, n1, n2);
    View3D<Kokkos::complex<double> > xc2c_hat("xc2c_hat", n0, n1, n2);
    View3D<Kokkos::complex<double> > xc2c_inv("xc2c_inv", n0, n1, n2);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, xc2c, random_pool, z);
    exec.fence();

    KokkosFFT::fftn(exec, xc2c, xc2c_hat, axis_type<3>{-3, -2, -1});
    KokkosFFT::ifftn(exec, xc2c_hat, xc2c_inv, axis_type<3>{-3, -2, -1});
    exec.fence();

    // 3D R2C FFT
    View3D<double> xr2c("xr2c", n0, n1, n2);
    View3D<Kokkos::complex<double> > xr2c_hat("xr2c_hat", n0, n1, n2 / 2 + 1);
    Kokkos::fill_random(exec, xr2c, random_pool, 1);
    exec.fence();

    KokkosFFT::rfftn(exec, xr2c, xr2c_hat, axis_type<3>{-3, -2, -1});
    exec.fence();

    // 3D C2R FFT
    View3D<Kokkos::complex<double> > xc2r("xr2c_hat", n0, n1, n2 / 2 + 1);
    View3D<double> xc2r_hat("xc2r", n0, n1, n2);
    Kokkos::fill_random(exec, xc2r, random_pool, z);
    exec.fence();

    KokkosFFT::irfftn(exec, xc2r, xc2r_hat, axis_type<3>{-3, -2, -1});
    exec.fence();
  }
  Kokkos::finalize();

  return 0;
}
