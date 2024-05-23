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

template <typename T>
using UView3D = Kokkos::View<T***, Kokkos::SharedSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <std::size_t DIM>
using axis_type = KokkosFFT::axis_type<DIM>;
template <std::size_t DIM>
using shape_type = KokkosFFT::shape_type<DIM>;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    constexpr int n0 = 128, n1 = 128, n2 = 16;
    const Kokkos::complex<double> I(1.0, 1.0);

    shape_type<3> shape;
    shape[0] = n0;
    shape[1] = n1;
    shape[2] = n2;

    // 3D C2C FFT (Forward and Backward)
    View3D<Kokkos::complex<double>> xc2c("xc2c", n0, n1, n2);
    View3D<Kokkos::complex<double>> xc2c_hat("xc2c_hat", n0, n1, n2);
    View3D<Kokkos::complex<double>> xc2c_inv("xc2c_inv", n0, n1, n2);

    UView3D<Kokkos::complex<double>> uxc2c(xc2c.data(), xc2c.layout());
    UView3D<Kokkos::complex<double>> uxc2c_hat(xc2c_hat.data(),
                                               xc2c_hat.layout());
    UView3D<Kokkos::complex<double>> uxc2c_inv(xc2c_inv.data(),
                                               xc2c_inv.layout());

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, uxc2c, random_pool, I);
    exec.fence();

    KokkosFFT::fftn(exec, uxc2c, uxc2c_hat, axis_type<3>{-3, -2, -1});
    KokkosFFT::ifftn(exec, uxc2c_hat, uxc2c_inv, axis_type<3>{-3, -2, -1});
    exec.fence();

    // 3D R2C FFT
    View3D<double> xr2c("xr2c", n0, n1, n2);
    UView3D<double> uxr2c(xr2c.data(), xr2c.layout());
    View3D<Kokkos::complex<double>> xr2c_hat("xr2c_hat", n0, n1, n2 / 2 + 1);
    View3D<Kokkos::complex<double>> uxr2c_hat(xr2c_hat.data(), n0, n1,
                                              n2 / 2 + 1);
    Kokkos::fill_random(exec, uxr2c, random_pool, 1);
    exec.fence();

    KokkosFFT::rfftn(exec, uxr2c, uxr2c_hat, axis_type<3>{-3, -2, -1},
                     KokkosFFT::Normalization::backward, shape);
    exec.fence();

    // 3D C2R FFT
    View3D<Kokkos::complex<double>> xc2r("xr2c_hat", n0, n1, n2 / 2 + 1);
    UView3D<Kokkos::complex<double>> uxc2r(xc2r.data(), xc2r.layout());
    View3D<double> xc2r_hat("xc2r", n0, n1, n2);
    UView3D<double> uxc2r_hat(xc2r_hat.data(), xc2r_hat.layout());
    Kokkos::fill_random(exec, uxc2r, random_pool, I);
    exec.fence();

    KokkosFFT::irfftn(exec, uxc2r, uxc2r_hat, axis_type<3>{-3, -2, -1},
                      KokkosFFT::Normalization::backward, shape);
    exec.fence();
  }
  Kokkos::finalize();

  return 0;
}
