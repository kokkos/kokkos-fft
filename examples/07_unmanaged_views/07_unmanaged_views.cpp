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

template <typename T>
using View3D = Kokkos::View<T***, execution_space>;

template <typename T>
using UView3D = Kokkos::View<T***, execution_space,
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

    // any combination of inputs/outputs can be managed or unmanaged views
    // should work on all functions in any number of dimensions
    // 3D C2C FFT (Forward and Backward)

    // combined storage buffer for xc2c and xc2c_inv
    View1D<Kokkos::complex<double>> storage(
        "storage", (UView3D<Kokkos::complex<double>>::required_allocation_size(
                        n0, n1, n2) +
                    sizeof(Kokkos::complex<double>)) /
                       sizeof(Kokkos::complex<double>) * 2);
    UView3D<Kokkos::complex<double>> xc2c(storage.data(), n0, n1, n2);
    View3D<Kokkos::complex<double>> xc2c_hat("xc2c_hat", n0, n1, n2);
    UView3D<Kokkos::complex<double>> xc2c_inv(
        storage.data() +
            (UView3D<Kokkos::complex<double>>::required_allocation_size(n0, n1,
                                                                        n2) +
             sizeof(Kokkos::complex<double>)) /
                sizeof(Kokkos::complex<double>),
        n0, n1, n2);
    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, xc2c, random_pool, I);
    exec.fence();

    KokkosFFT::fftn(exec, xc2c, xc2c_hat, axis_type<3>{-3, -2, -1},
                    KokkosFFT::Normalization::backward, shape);
    KokkosFFT::ifftn(exec, xc2c_hat, xc2c_inv, axis_type<3>{-3, -2, -1},
                     KokkosFFT::Normalization::backward, shape);
    exec.fence();
  }
  Kokkos::finalize();

  return 0;
}
