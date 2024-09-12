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
using View2D = Kokkos::View<T**, execution_space>;

template <typename T>
using View3D = Kokkos::View<T***, execution_space>;

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

    View1D<Kokkos::complex<double> > xc("xc", n0);
    View2D<Kokkos::complex<double> > xc2("xc2", n0, n1);
    View1D<double> xr("xr", n0);
    View2D<double> xr2("xr2", n0, n1);

    Kokkos::Random_XorShift64_Pool<> random_pool(12345);
    execution_space exec;
    Kokkos::fill_random(exec, xc, random_pool, I);
    exec.fence();

#if 0
    // Compilte time error for inconsistent types
    // You will get the following compilation errors if you uncomment this block
    // error: static assertion failed with "rfft: InViewType must be real"
    // error: static assertion failed with "rfft: OutViewType must be complex"
    KokkosFFT::rfft(exec, xc, xr); // Incorrect, input is complex and output is real
    KokkosFFT::rfft(exec, xr, xc); // Correct, input is real and output is complex
#endif

#if 0
    // Compilte time error if FFT rank > View rank (2D FFT on 1D View)
    // You will get the following compilation errors if you uncomment this block
    // error: static assertion failed with "rfft2: View rank must be larger than or equal to 2"
    KokkosFFT::rfft2(exec, xr, xc); // Incorrect, input and output are 1D Views
    KokkosFFT::rfft2(exec, xr2, xc2); // Correct, input and output are 2D Views
#endif

#if 0
    // Compilte time error if FFT plan and execution is inconsistent
    // You will get the following compilation errors if you uncomment this block
    // error: static assertion failed with "Plan::good: InViewType for plan and execution are not identical."
    // error: static assertion failed with "Plan::good: OutViewType for plan and execution are not identical."
    int axis = -1;
    KokkosFFT::Impl::Plan rfft_plan(exec, xr, xc,
                                    KokkosFFT::Direction::forward, axis);
    KokkosFFT::Impl::fft_exec_impl(rfft_plan, xc, xr); // Incorrect, input and output are reversed
    KokkosFFT::Impl::fft_exec_impl(rfft_plan, xr, xc); // Correct, same input and output
#endif

    exec.fence();
  }
  Kokkos::finalize();

  return 0;
}
