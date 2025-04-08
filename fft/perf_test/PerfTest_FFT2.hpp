// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_PERFTEST_FFT2_HPP
#define KOKKOSFFT_PERFTEST_FFT2_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"

#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
using execution_space = Kokkos::DefaultExecutionSpace;
#else
using execution_space = Kokkos::DefaultHostExecutionSpace;
#endif
using axis_type = KokkosFFT::axis_type<2>;

namespace KokkosFFTBenchmark {

template <typename InViewType, typename OutViewType>
void fft2(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::fft2(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename InViewType, typename OutViewType>
void ifft2(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::ifft2(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename InViewType, typename OutViewType>
void rfft2(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::rfft2(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename InViewType, typename OutViewType>
void irfft2(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::irfft2(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename T, typename LayoutType>
static void FFT2_2DView(benchmark::State& state) {
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  const int n = state.range(0);
  ComplexView2DType x("x", n, n), x_hat("x_hat", n, n);

  fft2(x, x_hat, state);
}

template <typename T, typename LayoutType>
static void IFFT2_2DView(benchmark::State& state) {
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  const int n = state.range(0);
  ComplexView2DType x("x", n, n), x_hat("x_hat", n, n);

  ifft2(x, x_hat, state);
}

template <typename T, typename LayoutType>
static void RFFT2_2DView(benchmark::State& state) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  const int n = state.range(0);
  RealView2DType x("x", n, n);
  ComplexView2DType x_hat("x_hat", n, n / 2 + 1);

  rfft2(x, x_hat, state);
}

template <typename T, typename LayoutType>
static void IRFFT2_2DView(benchmark::State& state) {
  using RealView2DType = Kokkos::View<T**, LayoutType, execution_space>;
  using ComplexView2DType =
      Kokkos::View<Kokkos::complex<T>**, LayoutType, execution_space>;

  const int n = state.range(0);
  ComplexView2DType x("x", n, n / 2 + 1);
  RealView2DType x_hat("x_hat", n, n);

  irfft2(x, x_hat, state);
}

}  // namespace KokkosFFTBenchmark

#endif
