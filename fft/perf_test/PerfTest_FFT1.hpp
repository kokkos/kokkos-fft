// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_PERFTEST_FFT1_HPP
#define KOKKOSFFT_PERFTEST_FFT1_HPP

#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>
#include <benchmark/benchmark.h>
#include "Benchmark_Context.hpp"

#if defined(KOKKOSFFT_HAS_DEVICE_TPL)
using execution_space = Kokkos::DefaultExecutionSpace;
#else
using execution_space = Kokkos::DefaultHostExecutionSpace;
#endif

namespace KokkosFFTBenchmark {

template <typename InViewType, typename OutViewType>
void fft(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::fft(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename InViewType, typename OutViewType>
void ifft(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::ifft(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename InViewType, typename OutViewType>
void rfft(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::rfft(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename InViewType, typename OutViewType>
void irfft(const InViewType& in, OutViewType& out, benchmark::State& state) {
  for (auto _ : state) {
    Kokkos::fence();
    Kokkos::Timer timer;
    KokkosFFT::irfft(execution_space(), in, out);
    KokkosFFTBenchmark::report_results(state, in, out, timer.seconds());
  }
}

template <typename T, typename LayoutType>
static void FFT_1DView(benchmark::State& state) {
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  const int n = state.range(0);
  ComplexView1DType x("x", n), x_hat("x_hat", n);

  fft(x, x_hat, state);
}

template <typename T, typename LayoutType>
static void IFFT_1DView(benchmark::State& state) {
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  const int n = state.range(0);
  ComplexView1DType x("x", n), x_hat("x_hat", n);

  ifft(x, x_hat, state);
}

template <typename T, typename LayoutType>
static void RFFT_1DView(benchmark::State& state) {
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  const int n = state.range(0);
  RealView1DType x("x", n);
  ComplexView1DType x_hat("x_hat", n / 2 + 1);

  rfft(x, x_hat, state);
}

template <typename T, typename LayoutType>
static void IRFFT_1DView(benchmark::State& state) {
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  const int n = state.range(0);
  ComplexView1DType x("x", n / 2 + 1);
  RealView1DType x_hat("x_hat", n);

  irfft(x, x_hat, state);
}

}  // namespace KokkosFFTBenchmark

#endif
