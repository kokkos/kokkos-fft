// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include "PerfTest_FFT1.hpp"

namespace KokkosFFTBenchmark {

// 1D FFT on 1D View
BENCHMARK(FFT_1DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(FFT_1DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(FFT_1DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(FFT_1DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// 1D IFFT on 1D View
BENCHMARK(IFFT_1DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IFFT_1DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IFFT_1DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IFFT_1DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// 1D RFFT on 1D View
BENCHMARK(RFFT_1DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(RFFT_1DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(RFFT_1DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(RFFT_1DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// 1D IRFFT on 1D View
BENCHMARK(IRFFT_1DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IRFFT_1DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IRFFT_1DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IRFFT_1DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(4096, 65536)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace KokkosFFTBenchmark
