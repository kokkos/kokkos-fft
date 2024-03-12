// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include "PerfTest_FFT2.hpp"

namespace KokkosFFTBenchmark {

// 2D FFT on 2D View
BENCHMARK(FFT2_2DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(FFT2_2DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(FFT2_2DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(FFT2_2DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// 2D IFFT on 2D View
BENCHMARK(IFFT2_2DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IFFT2_2DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IFFT2_2DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IFFT2_2DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// 2D RFFT on 2D View
BENCHMARK(RFFT2_2DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(RFFT2_2DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(RFFT2_2DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(RFFT2_2DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

// 2D IRFFT on 2D View
BENCHMARK(IRFFT2_2DView<float, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IRFFT2_2DView<float, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IRFFT2_2DView<double, Kokkos::LayoutLeft>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(IRFFT2_2DView<double, Kokkos::LayoutRight>)
    ->ArgName("N")
    ->RangeMultiplier(2)
    ->Range(256, 4096)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace KokkosFFTBenchmark