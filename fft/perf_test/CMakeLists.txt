# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

set(BENCHMARK_NAME PerformanceTest_Benchmark)

check_git_version()

add_executable(${BENCHMARK_NAME} BenchmarkMain.cpp PerfTest_FFT1.cpp PerfTest_FFT2.cpp)

target_link_libraries(${BENCHMARK_NAME} PUBLIC benchmark::benchmark KokkosFFT::fft)
target_include_directories(${BENCHMARK_NAME} PUBLIC ${CMAKE_BINARY_DIR})

string(TIMESTAMP BENCHMARK_TIME "%Y-%m-%d_T%H-%M-%S" UTC)
set(BENCHMARK_ARGS --benchmark_counters_tabular=true --benchmark_out=${BENCHMARK_NAME}_${BENCHMARK_TIME}.json)

add_test(NAME ${BENCHMARK_NAME} COMMAND ${BENCHMARK_NAME} ${BENCHMARK_ARGS})
