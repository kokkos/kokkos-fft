// SPDX-FileCopyrightText: (C) 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
// SPDX-FileCopyrightText: Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "Benchmark_Context.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
    benchmark::Initialize(&argc, argv);
    benchmark::SetDefaultTimeUnit(benchmark::kSecond);
    KokkosFFTBenchmark::add_benchmark_context(true);

    benchmark::RunSpecifiedBenchmarks();

    benchmark::Shutdown();
  }
  Kokkos::finalize();
  return 0;
}
