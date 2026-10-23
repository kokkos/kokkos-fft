#include "PerfTest_Utils.hpp"
#include <Kokkos_Core.hpp>
#include <benchmark/benchmark.h>

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    benchmark::Initialize(&argc, argv);
    benchmark::SetDefaultTimeUnit(benchmark::kMicrosecond);
    BatchedFFTBenchmark::add_benchmark_context();
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
  }
  Kokkos::finalize();
  return 0;
}
