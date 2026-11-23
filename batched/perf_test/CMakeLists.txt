set(BENCHMARK_NAME PerformanceTest_BatchedFFT)

add_executable(${BENCHMARK_NAME} BenchmarkMain.cpp PerfTest_Batched1D.cpp PerfTest_BatchedND.cpp)

target_link_libraries(${BENCHMARK_NAME} PUBLIC batched benchmark::benchmark)

# Smoke test (registered when ENABLE_TESTS=ON): the smallest size of every
# benchmark, one iteration each, so that CI checks the benchmarks run. Team
# sizes that the backend cannot provide are skipped. Full measurements are run
# by hand: see optimization/scripts/job_A100.sh
add_test(NAME ${BENCHMARK_NAME}_Smoke COMMAND ${BENCHMARK_NAME} --benchmark_filter=/n:8/ --benchmark_min_time=1x)
