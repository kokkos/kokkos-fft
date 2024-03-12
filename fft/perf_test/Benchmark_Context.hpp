// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_BENCHMARK_CONTEXT_HPP
#define KOKKOSFFT_BENCHMARK_CONTEXT_HPP

#include <cstdlib>
#include <string>

#include <benchmark/benchmark.h>

#include <Kokkos_Core.hpp>
#include "KokkosFFT_PrintConfiguration.hpp"
#include <KokkosFFT_Version_Info.hpp>

namespace KokkosFFTBenchmark {
/// \brief Remove unwanted spaces and colon signs from input string. In case of
/// invalid input it will return an empty string.
inline std::string remove_unwanted_characters(std::string str) {
  auto from = str.find_first_not_of(" :");
  auto to   = str.find_last_not_of(" :");

  if (from == std::string::npos || to == std::string::npos) {
    return "";
  }

  // return extracted part of string without unwanted spaces and colon signs
  return str.substr(from, to + 1);
}

/// \brief Extract all key:value pairs from kokkos configuration and add it to
/// the benchmark context
inline void add_kokkos_configuration(bool verbose) {
  std::ostringstream msg;
  Kokkos::print_configuration(msg, verbose);
  KokkosFFT::print_configuration(msg);

  // Iterate over lines returned from kokkos and extract key:value pairs
  std::stringstream ss{msg.str()};
  for (std::string line; std::getline(ss, line, '\n');) {
    auto found = line.find_first_of(':');
    if (found != std::string::npos) {
      auto val = remove_unwanted_characters(line.substr(found + 1));
      // Ignore line without value, for example a category name
      if (!val.empty()) {
        benchmark::AddCustomContext(
            remove_unwanted_characters(line.substr(0, found)), val);
      }
    }
  }
}

/// \brief Add Kokkos Kernels git info and google benchmark release to
/// benchmark context.
inline void add_version_info() {
  using namespace KokkosFFT::Impl;

  if (!GIT_BRANCH.empty()) {
    benchmark::AddCustomContext("GIT_BRANCH", std::string(GIT_BRANCH));
    benchmark::AddCustomContext("GIT_COMMIT_HASH",
                                std::string(GIT_COMMIT_HASH));
    benchmark::AddCustomContext("GIT_CLEAN_STATUS",
                                std::string(GIT_CLEAN_STATUS));
    benchmark::AddCustomContext("GIT_COMMIT_DESCRIPTION",
                                std::string(GIT_COMMIT_DESCRIPTION));
    benchmark::AddCustomContext("GIT_COMMIT_DATE",
                                std::string(GIT_COMMIT_DATE));
  }
}

inline void add_env_info() {
  auto num_threads = std::getenv("OMP_NUM_THREADS");
  if (num_threads) {
    benchmark::AddCustomContext("OMP_NUM_THREADS", num_threads);
  }
  auto dynamic = std::getenv("OMP_DYNAMIC");
  if (dynamic) {
    benchmark::AddCustomContext("OMP_DYNAMIC", dynamic);
  }
  auto proc_bind = std::getenv("OMP_PROC_BIND");
  if (proc_bind) {
    benchmark::AddCustomContext("OMP_PROC_BIND", proc_bind);
  }
  auto places = std::getenv("OMP_PLACES");
  if (places) {
    benchmark::AddCustomContext("OMP_PLACES", places);
  }
}

/// \brief Gather all context information and add it to benchmark context
inline void add_benchmark_context(bool verbose = false) {
  add_kokkos_configuration(verbose);
  add_version_info();
  add_env_info();
}

/**
 * \brief Report throughput and amount of data processed for simple View
 * operations
 */
template <typename InViewType, typename OutViewType>
void report_results(benchmark::State& state, InViewType in, OutViewType out,
                    double time) {
  // data processed in megabytes
  const double in_data_processed =
      static_cast<double>(in.size() * sizeof(typename InViewType::value_type)) /
      1.0e6;
  const double out_data_processed =
      static_cast<double>(out.size() *
                          sizeof(typename OutViewType::value_type)) /
      1.0e6;

  state.SetIterationTime(time);
  state.counters["MB (In)"]  = benchmark::Counter(in_data_processed);
  state.counters["MB (Out)"] = benchmark::Counter(out_data_processed);
  state.counters["GB/s"] =
      benchmark::Counter((in_data_processed + out_data_processed) / 1.0e3,
                         benchmark::Counter::kIsIterationInvariantRate);
}

}  // namespace KokkosFFTBenchmark

#endif