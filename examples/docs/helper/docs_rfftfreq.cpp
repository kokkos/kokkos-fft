// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of rfftfreq usage in documentation
/// For n = 9, the output is
/// freq = [0 0.111111 0.222222 0.333333 0.444444]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  const int n0 = 9;
  ExecutionSpace exec;
  auto freq = KokkosFFT::rfftfreq<ExecutionSpace, double>(exec, n0);

  auto h_freq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, freq);
  for (int i = 0; i < freq.extent_int(0); ++i) {
    std::cout << " " << h_freq(i);
  }
  std::cout << std::endl;

  return 0;
}
