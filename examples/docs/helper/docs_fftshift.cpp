// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of fftshift usage in documentation
/// freqs = KokkosFFT::fftfreq(10, 0.1);
/// freqs: [0 1 2 3 4 5 -4 -3 -2 -1]
/// KokkosFFT::fftshift(freqs);
/// freqs_shifted: [-5 -4 -3 -2 -1 0 1 2 3 4]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  const int n0 = 10;
  ExecutionSpace exec;
  auto freq = KokkosFFT::fftfreq(exec, n0, 0.1);
  KokkosFFT::fftshift(exec, freq);
  auto h_freq = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, freq);
  for (int i = 0; i < freq.extent_int(0); ++i) {
    std::cout << " " << h_freq(i);
  }
  std::cout << std::endl;

  return 0;
}
