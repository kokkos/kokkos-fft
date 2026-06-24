// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of ifftshift usage in documentation
/// freqs: [-4 -3 -2 -1 0 1 2 3 4]
/// KokkosFFT::ifftshift(freqs);
/// freqs_shifted: [0 1 2 3 4 -4 -3 -2 -1]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using ViewType       = Kokkos::View<double*, ExecutionSpace>;

  ExecutionSpace exec;
  ViewType freq_shifted("freq_shifted", 9);
  auto h_freq_shifted = Kokkos::create_mirror_view(freq_shifted);
  for (int i = 0; i < freq_shifted.extent_int(0); ++i) {
    h_freq_shifted(i) = static_cast<double>(i - 4);
  }
  Kokkos::deep_copy(freq_shifted, h_freq_shifted);

  KokkosFFT::ifftshift(exec, freq_shifted);
  Kokkos::deep_copy(h_freq_shifted, freq_shifted);
  for (int i = 0; i < h_freq_shifted.extent_int(0); ++i) {
    std::cout << " " << h_freq_shifted(i);
  }
  std::cout << std::endl;

  return 0;
}
