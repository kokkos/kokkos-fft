// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of hfft usage in documentation
/// x = [1, 2, 3, 4]
/// x_hat = [15.0, -4.0, 0.0, -1.0, 0.0, -4.0]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View1D         = Kokkos::View<double*, ExecutionSpace>;

  const int n0 = 4;

  View1D x("x", n0), x_hat("x_hat", 2 * (n0 - 1));
  auto h_x = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n0; ++i) {
    h_x(i) = i + 1;
  }
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::hfft(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < x_hat.extent_int(0); ++i) {
    std::cout << " " << h_x_hat(i);
  }
  std::cout << std::endl;

  return 0;
}
