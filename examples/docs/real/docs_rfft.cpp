// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of rfft usage in documentation
/// x = [1, 2, 3, 4]
/// x_hat = [10, -2+2j, -2]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View1D         = Kokkos::View<double*, ExecutionSpace>;
  using ComplexView1D  = Kokkos::View<Kokkos::complex<double>*, ExecutionSpace>;

  const int n0 = 4;

  View1D x("x", n0);
  ComplexView1D x_hat("x_hat", n0 / 2 + 1);
  auto h_x = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n0; ++i) {
    h_x(i) = i + 1;
  }
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::rfft(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < n0 / 2 + 1; ++i) {
    std::cout << " " << h_x_hat(i);
  }
  std::cout << std::endl;

  return 0;
}
