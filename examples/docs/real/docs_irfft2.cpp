// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of irfft2 usage in documentation
/// x_hat = [[78, -6+6j, -6],
///          [-24+13.8564j, 0, 0],
///          [-24-13.8564j, 0, 0]]
/// x = [[1, 2, 3, 4],
///      [5, 6, 7, 8],
///      [9, 10, 11, 12]]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View2D         = Kokkos::View<double**, ExecutionSpace>;
  using ComplexView2D = Kokkos::View<Kokkos::complex<double>**, ExecutionSpace>;

  const int n0 = 3, n1 = 4;

  ComplexView2D x_hat("x_hat", n0, n1 / 2 + 1);
  View2D x("x", n0, n1);
  auto h_x_hat  = Kokkos::create_mirror_view(x_hat);
  h_x_hat(0, 0) = Kokkos::complex<double>(78, 0);
  h_x_hat(0, 1) = Kokkos::complex<double>(-6, 6);
  h_x_hat(0, 2) = Kokkos::complex<double>(-6, 0);
  h_x_hat(1, 0) = Kokkos::complex<double>(-24, 13.8564);
  h_x_hat(2, 0) = Kokkos::complex<double>(-24, -13.8564);
  Kokkos::deep_copy(x_hat, h_x_hat);

  ExecutionSpace exec;
  KokkosFFT::irfft2(exec, x_hat, x);

  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      std::cout << " " << h_x(i, j);
    }
    std::cout << std::endl;
  }

  return 0;
}
