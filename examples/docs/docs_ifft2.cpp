// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of ifft2 usage in documentation
/// x = [[78, -6+3.46410162j, -6-3.46410162j],
///      [-18+18j, 0, 0],
///      [-18, 0, 0],
///      [-18-18j, 0, 0]]
/// x_hat = [[1, 2, 3],
///          [4, 5, 6],
///          [7, 8, 9],
///          [10, 11, 12]]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View2D         = Kokkos::View<Kokkos::complex<double>**, Kokkos::LayoutRight, ExecutionSpace>;

  const int n0 = 4, n1 = 3;

  View2D x("x", n0, n1), x_hat("x_hat", n0, n1);
  auto h_x = Kokkos::create_mirror_view(x);
  h_x(0, 0) = Kokkos::complex<double>(78, 0);
  h_x(0, 1) = Kokkos::complex<double>(-6, 3.46410162);
  h_x(0, 2) = Kokkos::complex<double>(-6, -3.46410162);
  h_x(1, 0) = Kokkos::complex<double>(-18, 18);
  h_x(2, 0) = Kokkos::complex<double>(-18, 0);
  h_x(3, 0) = Kokkos::complex<double>(-18, -18);
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::ifft2(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      std::cout << h_x_hat(i, j) << std::endl;
    }
  }

  return 0;
}
