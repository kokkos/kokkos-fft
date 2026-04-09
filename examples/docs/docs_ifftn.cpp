// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of ifftn usage in documentation
/// x = [[[78, -6],
///       [-12, 0]],
///      [[-24+13.85640646j, 0],
///       [0, 0]],
///      [[-24-13.85640646j, 0],
///       [0, 0]]]
/// x_hat = [[[1, 2],
///           [3, 4]],
///          [[5, 6],
///           [7, 8]],
///          [[9, 10],
///           [11, 12]]]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View3D = Kokkos::View<Kokkos::complex<double>***, Kokkos::LayoutRight,
                              ExecutionSpace>;

  const int n0 = 3, n1 = 2, n2 = 2;

  View3D x("x", n0, n1, n2), x_hat("x_hat", n0, n1, n2);
  auto h_x     = Kokkos::create_mirror_view(x);
  h_x(0, 0, 0) = Kokkos::complex<double>(78, 0);
  h_x(0, 0, 1) = Kokkos::complex<double>(-6, 0);
  h_x(0, 1, 0) = Kokkos::complex<double>(-12, 0);
  h_x(1, 0, 0) = Kokkos::complex<double>(-24, 13.85640646);
  h_x(2, 0, 0) = Kokkos::complex<double>(-24, -13.85640646);
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::ifftn(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2; ++k) {
        std::cout << " " << h_x_hat(i, j, k);
      }
      std::cout << "\n";
    }
    if (i < n0 - 1) std::cout << "\n";
  }
  std::cout << std::endl;

  return 0;
}
