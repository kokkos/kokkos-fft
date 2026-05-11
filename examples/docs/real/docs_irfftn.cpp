// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of irfftn usage in documentation
/// x_hat = [[[78, -6j],
///           [-12, 0]],
///          [[-24+13.8564j, 0],
///           [0, 0]],
///          [[-24-13.8564j, 0],
///           [0, 0]]]
/// x = [[[1, 2],
///       [3, 4]],
///      [[5, 6],
///       [7, 8]],
///      [[9, 10],
///       [11, 12]]]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View3D = Kokkos::View<double***, Kokkos::LayoutRight, ExecutionSpace>;
  using ComplexView3D = Kokkos::View<Kokkos::complex<double>***,
                                     Kokkos::LayoutRight, ExecutionSpace>;

  const int n0 = 3, n1 = 2, n2 = 2;

  ComplexView3D x_hat("x_hat", n0, n1, n2 / 2 + 1);
  View3D x("x", n0, n1, n2);
  auto h_x_hat     = Kokkos::create_mirror_view(x_hat);
  h_x_hat(0, 0, 0) = Kokkos::complex<double>(78, 0);
  h_x_hat(0, 0, 1) = Kokkos::complex<double>(-6, 0);
  h_x_hat(0, 1, 0) = Kokkos::complex<double>(-12, 0);
  h_x_hat(1, 0, 0) = Kokkos::complex<double>(-24, 13.8564);
  h_x_hat(2, 0, 0) = Kokkos::complex<double>(-24, -13.8564);
  Kokkos::deep_copy(x_hat, h_x_hat);

  ExecutionSpace exec;
  KokkosFFT::irfftn(exec, x_hat, x);

  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2; ++k) {
        std::cout << " " << h_x(i, j, k);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
