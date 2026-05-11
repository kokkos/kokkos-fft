// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of rfftn usage in documentation
/// x = [[[1, 2],
///       [3, 4]],
///      [[5, 6],
///       [7, 8]],
///      [[9, 10],
///       [11, 12]]]
/// x_hat = [[[78, -6],
///           [-12, 0]],
///          [[-24+13.8564j, 0],
///           [0, 0]],
///          [[-24-13.8564j, 0],
///           [0, 0]]]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View3D = Kokkos::View<double***, Kokkos::LayoutRight, ExecutionSpace>;
  using ComplexView3D = Kokkos::View<Kokkos::complex<double>***,
                                     Kokkos::LayoutRight, ExecutionSpace>;

  const int n0 = 3, n1 = 2, n2 = 2;

  View3D x("x", n0, n1, n2);
  ComplexView3D x_hat("x_hat", n0, n1, n2 / 2 + 1);
  auto h_x = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2; ++k) {
        h_x(i, j, k) = i * n1 * n2 + j * n2 + k + 1;
      }
    }
  }
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::rfftn(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      for (int k = 0; k < n2 / 2 + 1; ++k) {
        std::cout << " " << h_x_hat(i, j, k);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  return 0;
}
