// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of Plan usage in documentation
/// This corresponds to 1D batched rfft on 2D Views
/// x = [[1, 2, 3, 4],
///      [5, 6, 7, 8],
///      [9, 10, 11, 12]]
/// x_hat = [[10.+0.j, -2.+2.j, -2.+0.j],
///          [26.+0.j, -2.+2.j, -2.+0.j],
///          [42.+0.j, -2.+2.j, -2.+0.j]]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View2D = Kokkos::View<double**, Kokkos::LayoutRight, ExecutionSpace>;
  using ComplexView2D = Kokkos::View<Kokkos::complex<double>**,
                                     Kokkos::LayoutRight, ExecutionSpace>;

  const int n0 = 3, n1 = 4;

  View2D x("x", n0, n1);
  ComplexView2D x_hat("x_hat", n0, n1 / 2 + 1),
      x_hat2("x_hat2", n0, n1 / 2 + 1);
  auto h_x = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n0; ++i) {
    for (int j = 0; j < n1; ++j) {
      h_x(i, j) = i * n1 + j + 1;
    }
  }
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;

  int axis = -1;
  KokkosFFT::Plan plan(exec, x, x_hat, KokkosFFT::Direction::forward, axis);
  KokkosFFT::execute(plan, x, x_hat2);

  auto h_x_hat2 =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat2);

  for (int i = 0; i < h_x_hat2.extent_int(0); ++i) {
    for (int j = 0; j < h_x_hat2.extent_int(1); ++j) {
      std::cout << " " << h_x_hat2(i, j);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
