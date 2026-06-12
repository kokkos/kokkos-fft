// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of ihfft usage in documentation
/// x_hat = [15.0, -4.0, 0.0, -1.0, 0.0, -4.0]
/// x = [1.-0.j, 2.-0.j, 3.-0.j, 4.-0.j]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View1D         = Kokkos::View<double*, ExecutionSpace>;
  using ComplexView1D  = Kokkos::View<Kokkos::complex<double>*, ExecutionSpace>;

  const int n0 = 4;

  View1D x_hat("x_hat", 2 * (n0 - 1));
  ComplexView1D x("x", n0);
  auto h_x_hat = Kokkos::create_mirror_view(x_hat);
  h_x_hat(0)   = 15.0;
  h_x_hat(1)   = -4.0;
  h_x_hat(2)   = 0.0;
  h_x_hat(3)   = -1.0;
  h_x_hat(4)   = 0.0;
  h_x_hat(5)   = -4.0;
  Kokkos::deep_copy(x_hat, h_x_hat);

  ExecutionSpace exec;
  KokkosFFT::ihfft(exec, x_hat, x);

  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  for (int i = 0; i < x.extent_int(0); ++i) {
    std::cout << " " << h_x(i);
  }
  std::cout << std::endl;

  return 0;
}
