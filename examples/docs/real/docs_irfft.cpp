// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of irfft usage in documentation
/// x_hat = [10, -2+2j, -2]
/// x = [1, 2, 3, 4]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View1D         = Kokkos::View<double*, ExecutionSpace>;
  using ComplexView1D  = Kokkos::View<Kokkos::complex<double>*, ExecutionSpace>;

  const int n0 = 4;

  ComplexView1D x_hat("x_hat", n0 / 2 + 1);
  View1D x("x", n0);
  auto h_x_hat = Kokkos::create_mirror_view(x_hat);
  h_x_hat(0)   = Kokkos::complex<double>(10, 0);
  h_x_hat(1)   = Kokkos::complex<double>(-2, 2);
  h_x_hat(2)   = Kokkos::complex<double>(-2, 0);
  Kokkos::deep_copy(x_hat, h_x_hat);

  ExecutionSpace exec;
  KokkosFFT::irfft(exec, x_hat, x);

  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x);
  for (int i = 0; i < n0; ++i) {
    std::cout << " " << h_x(i);
  }
  std::cout << std::endl;

  return 0;
}
