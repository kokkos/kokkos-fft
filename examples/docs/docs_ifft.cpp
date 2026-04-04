// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of ifft usage in documentation
/// x = [10, -2+2j, -2, -2-2j]
/// x_hat = [1, 2, 3, 4]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View1D         = Kokkos::View<Kokkos::complex<double>*, ExecutionSpace>;

  const int n0 = 4;

  View1D x("x", n0), x_hat("x_hat", n0);
  auto h_x = Kokkos::create_mirror_view(x);
  h_x(0)   = Kokkos::complex<double>(10, 0);
  h_x(1)   = Kokkos::complex<double>(-2, 2);
  h_x(2)   = Kokkos::complex<double>(-2, 0);
  h_x(3)   = Kokkos::complex<double>(-2, -2);
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::ifft(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < n0; ++i) {
    std::cout << h_x_hat(i) << std::endl;
  }

  return 0;
}
