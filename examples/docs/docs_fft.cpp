// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

/// \brief Example of fft usage in documentation
/// x = [1, 2, 3, 4]
/// x_hat = [10, -2+2j, -2, -2-2j]
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using View1D         = Kokkos::View<Kokkos::complex<double>*, ExecutionSpace>;

  const int n0 = 4;

  View1D x("x", n0), x_hat("x_hat", n0);
  auto h_x = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n0; ++i) {
    h_x(i) = Kokkos::complex<double>(i + 1);
  }
  Kokkos::deep_copy(x, h_x);

  ExecutionSpace exec;
  KokkosFFT::fft(exec, x, x_hat);

  auto h_x_hat =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, x_hat);
  for (int i = 0; i < n0; ++i) {
    std::cout << h_x_hat(i) << std::endl;
  }

  return 0;
}
