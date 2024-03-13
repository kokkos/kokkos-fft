// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_TRANSFORM_HPP
#define KOKKOSFFT_SYCL_TRANSFORM_HPP

#include <oneapi/mkl/dfti.hpp>
#include <complex>

namespace KokkosFFT {
namespace Impl {
template <typename PlanType, typename... Args>
void _exec(PlanType& plan, float* idata, std::complex<float>* odata,
           [[maybe_unused]] int direction, [[maybe_unused]] Args... args) {
  [[maybe_unused]] auto r2c = oneapi::mkl::dft::compute_forward(plan, idata,
                                               reinterpret_cast<float*>(odata));
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, double* idata, std::complex<double>* odata,
           [[maybe_unused]] int direction, [[maybe_unused]] Args... args) {
  [[maybe_unused]] auto d2z = oneapi::mkl::dft::compute_forward(
      plan, idata, reinterpret_cast<double*>(odata));
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, std::complex<float>* idata, float* odata,
           [[maybe_unused]] int direction, [[maybe_unused]] Args... args) {
  [[maybe_unused]] auto c2r = oneapi::mkl::dft::compute_backward(
      plan, reinterpret_cast<float*>(idata), odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, std::complex<double>* idata, double* odata,
           [[maybe_unused]] int direction, [[maybe_unused]] Args... args) {
  [[maybe_unused]] auto z2d = oneapi::mkl::dft::compute_backward(
      plan, reinterpret_cast<double*>(idata), odata);
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, std::complex<float>* idata,
           std::complex<float>* odata, [[maybe_unused]] int direction,
           [[maybe_unused]] Args... args) {
  if (direction == 1) {
    [[maybe_unused]] auto c2c = oneapi::mkl::dft::compute_forward(plan, idata, odata);
  } else {
    [[maybe_unused]] auto c2c = oneapi::mkl::dft::compute_backward(plan, idata, odata);
  }
}

template <typename PlanType, typename... Args>
void _exec(PlanType& plan, std::complex<double>* idata,
           std::complex<double>* odata, [[maybe_unused]] int direction,
           [[maybe_unused]] Args... args) {
  if (direction == 1) {
    [[maybe_unused]] auto z2z = oneapi::mkl::dft::compute_forward(plan, idata, odata);
  } else {
    [[maybe_unused]] auto z2z = oneapi::mkl::dft::compute_backward(plan, idata, odata);
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif