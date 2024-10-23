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
void exec_plan(PlanType& plan, float* idata, std::complex<float>* odata,
               int /*direction*/, Args...) {
  oneapi::mkl::dft::compute_forward(plan, idata,
                                    reinterpret_cast<float*>(odata));
}

template <typename PlanType, typename... Args>
void exec_plan(PlanType& plan, double* idata, std::complex<double>* odata,
               int /*direction*/, Args...) {
  oneapi::mkl::dft::compute_forward(plan, idata,
                                    reinterpret_cast<double*>(odata));
}

template <typename PlanType, typename... Args>
void exec_plan(PlanType& plan, std::complex<float>* idata, float* odata,
               int /*direction*/, Args...) {
  oneapi::mkl::dft::compute_backward(plan, reinterpret_cast<float*>(idata),
                                     odata);
}

template <typename PlanType, typename... Args>
void exec_plan(PlanType& plan, std::complex<double>* idata, double* odata,
               int /*direction*/, Args...) {
  oneapi::mkl::dft::compute_backward(plan, reinterpret_cast<double*>(idata),
                                     odata);
}

template <typename PlanType, typename... Args>
void exec_plan(PlanType& plan, std::complex<float>* idata,
               std::complex<float>* odata, int direction, Args...) {
  if (direction == 1) {
    oneapi::mkl::dft::compute_forward(plan, idata, odata);
  } else {
    oneapi::mkl::dft::compute_backward(plan, idata, odata);
  }
}

template <typename PlanType, typename... Args>
void exec_plan(PlanType& plan, std::complex<double>* idata,
               std::complex<double>* odata, int direction, Args...) {
  if (direction == 1) {
    oneapi::mkl::dft::compute_forward(plan, idata, odata);
  } else {
    oneapi::mkl::dft::compute_backward(plan, idata, odata);
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
