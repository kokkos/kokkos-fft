// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_TRANSFORM_HPP
#define KOKKOSFFT_SYCL_TRANSFORM_HPP

#include <complex>
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250100
#include <oneapi/mkl/dft.hpp>
#else
#include <oneapi/mkl/dfti.hpp>
#endif
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Common_Types.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename PlanType>
void exec_plan(PlanType& plan, float* idata, std::complex<float>* odata,
               KokkosFFT::Direction /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_oneMKLExecR2C]");
  oneapi::mkl::dft::compute_forward(plan.plan(), idata,
                                    reinterpret_cast<float*>(odata));
}

template <typename PlanType>
void exec_plan(PlanType& plan, double* idata, std::complex<double>* odata,
               KokkosFFT::Direction /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_oneMKLExecD2Z]");
  oneapi::mkl::dft::compute_forward(plan.plan(), idata,
                                    reinterpret_cast<double*>(odata));
}

template <typename PlanType>
void exec_plan(PlanType& plan, std::complex<float>* idata, float* odata,
               KokkosFFT::Direction /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_oneMKLExecC2R]");
  oneapi::mkl::dft::compute_backward(plan.plan(),
                                     reinterpret_cast<float*>(idata), odata);
}

template <typename PlanType>
void exec_plan(PlanType& plan, std::complex<double>* idata, double* odata,
               KokkosFFT::Direction /*direction*/) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_oneMKLExecZ2D]");
  oneapi::mkl::dft::compute_backward(plan.plan(),
                                     reinterpret_cast<double*>(idata), odata);
}

template <typename PlanType>
void exec_plan(PlanType& plan, std::complex<float>* idata,
               std::complex<float>* odata, KokkosFFT::Direction direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_oneMKLExecC2C]");
  if (direction == KokkosFFT::Direction::forward) {
    oneapi::mkl::dft::compute_forward(plan.plan(), idata, odata);
  } else {
    oneapi::mkl::dft::compute_backward(plan.plan(), idata, odata);
  }
}

template <typename PlanType>
void exec_plan(PlanType& plan, std::complex<double>* idata,
               std::complex<double>* odata, KokkosFFT::Direction direction) {
  Kokkos::Profiling::ScopedRegion region(
      "KokkosFFT::exec_plan[TPL_oneMKLExecZ2Z]");
  if (direction == KokkosFFT::Direction::forward) {
    oneapi::mkl::dft::compute_forward(plan.plan(), idata, odata);
  } else {
    oneapi::mkl::dft::compute_backward(plan.plan(), idata, odata);
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif
