#ifndef KOKKOSFFT_SYCL_TRANSFORM_HPP
#define KOKKOSFFT_SYCL_TRANSFORM_HPP

#include <oneapi/mkl/dfti.hpp>
#include <complex>

namespace KokkosFFT {
namespace Impl {
template <typename PlanType>
void _exec(PlanType& plan, float* idata, std::complex<float>* odata,
           [[maybe_unused]] int direction) {
  auto r2c = oneapi::mkl::dft::compute_forward(plan, idata,
                                               reinterpret_cast<float*>(odata));
  r2c.wait();
}

template <typename PlanType>
void _exec(PlanType& plan, double* idata, std::complex<double>* odata,
           [[maybe_unused]] int direction) {
  auto d2z = oneapi::mkl::dft::compute_forward(
      plan, idata, reinterpret_cast<double*>(odata));
  d2z.wait();
}

template <typename PlanType>
void _exec(PlanType& plan, std::complex<float>* idata, float* odata,
           [[maybe_unused]] int direction) {
  auto c2r = oneapi::mkl::dft::compute_backward(
      plan, reinterpret_cast<float*>(idata), odata);
  c2r.wait();
}

template <typename PlanType>
void _exec(PlanType& plan, std::complex<double>* idata, double* odata,
           [[maybe_unused]] int direction) {
  auto z2d = oneapi::mkl::dft::compute_backward(
      plan, reinterpret_cast<double*>(idata), odata);
  z2d.wait();
}

template <typename PlanType>
void _exec(PlanType& plan, std::complex<float>* idata,
           std::complex<float>* odata, [[maybe_unused]] int direction) {
  if (direction == 1) {
    auto c2c = oneapi::mkl::dft::compute_forward(plan, idata, odata);
    c2c.wait();
  } else {
    auto c2c = oneapi::mkl::dft::compute_backward(plan, idata, odata);
    c2c.wait();
  }
}

template <typename PlanType>
void _exec(PlanType& plan, std::complex<double>* idata,
           std::complex<double>* odata, [[maybe_unused]] int direction) {
  if (direction == 1) {
    auto z2z = oneapi::mkl::dft::compute_forward(plan, idata, odata);
    z2z.wait();
  } else {
    auto z2z = oneapi::mkl::dft::compute_backward(plan, idata, odata);
    z2z.wait();
  }
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif