// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR MIT

#ifndef KOKKOSFFT_ROCM_TRANSFORM_HPP
#define KOKKOSFFT_ROCM_TRANSFORM_HPP

#include <complex>
#include <rocfft/rocfft.h>

namespace KokkosFFT {
namespace Impl {
inline void _exec(rocfft_plan& plan, float* idata, std::complex<float>* odata,
                  [[maybe_unused]] int direction,
                  const rocfft_execution_info& execution_info) {
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  if (status != rocfft_status_success)
    throw std::runtime_error("rocfft_execute for R2C failed");
}

inline void _exec(rocfft_plan& plan, double* idata, std::complex<double>* odata,
                  [[maybe_unused]] int direction,
                  const rocfft_execution_info& execution_info) {
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  if (status != rocfft_status_success)
    throw std::runtime_error("rocfft_execute for D2Z failed");
}

inline void _exec(rocfft_plan& plan, std::complex<float>* idata, float* odata,
                  [[maybe_unused]] int direction,
                  const rocfft_execution_info& execution_info) {
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  if (status != rocfft_status_success)
    throw std::runtime_error("rocfft_execute for C2R failed");
}

inline void _exec(rocfft_plan& plan, std::complex<double>* idata, double* odata,
                  [[maybe_unused]] int direction,
                  const rocfft_execution_info& execution_info) {
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  if (status != rocfft_status_success)
    throw std::runtime_error("rocfft_execute for Z2D failed");
}

inline void _exec(rocfft_plan& plan, std::complex<float>* idata,
                  std::complex<float>* odata, [[maybe_unused]] int direction,
                  const rocfft_execution_info& execution_info) {
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  if (status != rocfft_status_success)
    throw std::runtime_error("rocfft_execute for C2C failed");
}

inline void _exec(rocfft_plan& plan, std::complex<double>* idata,
                  std::complex<double>* odata, [[maybe_unused]] int direction,
                  const rocfft_execution_info& execution_info) {
  rocfft_status status =
      rocfft_execute(plan, (void**)&idata, (void**)&odata, execution_info);
  if (status != rocfft_status_success)
    throw std::runtime_error("rocfft_execute for Z2Z failed");
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif