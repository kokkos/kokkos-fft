// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR MIT

#ifndef KOKKOSFFT_HIP_TRANSFORM_HPP
#define KOKKOSFFT_HIP_TRANSFORM_HPP

#include <hipfft/hipfft.h>

namespace KokkosFFT {
namespace Impl {
template <typename... Args>
inline void _exec(hipfftHandle& plan, hipfftReal* idata, hipfftComplex* odata,
                  [[maybe_unused]] int direction,
                  [[maybe_unused]] Args... args) {
  hipfftResult hipfft_rt = hipfftExecR2C(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecR2C failed");
}

template <typename... Args>
inline void _exec(hipfftHandle& plan, hipfftDoubleReal* idata,
                  hipfftDoubleComplex* odata, [[maybe_unused]] int direction,
                  [[maybe_unused]] Args... args) {
  hipfftResult hipfft_rt = hipfftExecD2Z(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecD2Z failed");
}

template <typename... Args>
inline void _exec(hipfftHandle& plan, hipfftComplex* idata, hipfftReal* odata,
                  [[maybe_unused]] int direction,
                  [[maybe_unused]] Args... args) {
  hipfftResult hipfft_rt = hipfftExecC2R(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecC2R failed");
}

template <typename... Args>
inline void _exec(hipfftHandle& plan, hipfftDoubleComplex* idata,
                  hipfftDoubleReal* odata, [[maybe_unused]] int direction,
                  [[maybe_unused]] Args... args) {
  hipfftResult hipfft_rt = hipfftExecZ2D(plan, idata, odata);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecZ2D failed");
}

template <typename... Args>
inline void _exec(hipfftHandle& plan, hipfftComplex* idata,
                  hipfftComplex* odata, int direction,
                  [[maybe_unused]] Args... args) {
  hipfftResult hipfft_rt = hipfftExecC2C(plan, idata, odata, direction);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecC2C failed");
}

template <typename... Args>
inline void _exec(hipfftHandle& plan, hipfftDoubleComplex* idata,
                  hipfftDoubleComplex* odata, int direction,
                  [[maybe_unused]] Args... args) {
  hipfftResult hipfft_rt = hipfftExecZ2Z(plan, idata, odata, direction);
  if (hipfft_rt != HIPFFT_SUCCESS)
    throw std::runtime_error("hipfftExecZ2Z failed");
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif