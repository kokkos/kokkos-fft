// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TYPES_HPP
#define KOKKOSFFT_HIP_TYPES_HPP

#include <iostream>
#include <hipfft/hipfft.h>
#include <Kokkos_Abort.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_asserts.hpp"

#if defined(ENABLE_HOST_AND_DEVICE)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

// Check the size of complex type
static_assert(sizeof(hipfftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(hipfftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(hipfftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(hipfftDoubleComplex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

/// \brief A class that wraps hipfft for RAII
struct ScopedHIPfftPlan {
 private:
  hipfftHandle m_plan;

 public:
  ScopedHIPfftPlan(int nx, hipfftType type, int batch) {
    hipfftResult hipfft_rt = hipfftPlan1d(&m_plan, nx, type, batch);
    KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftPlan1d failed");
  }

  ScopedHIPfftPlan(int nx, int ny, hipfftType type) {
    hipfftResult hipfft_rt = hipfftPlan2d(&m_plan, nx, ny, type);
    KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftPlan2d failed");
  }

  ScopedHIPfftPlan(int nx, int ny, int nz, hipfftType type) {
    hipfftResult hipfft_rt = hipfftPlan3d(&m_plan, nx, ny, nz, type);
    KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftPlan3d failed");
  }

  ScopedHIPfftPlan(int rank, int *n, int *inembed, int istride, int idist,
                   int *onembed, int ostride, int odist, hipfftType type,
                   int batch) {
    hipfftResult hipfft_rt =
        hipfftPlanMany(&m_plan, rank, n, inembed, istride, idist, onembed,
                       ostride, odist, type, batch);
    KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftPlanMany failed");
  }

  ~ScopedHIPfftPlan() noexcept { cleanup(); }

  ScopedHIPfftPlan()                                    = delete;
  ScopedHIPfftPlan(const ScopedHIPfftPlan &)            = delete;
  ScopedHIPfftPlan &operator=(const ScopedHIPfftPlan &) = delete;
  ScopedHIPfftPlan &operator=(ScopedHIPfftPlan &&)      = delete;
  ScopedHIPfftPlan(ScopedHIPfftPlan &&)                 = delete;

  hipfftHandle plan() const noexcept { return m_plan; }
  void commit(const Kokkos::HIP &exec_space) {
    hipStream_t stream = exec_space.hip_stream();
    try {
      hipfftResult hipfft_rt = hipfftSetStream(m_plan, stream);
      KOKKOSFFT_THROW_IF(hipfft_rt != HIPFFT_SUCCESS, "hipfftSetStream failed");
    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      cleanup();
      throw;
    }
  }

 private:
  void cleanup() {
    hipfftResult hipfft_rt = hipfftDestroy(m_plan);
    if (hipfft_rt != HIPFFT_SUCCESS) Kokkos::abort("hipfftDestroy failed");
  }
};

#if defined(ENABLE_HOST_AND_DEVICE)
template <typename ExecutionSpace>
struct FFTDataType {
  using float32 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftReal, float>;
  using float64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftDoubleReal, double>;
  using complex64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftComplex, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         hipfftDoubleComplex, fftw_complex>;
};

template <typename ExecutionSpace>
using TransformType =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>, hipfftType,
                       FFTWTransformType>;

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, T1, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using TransformTypeOnExecSpace = TransformType<ExecutionSpace>;

  static constexpr TransformTypeOnExecSpace m_hip_type =
      std::is_same_v<T1, float> ? HIPFFT_R2C : HIPFFT_D2Z;
  static constexpr TransformTypeOnExecSpace m_cpu_type =
      std::is_same_v<T1, float> ? FFTWTransformType::R2C
                                : FFTWTransformType::D2Z;

  static constexpr TransformTypeOnExecSpace type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      return m_hip_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using TransformTypeOnExecSpace = TransformType<ExecutionSpace>;

  static constexpr TransformTypeOnExecSpace m_hip_type =
      std::is_same_v<T1, float> ? HIPFFT_C2R : HIPFFT_Z2D;
  static constexpr TransformTypeOnExecSpace m_cpu_type =
      std::is_same_v<T1, float> ? FFTWTransformType::C2R
                                : FFTWTransformType::Z2D;

  static constexpr TransformTypeOnExecSpace type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      return m_hip_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  using TransformTypeOnExecSpace = TransformType<ExecutionSpace>;

  static constexpr TransformTypeOnExecSpace m_hip_type =
      std::is_same_v<T1, float> ? HIPFFT_C2C : HIPFFT_Z2Z;
  static constexpr TransformTypeOnExecSpace m_cpu_type =
      std::is_same_v<T1, float> ? FFTWTransformType::C2C
                                : FFTWTransformType::Z2Z;

  static constexpr TransformTypeOnExecSpace type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::HIP>) {
      return m_hip_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftw_plan_type   = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using hipfft_plan_type = ScopedHIPfftPlan;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                  hipfft_plan_type, fftw_plan_type>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? HIPFFT_FORWARD
                                                  : FFTW_FORWARD;
  static constexpr FFTDirectionType BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? HIPFFT_BACKWARD
                                                  : FFTW_BACKWARD;
  return direction == Direction::forward ? FORWARD : BACKWARD;
}
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = hipfftReal;
  using float64    = hipfftDoubleReal;
  using complex64  = hipfftComplex;
  using complex128 = hipfftDoubleComplex;
};

template <typename ExecutionSpace>
using TransformType = hipfftType;

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, T1, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr hipfftType m_type =
      std::is_same_v<T1, float> ? HIPFFT_R2C : HIPFFT_D2Z;
  static constexpr hipfftType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr hipfftType m_type =
      std::is_same_v<T2, float> ? HIPFFT_C2R : HIPFFT_Z2D;
  static constexpr hipfftType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr hipfftType m_type =
      std::is_same_v<T1, float> ? HIPFFT_C2C : HIPFFT_Z2Z;
  static constexpr hipfftType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = ScopedHIPfftPlan;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
