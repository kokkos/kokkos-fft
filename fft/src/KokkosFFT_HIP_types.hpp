// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TYPES_HPP
#define KOKKOSFFT_HIP_TYPES_HPP

#include <hipfft/hipfft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_HIP_asserts.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
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
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftPlan1d(&m_plan, nx, type, batch));
  }

  ScopedHIPfftPlan(int nx, int ny, hipfftType type) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftPlan2d(&m_plan, nx, ny, type));
  }

  ScopedHIPfftPlan(int nx, int ny, int nz, hipfftType type) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftPlan3d(&m_plan, nx, ny, nz, type));
  }

  ScopedHIPfftPlan(int rank, int *n, int *inembed, int istride, int idist,
                   int *onembed, int ostride, int odist, hipfftType type,
                   int batch) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftPlanMany(&m_plan, rank, n, inembed,
                                               istride, idist, onembed, ostride,
                                               odist, type, batch));
  }

  ~ScopedHIPfftPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_hipfft]");
    hipfftResult hipfft_rt = hipfftDestroy(m_plan);
    if (hipfft_rt != HIPFFT_SUCCESS) Kokkos::abort("hipfftDestroy failed");
  }

  ScopedHIPfftPlan()                                    = delete;
  ScopedHIPfftPlan(const ScopedHIPfftPlan &)            = delete;
  ScopedHIPfftPlan &operator=(const ScopedHIPfftPlan &) = delete;
  ScopedHIPfftPlan &operator=(ScopedHIPfftPlan &&)      = delete;
  ScopedHIPfftPlan(ScopedHIPfftPlan &&)                 = delete;

  hipfftHandle plan() const noexcept { return m_plan; }
  void commit(const Kokkos::HIP &exec_space) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftSetStream(m_plan, exec_space.hip_stream()));
  }
};

/// \brief A class that wraps hipfft for RAII
struct ScopedHIPfftDynPlan {
 private:
  hipfftHandle m_plan;
  std::size_t m_workspace_size;

 public:
  ScopedHIPfftDynPlan(int nx, hipfftType type, int batch) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftCreate(&m_plan));
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetAutoAllocation(m_plan, 0));
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftMakePlan1d(m_plan, nx, type, batch, &m_workspace_size));
  }

  ScopedHIPfftDynPlan(const std::vector<int> &fft_extents, hipfftType type) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftCreate(&m_plan));
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetAutoAllocation(m_plan, 0));

    if (fft_extents.size() == 2) {
      auto nx = fft_extents.at(0), ny = fft_extents.at(1);
      KOKKOSFFT_CHECK_HIPFFT_CALL(
          hipfftMakePlan2d(m_plan, nx, ny, type, &m_workspace_size));
    } else if (fft_extents.size() == 3) {
      auto nx = fft_extents.at(0), ny = fft_extents.at(1),
           nz = fft_extents.at(2);
      KOKKOSFFT_CHECK_HIPFFT_CALL(
          hipfftMakePlan3d(m_plan, nx, ny, nz, type, &m_workspace_size));
    } else {
      KOKKOSFFT_THROW_IF(true, "FFT dimension can be 2D or 3D only");
    }
  }

  ScopedHIPfftDynPlan(int rank, int *n, int *inembed, int istride, int idist,
                      int *onembed, int ostride, int odist, hipfftType type,
                      int batch) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftCreate(&m_plan));
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetAutoAllocation(m_plan, 0));
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftMakePlanMany(m_plan, rank, n, inembed, istride, idist, onembed,
                           ostride, odist, type, batch, &m_workspace_size));
  }

  ~ScopedHIPfftDynPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_hipfft]");
    hipfftResult hipfft_rt = hipfftDestroy(m_plan);
    if (hipfft_rt != HIPFFT_SUCCESS) Kokkos::abort("hipfftDestroy failed");
  }

  ScopedHIPfftDynPlan()                                       = delete;
  ScopedHIPfftDynPlan(const ScopedHIPfftDynPlan &)            = delete;
  ScopedHIPfftDynPlan &operator=(const ScopedHIPfftDynPlan &) = delete;
  ScopedHIPfftDynPlan &operator=(ScopedHIPfftDynPlan &&)      = delete;
  ScopedHIPfftDynPlan(ScopedHIPfftDynPlan &&)                 = delete;

  hipfftHandle plan() const noexcept { return m_plan; }

  /// \brief Return the workspace size in Byte
  /// \return the workspace size in Byte
  std::size_t workspace_size() const noexcept { return m_workspace_size; }

  template <typename WorkViewType>
  void set_work_area(const WorkViewType &work) {
    using value_type           = typename WorkViewType::non_const_value_type;
    std::size_t workspace_size = work.size() * sizeof(value_type);
    KOKKOSFFT_THROW_IF(
        workspace_size < m_workspace_size,
        "insufficient work buffer size. buffer size: " +
            std::to_string(workspace_size) +
            ", required size: " + std::to_string(m_workspace_size));
    void *work_area = static_cast<void *>(work.data());
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetWorkArea(m_plan, work_area));
  }

  void commit(const Kokkos::HIP &exec_space) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftSetStream(m_plan, exec_space.hip_stream()));
  }
};

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
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

/// \brief The index type used in backend FFT plan
/// Both hipFFT and FFTW use int as index type
/// \tparam ExecutionSpace The type of Kokkos execution space
template <typename ExecutionSpace>
using FFTIndexType = int;

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

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using fftw_plan_type   = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using hipfft_plan_type = ScopedHIPfftDynPlan;
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

template <typename ExecutionSpace>
using FFTIndexType = int;

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

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using type = ScopedHIPfftDynPlan;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
