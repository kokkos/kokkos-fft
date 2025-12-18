// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TYPES_HPP
#define KOKKOSFFT_CUDA_TYPES_HPP

#include <cufft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_asserts.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

// Check the size of complex type
static_assert(sizeof(cufftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(cufftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(cufftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(cufftDoubleComplex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

/// \brief A class that wraps cufft for RAII
struct ScopedCufftPlan {
 private:
  cufftHandle m_plan;

 public:
  ScopedCufftPlan(int nx, cufftType type, int batch) {
    cufftResult cufft_rt = cufftPlan1d(&m_plan, nx, type, batch);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftPlan1d failed");
  }

  ScopedCufftPlan(int nx, int ny, cufftType type) {
    cufftResult cufft_rt = cufftPlan2d(&m_plan, nx, ny, type);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftPlan2d failed");
  }

  ScopedCufftPlan(int nx, int ny, int nz, cufftType type) {
    cufftResult cufft_rt = cufftPlan3d(&m_plan, nx, ny, nz, type);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftPlan3d failed");
  }

  ScopedCufftPlan(int rank, int *n, int *inembed, int istride, int idist,
                  int *onembed, int ostride, int odist, cufftType type,
                  int batch) {
    cufftResult cufft_rt =
        cufftPlanMany(&m_plan, rank, n, inembed, istride, idist, onembed,
                      ostride, odist, type, batch);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftPlanMany failed");
  }

  ~ScopedCufftPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_cufft]");
    cufftResult cufft_rt = cufftDestroy(m_plan);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");
  }

  ScopedCufftPlan()                                   = delete;
  ScopedCufftPlan(const ScopedCufftPlan &)            = delete;
  ScopedCufftPlan &operator=(const ScopedCufftPlan &) = delete;
  ScopedCufftPlan &operator=(ScopedCufftPlan &&)      = delete;
  ScopedCufftPlan(ScopedCufftPlan &&)                 = delete;

  cufftHandle plan() const noexcept { return m_plan; }
  void commit(const Kokkos::Cuda &exec_space) const {
    cufftResult cufft_rt = cufftSetStream(m_plan, exec_space.cuda_stream());
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftSetStream failed");
  }
};

/// \brief A class that wraps cufft for RAII
struct ScopedCufftDynPlan {
 private:
  cufftHandle m_plan;
  std::size_t m_workspace_size;

 public:
  ScopedCufftDynPlan(int nx, cufftType type, int batch) {
    cufftResult cufft_rt = cufftCreate(&m_plan);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");

    cufft_rt = cufftSetAutoAllocation(m_plan, 0);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS,
                       "cufftSetAutoAllocation failed");

    cufft_rt = cufftMakePlan1d(m_plan, nx, type, batch, &m_workspace_size);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan1d failed");
  }

  ScopedCufftDynPlan(const std::vector<int> &fft_extents, cufftType type) {
    cufftResult cufft_rt = cufftCreate(&m_plan);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");

    cufft_rt = cufftSetAutoAllocation(m_plan, 0);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS,
                       "cufftSetAutoAllocation failed");

    if (fft_extents.size() == 2) {
      auto nx = fft_extents.at(0), ny = fft_extents.at(1);
      cufft_rt = cufftMakePlan2d(m_plan, nx, ny, type, &m_workspace_size);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan2d failed");
    } else if (fft_extents.size() == 3) {
      auto nx = fft_extents.at(0), ny = fft_extents.at(1),
           nz  = fft_extents.at(2);
      cufft_rt = cufftMakePlan3d(m_plan, nx, ny, nz, type, &m_workspace_size);
      KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlan3d failed");
    } else {
      KOKKOSFFT_THROW_IF(true, "FFT dimension can be 2D or 3D only");
    }
  }

  ScopedCufftDynPlan(int rank, int *n, int *inembed, int istride, int idist,
                     int *onembed, int ostride, int odist, cufftType type,
                     int batch) {
    cufftResult cufft_rt = cufftCreate(&m_plan);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftCreate failed");

    cufft_rt = cufftSetAutoAllocation(m_plan, 0);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS,
                       "cufftSetAutoAllocation failed");

    cufft_rt =
        cufftMakePlanMany(m_plan, rank, n, inembed, istride, idist, onembed,
                          ostride, odist, type, batch, &m_workspace_size);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftMakePlanMany failed");
  }

  ~ScopedCufftDynPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_cufft]");
    cufftResult cufft_rt = cufftDestroy(m_plan);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");
  }

  ScopedCufftDynPlan()                                      = delete;
  ScopedCufftDynPlan(const ScopedCufftDynPlan &)            = delete;
  ScopedCufftDynPlan &operator=(const ScopedCufftDynPlan &) = delete;
  ScopedCufftDynPlan &operator=(ScopedCufftDynPlan &&)      = delete;
  ScopedCufftDynPlan(ScopedCufftDynPlan &&)                 = delete;

  cufftHandle plan() const noexcept { return m_plan; }

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
    void *work_area      = static_cast<void *>(work.data());
    cufftResult cufft_rt = cufftSetWorkArea(m_plan, work_area);
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftSetWorkArea failed");
  }

  void commit(const Kokkos::Cuda &exec_space) {
    cufftResult cufft_rt = cufftSetStream(m_plan, exec_space.cuda_stream());
    KOKKOSFFT_THROW_IF(cufft_rt != CUFFT_SUCCESS, "cufftSetStream failed");
  }
};

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
template <typename ExecutionSpace>
struct FFTDataType {
  using float32 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftReal, float>;
  using float64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftDoubleReal, double>;
  using complex64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftComplex, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                         cufftDoubleComplex, fftw_complex>;
};

template <typename ExecutionSpace>
using TransformType =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>, cufftType,
                       FFTWTransformType>;

/// \brief The index type used in backend FFT plan
/// Both cuFFT and FFTW use int as index type
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

  static constexpr TransformTypeOnExecSpace m_cuda_type =
      std::is_same_v<T1, float> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr TransformTypeOnExecSpace m_cpu_type =
      std::is_same_v<T1, float> ? FFTWTransformType::R2C
                                : FFTWTransformType::D2Z;

  static constexpr TransformTypeOnExecSpace type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      return m_cuda_type;
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

  static constexpr TransformTypeOnExecSpace m_cuda_type =
      std::is_same_v<T1, float> ? CUFFT_C2R : CUFFT_Z2D;
  static constexpr TransformTypeOnExecSpace m_cpu_type =
      std::is_same_v<T1, float> ? FFTWTransformType::C2R
                                : FFTWTransformType::Z2D;

  static constexpr TransformTypeOnExecSpace type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      return m_cuda_type;
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

  static constexpr TransformTypeOnExecSpace m_cuda_type =
      std::is_same_v<T1, float> ? CUFFT_C2C : CUFFT_Z2Z;
  static constexpr TransformTypeOnExecSpace m_cpu_type =
      std::is_same_v<T1, float> ? FFTWTransformType::C2C
                                : FFTWTransformType::Z2Z;

  static constexpr TransformTypeOnExecSpace type() {
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
      return m_cuda_type;
    } else {
      return m_cpu_type;
    }
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftw_plan_type  = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using cufft_plan_type = ScopedCufftPlan;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                                  cufft_plan_type, fftw_plan_type>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using fftw_plan_type  = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using cufft_plan_type = ScopedCufftDynPlan;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::Cuda>,
                                  cufft_plan_type, fftw_plan_type>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Cuda> ? CUFFT_FORWARD
                                                   : FFTW_FORWARD;
  static constexpr FFTDirectionType BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Cuda> ? CUFFT_INVERSE
                                                   : FFTW_BACKWARD;
  return direction == Direction::forward ? FORWARD : BACKWARD;
}
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = cufftReal;
  using float64    = cufftDoubleReal;
  using complex64  = cufftComplex;
  using complex128 = cufftDoubleComplex;
};

template <typename ExecutionSpace>
using TransformType = cufftType;

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
  static constexpr cufftType m_type =
      std::is_same_v<T1, float> ? CUFFT_R2C : CUFFT_D2Z;
  static constexpr cufftType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr cufftType m_type =
      std::is_same_v<T2, float> ? CUFFT_C2R : CUFFT_Z2D;
  static constexpr cufftType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr cufftType m_type =
      std::is_same_v<T1, float> ? CUFFT_C2C : CUFFT_Z2Z;
  static constexpr cufftType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = ScopedCufftPlan;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using type = ScopedCufftDynPlan;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
