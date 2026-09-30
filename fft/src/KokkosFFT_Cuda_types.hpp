// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TYPES_HPP
#define KOKKOSFFT_CUDA_TYPES_HPP

#include <cufft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Cuda_asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
#include <cuda_runtime.h>
#include <cufftXt.h>

// Backend-agnostic aliases for the vendor callback function pointer types.
// Users writing a callback function or __device__ global should use these
// instead of naming cufftCallbackLoadR/etc. directly, so the same callback
// source stays portable as more backends gain callback support.
using kokkosfftCallbackLoadR = cufftCallbackLoadR;
using kokkosfftCallbackLoadD = cufftCallbackLoadD;
using kokkosfftCallbackLoadC = cufftCallbackLoadC;
using kokkosfftCallbackLoadZ = cufftCallbackLoadZ;

using kokkosfftCallbackStoreR = cufftCallbackStoreR;
using kokkosfftCallbackStoreD = cufftCallbackStoreD;
using kokkosfftCallbackStoreC = cufftCallbackStoreC;
using kokkosfftCallbackStoreZ = cufftCallbackStoreZ;
#endif

// Check the size of complex type
static_assert(sizeof(cufftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(cufftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(cufftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(cufftDoubleComplex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
// cufftXtCallbackType has distinct enum values for single vs double
// precision (CUFFT_CB_LD_REAL vs CUFFT_CB_LD_REAL_DOUBLE, etc.); collapsing
// R/D or C/Z onto the same value here would tell cufftXtSetCallback the
// wrong callback type for any double-precision callback.
//
// Primary template intentionally left undefined: instantiating it with an
// unsupported CallbackSymbol fails to compile with that type named in the
// error, instead of silently falling through.
template <typename CallbackSymbol>
struct deduce_callback_type;

template <>
struct deduce_callback_type<cufftCallbackLoadR> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_LD_REAL;
};
template <>
struct deduce_callback_type<cufftCallbackLoadD> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_LD_REAL_DOUBLE;
};
template <>
struct deduce_callback_type<cufftCallbackLoadC> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_LD_COMPLEX;
};
template <>
struct deduce_callback_type<cufftCallbackLoadZ> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_LD_COMPLEX_DOUBLE;
};
template <>
struct deduce_callback_type<cufftCallbackStoreR> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_ST_REAL;
};
template <>
struct deduce_callback_type<cufftCallbackStoreD> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_ST_REAL_DOUBLE;
};
template <>
struct deduce_callback_type<cufftCallbackStoreC> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_ST_COMPLEX;
};
template <>
struct deduce_callback_type<cufftCallbackStoreZ> {
  static constexpr cufftXtCallbackType value = CUFFT_CB_ST_COMPLEX_DOUBLE;
};

/// \brief Helper to deduce the cufftXtCallbackType enum value for a vendor
/// callback symbol typedef (e.g. cufftCallbackLoadR)
template <typename CallbackSymbol>
inline constexpr cufftXtCallbackType deduce_callback_type_v =
    deduce_callback_type<CallbackSymbol>::value;
#endif

/// \brief A class that wraps cufft for RAII
struct ScopedCufftPlan {
 private:
  cufftHandle m_plan;
  void *m_callback_params = nullptr;

 public:
  ScopedCufftPlan(int nx, cufftType type, int batch) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftPlan1d(&m_plan, nx, type, batch));
  }

  ScopedCufftPlan(int nx, int ny, cufftType type) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftPlan2d(&m_plan, nx, ny, type));
  }

  ScopedCufftPlan(int nx, int ny, int nz, cufftType type) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftPlan3d(&m_plan, nx, ny, nz, type));
  }

  ScopedCufftPlan(int rank, int *n, int *inembed, int istride, int idist,
                  int *onembed, int ostride, int odist, cufftType type,
                  int batch) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftPlanMany(&m_plan, rank, n, inembed, istride,
                                             idist, onembed, ostride, odist,
                                             type, batch));
  }

  ~ScopedCufftPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_cufft]");
    cufftResult cufft_rt = cufftDestroy(m_plan);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");

    // cuFFT only borrows the callerInfo pointer set in set_callback() for the
    // lifetime of the plan; it never frees it itself. This class owns that
    // allocation, so free it here now that the plan (and anything that might
    // still be reading it) is gone.
    if (m_callback_params != nullptr) {
      cudaError_t cuda_rt = cudaFree(m_callback_params);
      if (cuda_rt != cudaSuccess) Kokkos::abort("cudaFree failed");
    }
  }

  ScopedCufftPlan()                                   = delete;
  ScopedCufftPlan(const ScopedCufftPlan &)            = delete;
  ScopedCufftPlan &operator=(const ScopedCufftPlan &) = delete;
  ScopedCufftPlan &operator=(ScopedCufftPlan &&)      = delete;
  ScopedCufftPlan(ScopedCufftPlan &&)                 = delete;

  cufftHandle plan() const noexcept { return m_plan; }
  void commit(const Kokkos::Cuda &exec_space) const {
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftSetStream(m_plan, exec_space.cuda_stream()));
  }

  /// \brief Attach a load or store callback to this plan, deduced from
  /// CallbackSymbolType (the vendor typedef, e.g. cufftCallbackLoadR vs
  /// cufftCallbackStoreR, already encodes which one it is).
  ///
  /// \tparam CallbackSymbolType The type of the callback symbol
  /// \tparam CallbackParamsType The type of the caller-provided params
  /// \param d_callback_symbol The __device__ global holding the callback
  /// function pointer
  /// \param params The callback parameters. Copied into a device allocation
  /// owned by this ScopedCufftPlan, freed in its destructor -- the caller
  /// never has to manage that memory themselves.
  ///
  /// \todo When KOKKOSFFT_ENABLE_CALLBACK is off, this method currently
  /// compiles to a silent no-op instead of failing to compile or throwing --
  /// a caller who forgets -DKokkosFFT_ENABLE_CALLBACK=ON gets no error, just
  /// an FFT that silently runs without their callback attached. Needs a
  /// compile-time failure instead (discussed in review, not yet decided).
  template <typename CallbackSymbolType, typename CallbackParamsType>
  void set_callback(const CallbackSymbolType &d_callback_symbol,
                    const CallbackParamsType &params) {
#if defined(KOKKOSFFT_ENABLE_CALLBACK)
    CallbackSymbolType callback{};
    KOKKOSFFT_CHECK_CUDA_CALL(
        cudaMemcpyFromSymbol(&callback, d_callback_symbol, sizeof(callback)));

    constexpr cufftXtCallbackType cb_type =
        deduce_callback_type_v<CallbackSymbolType>;
    void *callback_ptr          = reinterpret_cast<void *>(callback);

    if (m_callback_params != nullptr) {
      KOKKOSFFT_CHECK_CUDA_CALL(cudaFree(m_callback_params));
      m_callback_params = nullptr;
    }
    KOKKOSFFT_CHECK_CUDA_CALL(
        cudaMalloc(&m_callback_params, sizeof(CallbackParamsType)));
    KOKKOSFFT_CHECK_CUDA_CALL(cudaMemcpy(m_callback_params, &params,
                                         sizeof(CallbackParamsType),
                                         cudaMemcpyHostToDevice));

    void *callback_params_ptr = m_callback_params;
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetCallback(
        m_plan, &callback_ptr, cb_type, &callback_params_ptr));
#endif
  }
};

/// \brief A class that wraps cufft for RAII
struct ScopedCufftDynPlan {
 private:
  cufftHandle m_plan;
  std::size_t m_workspace_size;
  void *m_callback_params = nullptr;

 public:
  ScopedCufftDynPlan(int nx, cufftType type, int batch) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan));
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftSetAutoAllocation(m_plan, 0));
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftMakePlan1d(m_plan, nx, type, batch, &m_workspace_size));
  }

  ScopedCufftDynPlan(const std::vector<int> &fft_extents, cufftType type) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan));
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftSetAutoAllocation(m_plan, 0));

    if (fft_extents.size() == 2) {
      auto nx = fft_extents.at(0), ny = fft_extents.at(1);
      KOKKOSFFT_CHECK_CUFFT_CALL(
          cufftMakePlan2d(m_plan, nx, ny, type, &m_workspace_size));
    } else if (fft_extents.size() == 3) {
      auto nx = fft_extents.at(0), ny = fft_extents.at(1),
           nz = fft_extents.at(2);
      KOKKOSFFT_CHECK_CUFFT_CALL(
          cufftMakePlan3d(m_plan, nx, ny, nz, type, &m_workspace_size));
    } else {
      KOKKOSFFT_THROW_IF(true, "FFT dimension can be 2D or 3D only");
    }
  }

  ScopedCufftDynPlan(int rank, int *n, int *inembed, int istride, int idist,
                     int *onembed, int ostride, int odist, cufftType type,
                     int batch) {
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftCreate(&m_plan));
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftSetAutoAllocation(m_plan, 0));
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftMakePlanMany(m_plan, rank, n, inembed, istride, idist, onembed,
                          ostride, odist, type, batch, &m_workspace_size));
  }

  ~ScopedCufftDynPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_cufft]");
    cufftResult cufft_rt = cufftDestroy(m_plan);
    if (cufft_rt != CUFFT_SUCCESS) Kokkos::abort("cufftDestroy failed");

    // cuFFT only borrows the callerInfo pointer set in set_callback() for the
    // lifetime of the plan; it never frees it itself. This class owns that
    // allocation, so free it here now that the plan (and anything that might
    // still be reading it) is gone.
    if (m_callback_params != nullptr) {
      cudaError_t cuda_rt = cudaFree(m_callback_params);
      if (cuda_rt != cudaSuccess) Kokkos::abort("cudaFree failed");
    }
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
    void *work_area = static_cast<void *>(work.data());

    KOKKOSFFT_CHECK_CUFFT_CALL(cufftSetWorkArea(m_plan, work_area));
  }

  void commit(const Kokkos::Cuda &exec_space) {
    KOKKOSFFT_CHECK_CUFFT_CALL(
        cufftSetStream(m_plan, exec_space.cuda_stream()));
  }

  /// \brief Attach a load or store callback to this plan, deduced from
  /// CallbackSymbolType (the vendor typedef, e.g. cufftCallbackLoadR vs
  /// cufftCallbackStoreR, already encodes which one it is).
  ///
  /// \tparam CallbackSymbolType The type of the callback symbol
  /// \tparam CallbackParamsType The type of the caller-provided params
  /// \param d_callback_symbol The __device__ global holding the callback
  /// function pointer
  /// \param params The callback parameters. Copied into a device allocation
  /// owned by this ScopedCufftDynPlan, freed in its destructor -- the caller
  /// never has to manage that memory themselves.
  ///
  /// \todo When KOKKOSFFT_ENABLE_CALLBACK is off, this method currently
  /// compiles to a silent no-op instead of failing to compile or throwing --
  /// a caller who forgets -DKokkosFFT_ENABLE_CALLBACK=ON gets no error, just
  /// an FFT that silently runs without their callback attached. Needs a
  /// compile-time failure instead (discussed in review, not yet decided).
  template <typename CallbackSymbolType, typename CallbackParamsType>
  void set_callback(const CallbackSymbolType &d_callback_symbol,
                    const CallbackParamsType &params) {
#if defined(KOKKOSFFT_ENABLE_CALLBACK)
    CallbackSymbolType callback{};
    KOKKOSFFT_CHECK_CUDA_CALL(
        cudaMemcpyFromSymbol(&callback, d_callback_symbol, sizeof(callback)));

    constexpr cufftXtCallbackType cb_type =
        deduce_callback_type_v<CallbackSymbolType>;
    void *callback_ptr          = reinterpret_cast<void *>(callback);

    if (m_callback_params != nullptr) {
      KOKKOSFFT_CHECK_CUDA_CALL(cudaFree(m_callback_params));
      m_callback_params = nullptr;
    }
    KOKKOSFFT_CHECK_CUDA_CALL(
        cudaMalloc(&m_callback_params, sizeof(CallbackParamsType)));
    KOKKOSFFT_CHECK_CUDA_CALL(cudaMemcpy(m_callback_params, &params,
                                         sizeof(CallbackParamsType),
                                         cudaMemcpyHostToDevice));

    void *callback_params_ptr = m_callback_params;
    KOKKOSFFT_CHECK_CUFFT_CALL(cufftXtSetCallback(
        m_plan, &callback_ptr, cb_type, &callback_params_ptr));
#endif
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
