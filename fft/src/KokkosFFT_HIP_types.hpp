// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TYPES_HPP
#define KOKKOSFFT_HIP_TYPES_HPP

#include <hipfft/hipfft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_HIP_asserts.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
#include <hip/hip_runtime_api.h>
#include <hipfft/hipfftXt.h>

// Backend-agnostic aliases for the vendor callback function pointer types.
// Users writing a callback function or __device__ global should use these
// instead of naming hipfftCallbackLoadR/etc. directly, so the same callback
// source stays portable across backends.
using kokkosfftCallbackLoadR = hipfftCallbackLoadR;
using kokkosfftCallbackLoadD = hipfftCallbackLoadD;
using kokkosfftCallbackLoadC = hipfftCallbackLoadC;
using kokkosfftCallbackLoadZ = hipfftCallbackLoadZ;

using kokkosfftCallbackStoreR = hipfftCallbackStoreR;
using kokkosfftCallbackStoreD = hipfftCallbackStoreD;
using kokkosfftCallbackStoreC = hipfftCallbackStoreC;
using kokkosfftCallbackStoreZ = hipfftCallbackStoreZ;
#endif

// Check the size of complex type
static_assert(sizeof(hipfftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(hipfftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(hipfftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(hipfftDoubleComplex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
// hipfftXtCallbackType has distinct enum values for single vs double
// precision (HIPFFT_CB_LD_REAL vs HIPFFT_CB_LD_REAL_DOUBLE, etc.), so each
// vendor callback typedef maps to its own value.
//
// Primary template intentionally left undefined: instantiating it with an
// unsupported CallbackSymbol fails to compile with that type named in the
// error, instead of silently falling through.
template <typename CallbackSymbol>
struct deduce_callback_type;

template <>
struct deduce_callback_type<hipfftCallbackLoadR> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_REAL;
  static constexpr bool is_load               = true;
};
template <>
struct deduce_callback_type<hipfftCallbackLoadD> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_REAL_DOUBLE;
  static constexpr bool is_load               = true;
};
template <>
struct deduce_callback_type<hipfftCallbackLoadC> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_COMPLEX;
  static constexpr bool is_load               = true;
};
template <>
struct deduce_callback_type<hipfftCallbackLoadZ> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_COMPLEX_DOUBLE;
  static constexpr bool is_load               = true;
};
template <>
struct deduce_callback_type<hipfftCallbackStoreR> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_REAL;
  static constexpr bool is_load               = false;
};
template <>
struct deduce_callback_type<hipfftCallbackStoreD> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_REAL_DOUBLE;
  static constexpr bool is_load               = false;
};
template <>
struct deduce_callback_type<hipfftCallbackStoreC> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_COMPLEX;
  static constexpr bool is_load               = false;
};
template <>
struct deduce_callback_type<hipfftCallbackStoreZ> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_COMPLEX_DOUBLE;
  static constexpr bool is_load               = false;
};

/// \brief Helper to deduce the hipfftXtCallbackType enum value for a vendor
/// callback symbol typedef (e.g. hipfftCallbackLoadR)
template <typename CallbackSymbol>
inline constexpr hipfftXtCallbackType deduce_callback_type_v =
    deduce_callback_type<CallbackSymbol>::value;

/// \brief Helper to check whether a vendor callback symbol typedef is a load
/// callback (true) or a store callback (false)
template <typename CallbackSymbol>
inline constexpr bool is_load_callback_v =
    deduce_callback_type<CallbackSymbol>::is_load;
#endif

/// \brief A class that wraps hipfft for RAII
struct ScopedHIPfftPlan {
 private:
  hipfftHandle m_plan;
  // hipFFT keeps the callbacks/callbackData arrays passed to
  // hipfftXtSetCallback by pointer instead of copying them (unlike cuFFT),
  // and re-reads them on every later set_callback() call, so they must
  // outlive set_callback(). These members are those one-element arrays.
  void *m_load_callback         = nullptr;
  void *m_store_callback        = nullptr;
  void *m_load_callback_params  = nullptr;
  void *m_store_callback_params = nullptr;

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

    // hipFFT only borrows the callerInfo pointers set in set_callback() for
    // the lifetime of the plan; it never frees them itself. This class owns
    // those allocations, so free them here now that the plan (and anything
    // that might still be reading them) is gone.
    if (m_load_callback_params != nullptr) {
      hipError_t hip_rt = hipFree(m_load_callback_params);
      if (hip_rt != hipSuccess) Kokkos::abort("hipFree failed");
    }
    if (m_store_callback_params != nullptr) {
      hipError_t hip_rt = hipFree(m_store_callback_params);
      if (hip_rt != hipSuccess) Kokkos::abort("hipFree failed");
    }
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

  /// \brief Attach a load or store callback to this plan, deduced from
  /// CallbackSymbolType (the vendor typedef, e.g. hipfftCallbackLoadR vs
  /// hipfftCallbackStoreR, already encodes which one it is).
  ///
  /// \tparam CallbackSymbolType The type of the callback symbol
  /// \tparam CallbackParamsType The type of the caller-provided params
  /// \param[in] d_callback_symbol The __device__ global holding the callback
  /// function pointer
  /// \param[in] params The callback parameters. Copied into a device allocation
  /// owned by this ScopedHIPfftPlan, freed in its destructor -- the caller
  /// never has to manage that memory themselves.
  ///
  /// \note When KOKKOSFFT_ENABLE_CALLBACK is off, calling this method fails
  /// to compile (static_assert) rather than silently doing nothing.
  ///
  /// \note \p params must always be a real struct, even if the callback
  /// itself never reads it. Passing nullptr fails to compile (static_assert)
  /// rather than silently producing a non-null callerInfo on the device
  /// that points to zeroed memory instead of a real struct.
  template <typename CallbackSymbolType, typename CallbackParamsType>
  void set_callback(const CallbackSymbolType &d_callback_symbol,
                    const CallbackParamsType &params) {
#if defined(KOKKOSFFT_ENABLE_CALLBACK)
    static_assert(!std::is_null_pointer_v<CallbackParamsType>,
                  "set_callback() requires a real params struct -- passing "
                  "nullptr does not produce a null callerInfo on the "
                  "device, it points to zeroed memory instead, which is "
                  "almost certainly not what you want");
    CallbackSymbolType callback{};
    // HIP_SYMBOL is a no-op on the AMD backend, but keeps this portable if
    // HIP is ever compiled through hip-nvcc.
    KOKKOSFFT_CHECK_HIP_CALL(hipMemcpyFromSymbol(
        &callback, HIP_SYMBOL(d_callback_symbol), sizeof(callback)));

    constexpr hipfftXtCallbackType cb_type =
        deduce_callback_type_v<CallbackSymbolType>;

    // Load and store callbacks each get their own device buffer -- the
    // caller-info is independent per callback type, and each may only be
    // set once per plan.
    void **callback_slot;
    void **callback_params_slot;
    if constexpr (is_load_callback_v<CallbackSymbolType>) {
      KOKKOSFFT_THROW_IF(
          m_load_callback_params != nullptr,
          "set_callback() may only set a load callback once per plan");
      callback_slot        = &m_load_callback;
      callback_params_slot = &m_load_callback_params;
    } else {
      KOKKOSFFT_THROW_IF(
          m_store_callback_params != nullptr,
          "set_callback() may only set a store callback once per plan");
      callback_slot        = &m_store_callback;
      callback_params_slot = &m_store_callback_params;
    }

    KOKKOSFFT_CHECK_HIP_CALL(
        hipMalloc(callback_params_slot, sizeof(CallbackParamsType)));
    KOKKOSFFT_CHECK_HIP_CALL(hipMemcpy(*callback_params_slot, &params,
                                       sizeof(CallbackParamsType),
                                       hipMemcpyHostToDevice));

    // hipFFT stores these two void** arrays as-is rather than copying their
    // contents, so pass the plan's own members (stable for the plan's
    // lifetime), never addresses of locals.
    *callback_slot = reinterpret_cast<void *>(callback);
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftXtSetCallback(
        m_plan, callback_slot, cb_type, callback_params_slot));
#else
    static_assert(std::is_void_v<CallbackSymbolType>,
                  "set_callback() requires KokkosFFT to be built with "
                  "-DKokkosFFT_ENABLE_CALLBACK=ON");
#endif
  }
};

/// \brief A class that wraps hipfft for RAII
struct ScopedHIPfftDynPlan {
 private:
  hipfftHandle m_plan;
  std::size_t m_workspace_size;
  // hipFFT keeps the callbacks/callbackData arrays passed to
  // hipfftXtSetCallback by pointer instead of copying them (unlike cuFFT),
  // and re-reads them on every later set_callback() call, so they must
  // outlive set_callback(). These members are those one-element arrays.
  void *m_load_callback         = nullptr;
  void *m_store_callback        = nullptr;
  void *m_load_callback_params  = nullptr;
  void *m_store_callback_params = nullptr;

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

    // hipFFT only borrows the callerInfo pointers set in set_callback() for
    // the lifetime of the plan; it never frees them itself. This class owns
    // those allocations, so free them here now that the plan (and anything
    // that might still be reading them) is gone.
    if (m_load_callback_params != nullptr) {
      hipError_t hip_rt = hipFree(m_load_callback_params);
      if (hip_rt != hipSuccess) Kokkos::abort("hipFree failed");
    }
    if (m_store_callback_params != nullptr) {
      hipError_t hip_rt = hipFree(m_store_callback_params);
      if (hip_rt != hipSuccess) Kokkos::abort("hipFree failed");
    }
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

  /// \brief Attach a load or store callback to this plan, deduced from
  /// CallbackSymbolType (the vendor typedef, e.g. hipfftCallbackLoadR vs
  /// hipfftCallbackStoreR, already encodes which one it is).
  ///
  /// \tparam CallbackSymbolType The type of the callback symbol
  /// \tparam CallbackParamsType The type of the caller-provided params
  /// \param[in] d_callback_symbol The __device__ global holding the callback
  /// function pointer
  /// \param[in] params The callback parameters. Copied into a device allocation
  /// owned by this ScopedHIPfftDynPlan, freed in its destructor -- the caller
  /// never has to manage that memory themselves.
  ///
  /// \note When KOKKOSFFT_ENABLE_CALLBACK is off, calling this method fails
  /// to compile (static_assert) rather than silently doing nothing.
  ///
  /// \note \p params must always be a real struct, even if the callback
  /// itself never reads it. Passing nullptr fails to compile (static_assert)
  /// rather than silently producing a non-null callerInfo on the device
  /// that points to zeroed memory instead of a real struct.
  template <typename CallbackSymbolType, typename CallbackParamsType>
  void set_callback(const CallbackSymbolType &d_callback_symbol,
                    const CallbackParamsType &params) {
#if defined(KOKKOSFFT_ENABLE_CALLBACK)
    static_assert(!std::is_null_pointer_v<CallbackParamsType>,
                  "set_callback() requires a real params struct -- passing "
                  "nullptr does not produce a null callerInfo on the "
                  "device, it points to zeroed memory instead, which is "
                  "almost certainly not what you want");
    CallbackSymbolType callback{};
    // HIP_SYMBOL is a no-op on the AMD backend, but keeps this portable if
    // HIP is ever compiled through hip-nvcc.
    KOKKOSFFT_CHECK_HIP_CALL(hipMemcpyFromSymbol(
        &callback, HIP_SYMBOL(d_callback_symbol), sizeof(callback)));

    constexpr hipfftXtCallbackType cb_type =
        deduce_callback_type_v<CallbackSymbolType>;

    // Load and store callbacks each get their own device buffer -- the
    // caller-info is independent per callback type, and each may only be
    // set once per plan.
    void **callback_slot;
    void **callback_params_slot;
    if constexpr (is_load_callback_v<CallbackSymbolType>) {
      KOKKOSFFT_THROW_IF(
          m_load_callback_params != nullptr,
          "set_callback() may only set a load callback once per plan");
      callback_slot        = &m_load_callback;
      callback_params_slot = &m_load_callback_params;
    } else {
      KOKKOSFFT_THROW_IF(
          m_store_callback_params != nullptr,
          "set_callback() may only set a store callback once per plan");
      callback_slot        = &m_store_callback;
      callback_params_slot = &m_store_callback_params;
    }

    KOKKOSFFT_CHECK_HIP_CALL(
        hipMalloc(callback_params_slot, sizeof(CallbackParamsType)));
    KOKKOSFFT_CHECK_HIP_CALL(hipMemcpy(*callback_params_slot, &params,
                                       sizeof(CallbackParamsType),
                                       hipMemcpyHostToDevice));

    // hipFFT stores these two void** arrays as-is rather than copying their
    // contents, so pass the plan's own members (stable for the plan's
    // lifetime), never addresses of locals.
    *callback_slot = reinterpret_cast<void *>(callback);
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftXtSetCallback(
        m_plan, callback_slot, cb_type, callback_params_slot));
#else
    static_assert(std::is_void_v<CallbackSymbolType>,
                  "set_callback() requires KokkosFFT to be built with "
                  "-DKokkosFFT_ENABLE_CALLBACK=ON");
#endif
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
