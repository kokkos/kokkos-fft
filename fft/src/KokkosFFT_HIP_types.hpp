// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_HIP_TYPES_HPP
#define KOKKOSFFT_HIP_TYPES_HPP

#include <concepts>
#include <vector>
#include <hipfft/hipfft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_HIP_asserts.hpp"
#include "KokkosFFT_Traits.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
#include <hip/hip_runtime_api.h>
#include <hipfft/hipfftXt.h>

// Backend-agnostic aliases for the callback function pointer types.
// Users writing a callback function or __device__ global should use these
// instead of naming hipfftCallbackLoadR/etc. directly, so the same callback
// source stays portable across backends.
//
// Real data are plain float/double and complex data are Kokkos::complex<T>,
// not the vendor types (hipfftReal, hipfftComplex, ...): they have the same
// layout, and callbacks are written the same way on every backend.
using kokkosfftCallbackLoadR = float (*)(void *dataIn, std::size_t offset,
                                         void *callerInfo, void *sharedPointer);
using kokkosfftCallbackLoadD = double (*)(void *dataIn, std::size_t offset,
                                          void *callerInfo,
                                          void *sharedPointer);
using kokkosfftCallbackLoadC = Kokkos::complex<float> (*)(void *dataIn,
                                                          std::size_t offset,
                                                          void *callerInfo,
                                                          void *sharedPointer);
using kokkosfftCallbackLoadZ = Kokkos::complex<double> (*)(void *dataIn,
                                                           std::size_t offset,
                                                           void *callerInfo,
                                                           void *sharedPointer);

using kokkosfftCallbackStoreR = void (*)(void *dataOut, std::size_t offset,
                                         float element, void *callerInfo,
                                         void *sharedPointer);
using kokkosfftCallbackStoreD = void (*)(void *dataOut, std::size_t offset,
                                         double element, void *callerInfo,
                                         void *sharedPointer);
using kokkosfftCallbackStoreC = void (*)(void *dataOut, std::size_t offset,
                                         Kokkos::complex<float> element,
                                         void *callerInfo, void *sharedPointer);
using kokkosfftCallbackStoreZ = void (*)(void *dataOut, std::size_t offset,
                                         Kokkos::complex<double> element,
                                         void *callerInfo, void *sharedPointer);
#endif

// Check the size of complex type
static_assert(sizeof(hipfftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(hipfftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(hipfftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(hipfftDoubleComplex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
#if defined(KOKKOSFFT_ENABLE_CALLBACK)
// hipfftXtCallbackType has distinct enum values for single vs double
// precision (HIPFFT_CB_LD_REAL vs HIPFFT_CB_LD_REAL_DOUBLE, etc.), so each
// callback alias maps to its own value.
//
// Primary template: any other type, in particular a callback written with the
// vendor complex type (hipfftComplex, float2, ...), is rejected with an
// explanatory message.
template <typename CallbackSymbol>
struct deduce_callback_type {
  static_assert(sizeof(CallbackSymbol) == 0,
                "set_callback(): unsupported callback type. Declare the "
                "callback with the kokkosfftCallback{Load,Store}{R,D,C,Z} "
                "aliases; complex data are Kokkos::complex<T>, not the vendor "
                "complex type (hipfftComplex, float2, ...)");
};

template <>
struct deduce_callback_type<kokkosfftCallbackLoadR> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_REAL;
  static constexpr bool is_load               = true;
  static constexpr bool is_real               = true;
  using value_type                            = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackLoadD> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_REAL_DOUBLE;
  static constexpr bool is_load               = true;
  static constexpr bool is_real               = true;
  using value_type                            = double;
};
template <>
struct deduce_callback_type<kokkosfftCallbackLoadC> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_COMPLEX;
  static constexpr bool is_load               = true;
  static constexpr bool is_real               = false;
  using value_type                            = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackLoadZ> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_LD_COMPLEX_DOUBLE;
  static constexpr bool is_load               = true;
  static constexpr bool is_real               = false;
  using value_type                            = double;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreR> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_REAL;
  static constexpr bool is_load               = false;
  static constexpr bool is_real               = true;
  using value_type                            = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreD> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_REAL_DOUBLE;
  static constexpr bool is_load               = false;
  static constexpr bool is_real               = true;
  using value_type                            = double;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreC> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_COMPLEX;
  static constexpr bool is_load               = false;
  static constexpr bool is_real               = false;
  using value_type                            = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreZ> {
  static constexpr hipfftXtCallbackType value = HIPFFT_CB_ST_COMPLEX_DOUBLE;
  static constexpr bool is_load               = false;
  static constexpr bool is_real               = false;
  using value_type                            = double;
};

/// \brief Helper to deduce the hipfftXtCallbackType enum value for a
/// callback symbol alias (e.g. kokkosfftCallbackLoadR)
template <typename CallbackSymbol>
inline constexpr hipfftXtCallbackType deduce_callback_type_v =
    deduce_callback_type<CallbackSymbol>::value;

/// \brief Helper to check whether a callback symbol alias is a load
/// callback (true) or a store callback (false)
template <typename CallbackSymbol>
inline constexpr bool is_load_callback_v =
    deduce_callback_type<CallbackSymbol>::is_load;

/// \brief Compile-time check of a callback against a plan.
///
/// hipFFT does not check that a callback suits the plan before running it, so
/// check here that the precision of the callback matches the plan and that it
/// takes the same kind of data (real or complex) as the input (load
/// callback) or the output (store callback) of the plan.
///
/// \tparam CallbackSymbolType The type of the callback symbol
/// \tparam InValueType The value type of the input of the plan
/// \tparam OutValueType The value type of the output of the plan
template <typename CallbackSymbolType, typename InValueType,
          typename OutValueType>
constexpr void static_check_callback() {
  using callback_traits        = deduce_callback_type<CallbackSymbolType>;
  constexpr TransformKind kind = transform_kind_v<InValueType, OutValueType>;
  static_assert(std::same_as<typename callback_traits::value_type,
                             base_floating_point_type<InValueType>>,
                "set_callback(): the precision of the callback does not match "
                "the precision of the plan");
  if constexpr (callback_traits::is_load) {
    // The input of the plan is real only for R2C/D2Z transforms
    static_assert(callback_traits::is_real == (kind == TransformKind::R2C),
                  "set_callback(): a load callback must take the same kind "
                  "of data (real or complex) as the input of the plan");
  } else {
    // The output of the plan is real only for C2R/Z2D transforms
    static_assert(callback_traits::is_real == (kind == TransformKind::C2R),
                  "set_callback(): a store callback must take the same kind "
                  "of data (real or complex) as the output of the plan");
  }
}
#endif

/// \brief Deduce the hipfftType from the base floating point type and the
/// transform kind
/// \tparam FloatType The base floating point type (float or double)
/// \tparam Kind The kind of transform (C2C, R2C or C2R)
template <typename FloatType, TransformKind Kind>
consteval hipfftType deduce_hipfft_type() {
  static_assert(is_real_v<FloatType>,
                "deduce_hipfft_type: FloatType must be float or double");
  constexpr bool is_single = std::same_as<FloatType, float>;
  if constexpr (Kind == TransformKind::C2C) {
    return is_single ? HIPFFT_C2C : HIPFFT_Z2Z;
  } else if constexpr (Kind == TransformKind::R2C) {
    return is_single ? HIPFFT_R2C : HIPFFT_D2Z;
  } else {
    static_assert(Kind == TransformKind::C2R);
    return is_single ? HIPFFT_C2R : HIPFFT_Z2D;
  }
}

/// \brief A class that wraps hipfft for RAII
/// \tparam InValueType The value type of the input view
/// \tparam OutValueType The value type of the output view
template <typename InValueType, typename OutValueType>
struct ScopedHIPfftPlan {
  static_assert(
      have_same_base_floating_point_type_v<InValueType, OutValueType>,
      "ScopedHIPfftPlan: input and output must have the same precision");

  //! Transform kind deduced from the input/output value types
  static constexpr TransformKind transform_kind =
      transform_kind_v<InValueType, OutValueType>;

  //! hipfftType deduced from the precision and the transform kind
  static constexpr hipfftType type =
      deduce_hipfft_type<base_floating_point_type<InValueType>,
                         transform_kind>();

 private:
  hipfftHandle m_plan;
  std::size_t m_workspace_size = 0;

  //! Internal work buffer for the plan
  void *m_workbuffer = nullptr;

  // hipfftXtSetCallback takes void** callbacks/callbackData: arrays with one
  // entry per device. We only support a single device, so each void* member
  // below is passed (by address) as an array of length 1. Unlike cuFFT,
  // hipFFT keeps those void** pointers instead of copying the values, and
  // re-reads them on every later hipfftXtSetCallback call, so what they
  // point to must outlive set_callback(): hence members, not locals.
  [[maybe_unused]] void *m_load_callback  = nullptr;
  [[maybe_unused]] void *m_store_callback = nullptr;
  void *m_load_callback_params            = nullptr;
  void *m_store_callback_params           = nullptr;

 public:
  ScopedHIPfftPlan(int nx, int batch) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftCreate(&m_plan));
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetAutoAllocation(m_plan, 0));
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftMakePlan1d(m_plan, nx, type, batch, &m_workspace_size));
  }

  ScopedHIPfftPlan(const std::vector<int> &fft_extents) {
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

  ScopedHIPfftPlan(int rank, int *n, int *inembed, int istride, int idist,
                   int *onembed, int ostride, int odist, int batch) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftCreate(&m_plan));
    KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetAutoAllocation(m_plan, 0));
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftMakePlanMany(m_plan, rank, n, inembed, istride, idist, onembed,
                           ostride, odist, type, batch, &m_workspace_size));
  }

  ~ScopedHIPfftPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_hipfft]");
    if (m_workbuffer != nullptr) {
      Kokkos::kokkos_free<Kokkos::HIP>(m_workbuffer);
      m_workbuffer = nullptr;
    }

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

  void set_work_area() {
    if (m_workspace_size > 0) {
      m_workbuffer =
          Kokkos::kokkos_malloc<Kokkos::HIP>("workbuffer", m_workspace_size);
      KOKKOSFFT_CHECK_HIPFFT_CALL(hipfftSetWorkArea(m_plan, m_workbuffer));
    }
  }

  void commit(const Kokkos::HIP &exec_space) {
    KOKKOSFFT_CHECK_HIPFFT_CALL(
        hipfftSetStream(m_plan, exec_space.hip_stream()));
  }

  /// \brief Attach a load or store callback to this plan, deduced from
  /// CallbackSymbolType (the kokkosfft alias, e.g. kokkosfftCallbackLoadR vs
  /// kokkosfftCallbackStoreR, already encodes which one it is).
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
  void set_callback(
      [[maybe_unused]] const CallbackSymbolType &d_callback_symbol,
      [[maybe_unused]] const CallbackParamsType &params) {
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

inline auto hipfft_direction(Direction direction) {
  return direction == Direction::forward ? HIPFFT_FORWARD : HIPFFT_BACKWARD;
}

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
template <typename ExecutionSpace>
struct FFTDataType {
  using float32 = std::conditional_t<std::same_as<ExecutionSpace, Kokkos::HIP>,
                                     hipfftReal, float>;
  using float64 = std::conditional_t<std::same_as<ExecutionSpace, Kokkos::HIP>,
                                     hipfftDoubleReal, double>;
  using complex64 =
      std::conditional_t<std::same_as<ExecutionSpace, Kokkos::HIP>,
                         hipfftComplex, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::same_as<ExecutionSpace, Kokkos::HIP>,
                         hipfftDoubleComplex, fftw_complex>;
};

template <typename ExecutionSpace, typename T1, typename T2>
using FFTPlanType =
    std::conditional_t<std::same_as<ExecutionSpace, Kokkos::HIP>,
                       ScopedHIPfftPlan<T1, T2>,
                       ScopedFFTWPlan<ExecutionSpace, T1, T2>>;
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = hipfftReal;
  using float64    = hipfftDoubleReal;
  using complex64  = hipfftComplex;
  using complex128 = hipfftDoubleComplex;
};

template <typename ExecutionSpace, typename T1, typename T2>
using FFTPlanType = ScopedHIPfftPlan<T1, T2>;
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
