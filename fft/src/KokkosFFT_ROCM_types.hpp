// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TYPES_HPP
#define KOKKOSFFT_ROCM_TYPES_HPP

#include <algorithm>
#include <complex>
#include <concepts>
#include <memory>
#include <numeric>
#include <type_traits>

#include <rocfft/rocfft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Convert_Types.hpp"
#include "KokkosFFT_Extents.hpp"
#include "KokkosFFT_ROCM_asserts.hpp"
#include "KokkosFFT_Traits.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

#include <hip/hip_vector_types.h>

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
#include <hip/hip_runtime_api.h>

// Backend-agnostic aliases for the callback function pointer types. rocFFT
// does not provide typedefs, so these mirror the ones of hipFFT/cuFFT
// (opaque `void*` data pointer) and keep a callback function or __device__
// global portable across backends. rocFFT documents a typed data pointer
// (`Tdata*`), but a function pointer taking `void*` is passed through as is.
using kokkosfftCallbackLoadR = float (*)(void *dataIn, std::size_t offset,
                                         void *callerInfo, void *sharedPointer);
using kokkosfftCallbackLoadD = double (*)(void *dataIn, std::size_t offset,
                                          void *callerInfo,
                                          void *sharedPointer);
using kokkosfftCallbackLoadC = float2 (*)(void *dataIn, std::size_t offset,
                                          void *callerInfo,
                                          void *sharedPointer);
using kokkosfftCallbackLoadZ = double2 (*)(void *dataIn, std::size_t offset,
                                           void *callerInfo,
                                           void *sharedPointer);

using kokkosfftCallbackStoreR = void (*)(void *dataOut, std::size_t offset,
                                         float element, void *callerInfo,
                                         void *sharedPointer);
using kokkosfftCallbackStoreD = void (*)(void *dataOut, std::size_t offset,
                                         double element, void *callerInfo,
                                         void *sharedPointer);
using kokkosfftCallbackStoreC = void (*)(void *dataOut, std::size_t offset,
                                         float2 element, void *callerInfo,
                                         void *sharedPointer);
using kokkosfftCallbackStoreZ = void (*)(void *dataOut, std::size_t offset,
                                         double2 element, void *callerInfo,
                                         void *sharedPointer);
#endif

// Check the size of complex type
static_assert(sizeof(float2) == sizeof(Kokkos::complex<float>));
static_assert(alignof(float2) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(double2) == sizeof(Kokkos::complex<double>));
static_assert(alignof(double2) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType                     = int;
constexpr FFTDirectionType ROCFFT_FORWARD  = 1;
constexpr FFTDirectionType ROCFFT_BACKWARD = -1;

#if defined(KOKKOSFFT_ENABLE_CALLBACK)
// Unlike cuFFT/hipFFT, rocFFT has no callback type enum: load and store
// callbacks are set through two different functions and nothing checks the
// callback against the plan. These traits therefore describe each callback
// typedef (load or store, real or complex data, precision), so set_callback()
// can pick the right setter and check the callback against the plan itself.
//
// Primary template intentionally left undefined: instantiating it with an
// unsupported CallbackSymbol fails to compile with that type named in the
// error, instead of silently falling through.
template <typename CallbackSymbol>
struct deduce_callback_type;

template <>
struct deduce_callback_type<kokkosfftCallbackLoadR> {
  static constexpr bool is_load = true;
  static constexpr bool is_real = true;
  using value_type              = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackLoadD> {
  static constexpr bool is_load = true;
  static constexpr bool is_real = true;
  using value_type              = double;
};
template <>
struct deduce_callback_type<kokkosfftCallbackLoadC> {
  static constexpr bool is_load = true;
  static constexpr bool is_real = false;
  using value_type              = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackLoadZ> {
  static constexpr bool is_load = true;
  static constexpr bool is_real = false;
  using value_type              = double;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreR> {
  static constexpr bool is_load = false;
  static constexpr bool is_real = true;
  using value_type              = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreD> {
  static constexpr bool is_load = false;
  static constexpr bool is_real = true;
  using value_type              = double;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreC> {
  static constexpr bool is_load = false;
  static constexpr bool is_real = false;
  using value_type              = float;
};
template <>
struct deduce_callback_type<kokkosfftCallbackStoreZ> {
  static constexpr bool is_load = false;
  static constexpr bool is_real = false;
  using value_type              = double;
};

/// \brief Helper to check whether a callback symbol typedef is a load
/// callback (true) or a store callback (false)
template <typename CallbackSymbol>
inline constexpr bool is_load_callback_v =
    deduce_callback_type<CallbackSymbol>::is_load;
#endif

#if !defined(KOKKOSFFT_ENABLE_TPL_FFTW)
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };
#endif

template <typename ExecutionSpace>
using TransformType = FFTWTransformType;

// Helper to get input and output array type and direction from transform type
inline auto get_in_out_array_type(FFTWTransformType type, Direction direction) {
  rocfft_array_type in_array_type, out_array_type;
  rocfft_transform_type fft_direction;

  if (type == FFTWTransformType::C2C || type == FFTWTransformType::Z2Z) {
    in_array_type  = rocfft_array_type_complex_interleaved;
    out_array_type = rocfft_array_type_complex_interleaved;
    fft_direction  = direction == Direction::forward
                         ? rocfft_transform_type_complex_forward
                         : rocfft_transform_type_complex_inverse;
  } else if (type == FFTWTransformType::R2C || type == FFTWTransformType::D2Z) {
    in_array_type  = rocfft_array_type_real;
    out_array_type = rocfft_array_type_hermitian_interleaved;
    fft_direction  = rocfft_transform_type_real_forward;
  } else if (type == FFTWTransformType::C2R || type == FFTWTransformType::Z2D) {
    in_array_type  = rocfft_array_type_hermitian_interleaved;
    out_array_type = rocfft_array_type_real;
    fft_direction  = rocfft_transform_type_real_inverse;
  }

  return std::tuple<rocfft_array_type, rocfft_array_type,
                    rocfft_transform_type>(
      {in_array_type, out_array_type, fft_direction});
};

// Helper to convert the integer type of vectors
template <typename InType, typename OutType>
auto convert_int_type_and_reverse(const std::vector<InType> &in)
    -> std::vector<OutType> {
  std::vector<OutType> out(in.size());
  std::transform(
      in.cbegin(), in.cend(), out.begin(),
      [](const InType v) -> OutType { return static_cast<OutType>(v); });

  std::reverse(out.begin(), out.end());
  return out;
}

/// \brief A class that wraps rocfft_plan_description for RAII
struct ScopedRocfftPlanDescription {
 private:
  rocfft_plan_description m_description;

 public:
  ScopedRocfftPlanDescription() {
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_create(&m_description));
  }
  ~ScopedRocfftPlanDescription() noexcept {
    rocfft_status status = rocfft_plan_description_destroy(m_description);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_plan_description_destroy failed");
  }

  ScopedRocfftPlanDescription(const ScopedRocfftPlanDescription &) = delete;
  ScopedRocfftPlanDescription &operator=(const ScopedRocfftPlanDescription &) =
      delete;
  ScopedRocfftPlanDescription &operator=(ScopedRocfftPlanDescription &&) =
      delete;
  ScopedRocfftPlanDescription(ScopedRocfftPlanDescription &&) = delete;

  rocfft_plan_description description() const noexcept { return m_description; }
};

/// \brief A class that wraps rocfft_execution_info for RAII
struct ScopedRocfftExecutionInfo {
 private:
  rocfft_execution_info m_execution_info;

 public:
  ScopedRocfftExecutionInfo() {
    // Prepare workbuffer and set execution information
    KOKKOSFFT_CHECK_ROCFFT_CALL(
        rocfft_execution_info_create(&m_execution_info));
  }
  ~ScopedRocfftExecutionInfo() noexcept {
    rocfft_status status = rocfft_execution_info_destroy(m_execution_info);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_execution_info_destroy failed");
  }

  ScopedRocfftExecutionInfo(const ScopedRocfftExecutionInfo &) = delete;
  ScopedRocfftExecutionInfo &operator=(const ScopedRocfftExecutionInfo &) =
      delete;
  ScopedRocfftExecutionInfo &operator=(ScopedRocfftExecutionInfo &&) = delete;
  ScopedRocfftExecutionInfo(ScopedRocfftExecutionInfo &&)            = delete;

  rocfft_execution_info execution_info() const noexcept {
    return m_execution_info;
  }

  void set_work_area(void *workbuffer, std::size_t workbuffersize) {
    // Set work buffer
    if (workbuffersize > 0) {
      KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execution_info_set_work_buffer(
          m_execution_info, workbuffer, workbuffersize));
    }
  }

  void commit(const Kokkos::HIP &exec_space) {
    // set stream
    // NOTE: The stream must be of type hipStream_t.
    // It is an error to pass the address of a hipStream_t object.
    hipStream_t stream = exec_space.hip_stream();
    KOKKOSFFT_CHECK_ROCFFT_CALL(
        rocfft_execution_info_set_stream(m_execution_info, stream));
  }

  // The callback setters take void** arrays with one entry per brick of the
  // plan's input/output field. We never use fields, so each argument is a
  // single void* passed by address as an array of length 1. rocFFT reads
  // cb_functions[0] and cb_data[0] right away and keeps those values, not the
  // arrays, so passing the address of a local is fine here (unlike hipFFT,
  // which keeps the array pointers). The pointed-to callerInfo must still
  // outlive every execution, which the plan guarantees (see set_callback()).
  // shared_mem_bytes must currently be 0.
  void set_load_callback(void *callback, void *callback_params) {
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execution_info_set_load_callback(
        m_execution_info, &callback, &callback_params, 0));
  }

  void set_store_callback(void *callback, void *callback_params) {
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_execution_info_set_store_callback(
        m_execution_info, &callback, &callback_params, 0));
  }
};

/// \brief A class that wraps rocfft for RAII
template <typename T>
struct ScopedRocfftPlan {
 private:
  using floating_point_type    = KokkosFFT::Impl::base_floating_point_type<T>;
  rocfft_precision m_precision = std::is_same_v<floating_point_type, float>
                                     ? rocfft_precision_single
                                     : rocfft_precision_double;
  rocfft_plan m_plan;
  std::size_t m_workspace_size;
  //! Internal work buffer
  void *m_workbuffer = nullptr;
  std::unique_ptr<ScopedRocfftExecutionInfo> m_execution_info;
  //! Transform type, used to check store callbacks against the plan
  [[maybe_unused]] FFTWTransformType m_transform_type;
  //! Device copies of the callerInfo of the load/store callbacks, owned by
  //! this plan (see set_callback()) and freed in its destructor
  void *m_load_callback_params  = nullptr;
  void *m_store_callback_params = nullptr;

 public:
  ScopedRocfftPlan(const FFTWTransformType transform_type,
                   const std::vector<std::size_t> &in_extents,
                   const std::vector<std::size_t> &out_extents,
                   const std::vector<std::size_t> &fft_extents,
                   std::size_t howmany, Direction direction, bool is_inplace)
      : m_transform_type(transform_type) {
    auto [in_array_type, out_array_type, fft_direction] =
        get_in_out_array_type(transform_type, direction);

    // Compute dist and strides from extents
    std::size_t idist = total_size(in_extents);
    std::size_t odist = total_size(out_extents);

    auto in_strides  = compute_strides(in_extents);
    auto out_strides = compute_strides(out_extents);
    auto reversed_fft_extents =
        convert_int_type_and_reverse<std::size_t, std::size_t>(fft_extents);

    // Create a plan description
    ScopedRocfftPlanDescription scoped_description;
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_description_set_data_layout(
        scoped_description.description(),  // description handle
        in_array_type,                     // input array type
        out_array_type,                    // output array type
        nullptr,                           // offsets to start of input data
        nullptr,                           // offsets to start of output data
        in_strides.size(),                 // input stride length
        in_strides.data(),                 // input stride data
        idist,                             // input batch distance
        out_strides.size(),                // output stride length
        out_strides.data(),                // output stride data
        odist));

    // inplace or Out-of-place transform
    const rocfft_result_placement place =
        is_inplace ? rocfft_placement_inplace : rocfft_placement_notinplace;

    // Create a plan
    KOKKOSFFT_CHECK_ROCFFT_CALL(rocfft_plan_create(
        &m_plan, place, fft_direction, m_precision,
        reversed_fft_extents.size(),      // Dimension
        reversed_fft_extents.data(),      // Lengths
        howmany,                          // Number of transforms
        scoped_description.description()  // Description
        ));

    KOKKOSFFT_CHECK_ROCFFT_CALL(
        rocfft_plan_get_work_buffer_size(m_plan, &m_workspace_size));
    m_execution_info = std::make_unique<ScopedRocfftExecutionInfo>();
  }
  ~ScopedRocfftPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region(
        "KokkosFFT::cleanup_plan[TPL_rocfft]");
    if (m_workbuffer != nullptr) {
      Kokkos::kokkos_free<Kokkos::HIP>(m_workbuffer);
    }
    rocfft_status status = rocfft_plan_destroy(m_plan);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_plan_destroy failed");

    // rocFFT only borrows the callerInfo pointers set in set_callback() for
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

  ScopedRocfftPlan()                                    = delete;
  ScopedRocfftPlan(const ScopedRocfftPlan &)            = delete;
  ScopedRocfftPlan &operator=(const ScopedRocfftPlan &) = delete;
  ScopedRocfftPlan &operator=(ScopedRocfftPlan &&)      = delete;
  ScopedRocfftPlan(ScopedRocfftPlan &&)                 = delete;

  rocfft_plan plan() const noexcept { return m_plan; }
  rocfft_execution_info execution_info() const noexcept {
    return m_execution_info->execution_info();
  }

  /// \brief Return the workspace size in Byte
  /// \return the workspace size in Byte
  std::size_t workspace_size() const noexcept { return m_workspace_size; }

  template <typename WorkViewType>
  void set_work_area(const WorkViewType &work) {
    using value_type           = typename WorkViewType::non_const_value_type;
    std::size_t workspace_size = work.size() * sizeof(value_type);
    if (workspace_size > 0) {
      KOKKOSFFT_THROW_IF(
          workspace_size < m_workspace_size,
          "insufficient work buffer size. buffer size: " +
              std::to_string(workspace_size) +
              ", required size: " + std::to_string(m_workspace_size));
      void *work_area = static_cast<void *>(work.data());
      m_execution_info->set_work_area(work_area, workspace_size);
    }
  }

  void set_work_area() {
    if (m_workspace_size > 0) {
      m_workbuffer =
          Kokkos::kokkos_malloc<Kokkos::HIP>("workbuffer", m_workspace_size);
      m_execution_info->set_work_area(m_workbuffer, m_workspace_size);
    }
  }

  void commit(const Kokkos::HIP &exec_space) {
    m_execution_info->commit(exec_space);
  }

  /// \brief Attach a load or store callback to this plan, deduced from
  /// CallbackSymbolType (the callback typedef, e.g. kokkosfftCallbackLoadR vs
  /// kokkosfftCallbackStoreR, already encodes which one it is).
  ///
  /// The callback is checked against the plan, since rocFFT does not do it:
  /// its precision and, for a load callback, its real/complex data type must
  /// match the plan (compile-time check), and so must the real/complex data
  /// type of a store callback (runtime check, because a complex-to-real and a
  /// complex-to-complex plan have the same type). Each of load and store may
  /// be set at most once per plan, as for the other backends.
  ///
  /// \tparam CallbackSymbolType The type of the callback symbol
  /// \tparam CallbackParamsType The type of the caller-provided params
  /// \param[in] d_callback_symbol The __device__ global holding the callback
  /// function pointer
  /// \param[in] params The callback parameters. Copied into a device allocation
  /// owned by this ScopedRocfftPlan, freed in its destructor -- the caller
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
    using callback_traits = deduce_callback_type<CallbackSymbolType>;
    static_assert(
        std::same_as<typename callback_traits::value_type, floating_point_type>,
        "set_callback(): the precision of the callback does not match the "
        "precision of the plan");

    // Load and store callbacks each get their own device buffer -- the
    // caller-info is independent per callback type, and each may only be
    // set once per plan.
    void **callback_params_slot;
    if constexpr (callback_traits::is_load) {
      // The input of the plan is real (R2C/D2Z) or complex (C2R, C2C)
      static_assert(callback_traits::is_real == is_real_v<T>,
                    "set_callback(): a load callback must take the same kind "
                    "of data (real or complex) as the input of the plan");
      KOKKOSFFT_THROW_IF(
          m_load_callback_params != nullptr,
          "set_callback() may only set a load callback once per plan");
      callback_params_slot = &m_load_callback_params;
    } else {
      // The output of the plan is real only for C2R/Z2D transforms
      const bool is_real_output = m_transform_type == FFTWTransformType::C2R ||
                                  m_transform_type == FFTWTransformType::Z2D;
      KOKKOSFFT_THROW_IF(callback_traits::is_real != is_real_output,
                         "set_callback(): a store callback must take the same "
                         "kind of data (real or complex) as the output of the "
                         "plan");
      KOKKOSFFT_THROW_IF(
          m_store_callback_params != nullptr,
          "set_callback() may only set a store callback once per plan");
      callback_params_slot = &m_store_callback_params;
    }

    CallbackSymbolType callback{};
    // HIP_SYMBOL is a no-op on the AMD backend, but keeps this portable if
    // HIP is ever compiled through hip-nvcc.
    KOKKOSFFT_CHECK_HIP_CALL(hipMemcpyFromSymbol(
        &callback, HIP_SYMBOL(d_callback_symbol), sizeof(callback)));

    KOKKOSFFT_CHECK_HIP_CALL(
        hipMalloc(callback_params_slot, sizeof(CallbackParamsType)));
    KOKKOSFFT_CHECK_HIP_CALL(hipMemcpy(*callback_params_slot, &params,
                                       sizeof(CallbackParamsType),
                                       hipMemcpyHostToDevice));

    void *callback_ptr = reinterpret_cast<void *>(callback);
    if constexpr (callback_traits::is_load) {
      m_execution_info->set_load_callback(callback_ptr, *callback_params_slot);
    } else {
      m_execution_info->set_store_callback(callback_ptr, *callback_params_slot);
    }
#else
    static_assert(std::is_void_v<CallbackSymbolType>,
                  "set_callback() requires KokkosFFT to be built with "
                  "-DKokkosFFT_ENABLE_CALLBACK=ON");
#endif
  }
};

// Define fft transform types
template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, T1, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::R2C
                                                  : FFTWTransformType::D2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T2, float>
                                                  ? FFTWTransformType::C2R
                                                  : FFTWTransformType::Z2D;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct transform_type<ExecutionSpace, Kokkos::complex<T1>,
                      Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::C2C
                                                  : FFTWTransformType::Z2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)

template <typename ExecutionSpace>
struct FFTDataType {
  using float32 = float;
  using float64 = double;
  using complex64 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>, float2,
                         fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>, double2,
                         fftw_complex>;
};

/// \brief The index type used in backend FFT plan
/// rocFFT and FFTW use std::size_t and int as index type
/// \tparam ExecutionSpace The type of Kokkos execution space
template <typename ExecutionSpace>
using FFTIndexType =
    std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>, std::size_t,
                       int>;

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftw_plan_type   = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using rocfft_plan_type = ScopedRocfftPlan<T1>;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                  rocfft_plan_type, fftw_plan_type>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using fftw_plan_type   = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using rocfft_plan_type = ScopedRocfftPlan<T1>;
  using type = std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                                  rocfft_plan_type, fftw_plan_type>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? ROCFFT_FORWARD
                                                  : FFTW_FORWARD;

  static constexpr FFTDirectionType BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::HIP> ? ROCFFT_BACKWARD
                                                  : FFTW_BACKWARD;
  return direction == Direction::forward ? FORWARD : BACKWARD;
}
#else
template <typename ExecutionSpace>
struct FFTDataType {
  using float32    = float;
  using float64    = double;
  using complex64  = float2;
  using complex128 = double2;
};

template <typename ExecutionSpace>
using FFTIndexType = std::size_t;

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using type = ScopedRocfftPlan<T1>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using type = ScopedRocfftPlan<T1>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? ROCFFT_FORWARD : ROCFFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
