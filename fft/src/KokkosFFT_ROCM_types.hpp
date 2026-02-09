// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TYPES_HPP
#define KOKKOSFFT_ROCM_TYPES_HPP

#include <numeric>
#include <algorithm>
#include <complex>
#include <rocfft/rocfft.h>
#include <Kokkos_Abort.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_ROCM_asserts.hpp"
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

// Check the size of complex type
static_assert(sizeof(std::complex<float>) == sizeof(Kokkos::complex<float>));
static_assert(alignof(std::complex<float>) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(std::complex<double>) == sizeof(Kokkos::complex<double>));
static_assert(alignof(std::complex<double>) <=
              alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType                     = int;
constexpr FFTDirectionType ROCFFT_FORWARD  = 1;
constexpr FFTDirectionType ROCFFT_BACKWARD = -1;

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

 public:
  ScopedRocfftPlan(const FFTWTransformType transform_type,
                   const std::vector<std::size_t> &in_extents,
                   const std::vector<std::size_t> &out_extents,
                   const std::vector<std::size_t> &fft_extents,
                   std::size_t howmany, Direction direction, bool is_inplace) {
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
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         std::complex<float>, fftwf_complex>;
  using complex128 =
      std::conditional_t<std::is_same_v<ExecutionSpace, Kokkos::HIP>,
                         std::complex<double>, fftw_complex>;
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
  using complex64  = std::complex<float>;
  using complex128 = std::complex<double>;
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
