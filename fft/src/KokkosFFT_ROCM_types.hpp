// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ROCM_TYPES_HPP
#define KOKKOSFFT_ROCM_TYPES_HPP

#include <numeric>
#include <algorithm>
#include <complex>
#include <iostream>
#include <rocfft/rocfft.h>
#include <Kokkos_Abort.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_asserts.hpp"
#if defined(ENABLE_HOST_AND_DEVICE)
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

#if !defined(ENABLE_HOST_AND_DEVICE)
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };
#endif

template <typename ExecutionSpace>
using TransformType = FFTWTransformType;

/// \brief A class that wraps rocfft_plan_description for RAII
struct ScopedRocfftPlanDescription {
 private:
  rocfft_plan_description m_description;

 public:
  ScopedRocfftPlanDescription() {
    rocfft_status status = rocfft_plan_description_create(&m_description);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_create failed");
  }
  ~ScopedRocfftPlanDescription() noexcept {
    rocfft_status status = rocfft_plan_description_destroy(m_description);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_plan_description_destroy failed");
  }

  rocfft_plan_description description() const noexcept { return m_description; }
};

/// \brief A class that wraps rocfft_execution_info for RAII
template <typename FloatingPointType>
struct ScopedRocfftExecutionInfo {
 private:
  using BufferViewType =
      Kokkos::View<Kokkos::complex<FloatingPointType> *, Kokkos::HIP>;
  rocfft_execution_info m_execution_info;

  //! Internal work buffer
  BufferViewType m_buffer;

 public:
  ScopedRocfftExecutionInfo() {
    // Prepare workbuffer and set execution information
    rocfft_status status = rocfft_execution_info_create(&m_execution_info);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_execution_info_create failed");
  }
  ~ScopedRocfftExecutionInfo() noexcept {
    rocfft_status status = rocfft_execution_info_destroy(m_execution_info);
    if (status != rocfft_status_success)
      Kokkos::abort("rocfft_execution_info_destroy failed");
  }

  rocfft_execution_info execution_info() const noexcept {
    return m_execution_info;
  }

  void setup(const Kokkos::HIP &exec_space, std::size_t workbuffersize) {
    // set stream
    // NOTE: The stream must be of type hipStream_t.
    // It is an error to pass the address of a hipStream_t object.
    hipStream_t stream = exec_space.hip_stream();
    rocfft_status status =
        rocfft_execution_info_set_stream(m_execution_info, stream);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_execution_info_set_stream failed");

    // Set work buffer
    if (workbuffersize > 0) {
      m_buffer = BufferViewType("workbuffer", workbuffersize);
      status   = rocfft_execution_info_set_work_buffer(
          m_execution_info, (void *)m_buffer.data(), workbuffersize);
      KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                         "rocfft_execution_info_set_work_buffer failed");
    }
  }
};

/// \brief A class that wraps rocfft for RAII
template <typename T>
struct ScopedRocfftPlan {
 private:
  using floating_point_type = KokkosFFT::Impl::base_floating_point_type<T>;
  using ScopedRocfftExecutionInfoType =
      ScopedRocfftExecutionInfo<floating_point_type>;
  rocfft_precision m_precision = std::is_same_v<floating_point_type, float>
                                     ? rocfft_precision_single
                                     : rocfft_precision_double;
  rocfft_plan m_plan;
  std::unique_ptr<ScopedRocfftExecutionInfoType> m_execution_info;

 public:
  ScopedRocfftPlan(const FFTWTransformType transform_type,
                   const std::vector<int> &in_extents,
                   const std::vector<int> &out_extents,
                   const std::vector<int> &fft_extents, int howmany,
                   Direction direction, bool is_inplace) {
    auto [in_array_type, out_array_type, fft_direction] =
        get_in_out_array_type(transform_type, direction);

    // Compute dist and strides from extents
    int idist = std::accumulate(in_extents.begin(), in_extents.end(), 1,
                                std::multiplies<>());
    int odist = std::accumulate(out_extents.begin(), out_extents.end(), 1,
                                std::multiplies<>());

    auto in_strides  = compute_strides<int, std::size_t>(in_extents);
    auto out_strides = compute_strides<int, std::size_t>(out_extents);
    auto reversed_fft_extents =
        convert_int_type_and_reverse<int, std::size_t>(fft_extents);

    // Create a plan description
    ScopedRocfftPlanDescription scoped_description;
    rocfft_status status = rocfft_plan_description_set_data_layout(
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
        odist);                            // output batch distance

    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_description_set_data_layout failed");

    // inplace or Out-of-place transform
    const rocfft_result_placement place =
        is_inplace ? rocfft_placement_inplace : rocfft_placement_notinplace;

    // Create a plan
    status = rocfft_plan_create(&m_plan, place, fft_direction, m_precision,
                                reversed_fft_extents.size(),  // Dimension
                                reversed_fft_extents.data(),  // Lengths
                                howmany,  // Number of transforms
                                scoped_description.description()  // Description
    );
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_create failed");
  }
  ~ScopedRocfftPlan() noexcept {
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

  void commit(const Kokkos::HIP &exec_space) {
    std::size_t workbuffersize = 0;
    rocfft_status status =
        rocfft_plan_get_work_buffer_size(m_plan, &workbuffersize);
    KOKKOSFFT_THROW_IF(status != rocfft_status_success,
                       "rocfft_plan_get_work_buffer_size failed");

    m_execution_info = std::make_unique<ScopedRocfftExecutionInfo>();
    m_execution_info->setup(exec_space, workbuffersize);
  }

  // Helper to get input and output array type and direction from transform type
  auto get_in_out_array_type(FFTWTransformType type, Direction direction) {
    rocfft_array_type in_array_type, out_array_type;
    rocfft_transform_type fft_direction;

    if (type == FFTWTransformType::C2C || type == FFTWTransformType::Z2Z) {
      in_array_type  = rocfft_array_type_complex_interleaved;
      out_array_type = rocfft_array_type_complex_interleaved;
      fft_direction  = direction == Direction::forward
                           ? rocfft_transform_type_complex_forward
                           : rocfft_transform_type_complex_inverse;
    } else if (type == FFTWTransformType::R2C ||
               type == FFTWTransformType::D2Z) {
      in_array_type  = rocfft_array_type_real;
      out_array_type = rocfft_array_type_hermitian_interleaved;
      fft_direction  = rocfft_transform_type_real_forward;
    } else if (type == FFTWTransformType::C2R ||
               type == FFTWTransformType::Z2D) {
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

  // Helper to compute strides from extents
  // (n0, n1, n2) -> (1, n0, n0*n1)
  // (n0, n1) -> (1, n0)
  // (n0) -> (1)
  template <typename InType, typename OutType>
  auto compute_strides(const std::vector<InType> &extents)
      -> std::vector<OutType> {
    std::vector<OutType> out = {1};
    auto reversed_extents    = extents;
    std::reverse(reversed_extents.begin(), reversed_extents.end());

    for (std::size_t i = 1; i < reversed_extents.size(); i++) {
      out.push_back(static_cast<OutType>(reversed_extents.at(i - 1)) *
                    out.at(i - 1));
    }

    return out;
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

#if defined(ENABLE_HOST_AND_DEVICE)

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

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
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

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
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
