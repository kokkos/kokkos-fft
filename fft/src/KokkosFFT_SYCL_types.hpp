// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_SYCL_TYPES_HPP
#define KOKKOSFFT_SYCL_TYPES_HPP

#include <complex>
#include <sycl/sycl.hpp>
#include <mkl.h>
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250100
#include <oneapi/mkl/dft.hpp>
#else
#include <oneapi/mkl/dfti.hpp>
#endif
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
#include "KokkosFFT_FFTW_Types.hpp"
#endif

// Check the size of complex type
// [TO DO] I guess this kind of test is already made by Kokkos itself
static_assert(sizeof(std::complex<float>) == sizeof(Kokkos::complex<float>));
static_assert(alignof(std::complex<float>) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(std::complex<double>) == sizeof(Kokkos::complex<double>));
static_assert(alignof(std::complex<double>) <=
              alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType                      = int;
constexpr FFTDirectionType MKL_FFT_FORWARD  = 1;
constexpr FFTDirectionType MKL_FFT_BACKWARD = -1;

#if !defined(KOKKOSFFT_ENABLE_TPL_FFTW)
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };
#endif

template <typename ExecutionSpace>
using TransformType = FFTWTransformType;

/// \brief A class that wraps oneMKL for RAII
template <typename T1, typename T2>
struct ScopedoneMKLPlan {
  static_assert(have_same_base_floating_point_type_v<T1, T2>,
                "ScopedoneMKLPlan: must be constructed with the same base "
                "floating point type");
  using floating_point_type = KokkosFFT::Impl::base_floating_point_type<T1>;
  static constexpr oneapi::mkl::dft::precision prec =
      std::is_same_v<floating_point_type, float>
          ? oneapi::mkl::dft::precision::SINGLE
          : oneapi::mkl::dft::precision::DOUBLE;
  static constexpr oneapi::mkl::dft::domain dom =
      is_complex_v<T1> && is_complex_v<T2> ? oneapi::mkl::dft::domain::COMPLEX
                                           : oneapi::mkl::dft::domain::REAL;
  using onemklHandle = oneapi::mkl::dft::descriptor<prec, dom>;

  onemklHandle m_plan;
  std::size_t m_workspace_size = 0;

 public:
  ScopedoneMKLPlan(const std::vector<std::int64_t> &lengths,
                   const std::vector<std::int64_t> &in_strides,
                   const std::vector<std::int64_t> &out_strides,
                   std::int64_t max_idist, std::int64_t max_odist,
                   std::int64_t howmany, KokkosFFT::Direction direction,
                   bool is_inplace)
      : m_plan(lengths) {
#if defined(INTEL_MKL_VERSION) && INTEL_MKL_VERSION >= 20250100
    const oneapi::mkl::dft::config_value placement =
        is_inplace ? oneapi::mkl::dft::config_value::INPLACE
                   : oneapi::mkl::dft::config_value::NOT_INPLACE;
    const oneapi::mkl::dft::config_value storage =
        oneapi::mkl::dft::config_value::COMPLEX_COMPLEX;
    auto fwd_strides =
        direction == KokkosFFT::Direction::forward ? in_strides : out_strides;
    auto bwd_strides =
        direction == KokkosFFT::Direction::forward ? out_strides : in_strides;
    m_plan.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, fwd_strides);
    m_plan.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, bwd_strides);
    m_plan.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE, storage);
#else
    const DFTI_CONFIG_VALUE placement =
        is_inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE;
    const DFTI_CONFIG_VALUE storage = DFTI_COMPLEX_COMPLEX;
    m_plan.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                     in_strides.data());
    m_plan.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                     out_strides.data());
    m_plan.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                     storage);
#endif

    // Configuration for batched plan
    m_plan.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, max_idist);
    m_plan.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, max_odist);
    m_plan.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                     howmany);

    // Data layout in conjugate-even domain
    m_plan.set_value(oneapi::mkl::dft::config_param::PLACEMENT, placement);
  }

  ScopedoneMKLPlan()                                    = delete;
  ScopedoneMKLPlan(const ScopedoneMKLPlan &)            = delete;
  ScopedoneMKLPlan &operator=(const ScopedoneMKLPlan &) = delete;
  ScopedoneMKLPlan &operator=(ScopedoneMKLPlan &&)      = delete;
  ScopedoneMKLPlan(ScopedoneMKLPlan &&)                 = delete;

  onemklHandle &plan() noexcept { return m_plan; }

  void use_external_workspace() {
    // Use externally allocated workspaces
    m_plan.set_value(oneapi::mkl::dft::config_param::WORKSPACE_PLACEMENT,
                     oneapi::mkl::dft::config_value::WORKSPACE_EXTERNAL);
  }

  /// \brief Return the workspace size in Byte
  /// \return the workspace size in Byte
  void set_workspace_size() {
    std::int64_t workspace_bytes = -1;
    m_plan.get_value(oneapi::mkl::dft::config_param::WORKSPACE_EXTERNAL_BYTES,
                     &workspace_bytes);

    KOKKOSFFT_THROW_IF(workspace_bytes < 0,
                       "get_value: WORKSPACE_EXTERNAL_BYTES failed");
    m_workspace_size = static_cast<std::size_t>(workspace_bytes);
  }

  /// \brief Return the workspace size in Byte
  /// \return the workspace size in Byte
  std::size_t workspace_size() const noexcept { return m_workspace_size; }

  template <typename WorkViewType>
  void set_work_area([[maybe_unused]] const WorkViewType &work) {
    using value_type           = typename WorkViewType::non_const_value_type;
    std::size_t workspace_size = work.size() * sizeof(value_type);
    if (m_workspace_size > 0) {
      KOKKOSFFT_THROW_IF(
          workspace_size < m_workspace_size,
          "insufficient work buffer size. buffer size: " +
              std::to_string(workspace_size) +
              ", required size: " + std::to_string(m_workspace_size));
      floating_point_type *work_area =
          reinterpret_cast<floating_point_type *>(work.data());
      m_plan.set_workspace(work_area);
    }
  }

  void commit(const Kokkos::SYCL &exec_space) {
    sycl::queue q = exec_space.sycl_queue();
    m_plan.commit(q);
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

  using complex64 = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
      std::complex<float>, fftwf_complex>;
  using complex128 = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>,
      std::complex<double>, fftw_complex>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftwHandle   = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using onemklHandle = ScopedoneMKLPlan<T1, T2>;
  using type         = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>, onemklHandle,
      fftwHandle>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using fftwHandle   = ScopedFFTWPlan<ExecutionSpace, T1, T2>;
  using onemklHandle = ScopedoneMKLPlan<T1, T2>;
  using type         = std::conditional_t<
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>, onemklHandle,
      fftwHandle>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  static constexpr FFTDirectionType FORWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>
          ? MKL_FFT_FORWARD
          : FFTW_FORWARD;
  static constexpr FFTDirectionType BACKWARD =
      std::is_same_v<ExecutionSpace, Kokkos::Experimental::SYCL>
          ? MKL_FFT_BACKWARD
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
  using type = ScopedoneMKLPlan<T1, T2>;
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTDynPlanType {
  using type = ScopedoneMKLPlan<T1, T2>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? MKL_FFT_FORWARD : MKL_FFT_BACKWARD;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
