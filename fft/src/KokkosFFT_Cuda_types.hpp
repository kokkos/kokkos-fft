// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_CUDA_TYPES_HPP
#define KOKKOSFFT_CUDA_TYPES_HPP

#include <cufft.h>
#include "KokkosFFT_common_types.hpp"

// Check the size of complex type
static_assert(sizeof(cufftComplex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(cufftComplex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(cufftDoubleComplex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(cufftDoubleComplex) <= alignof(Kokkos::complex<double>));

#if defined(ENABLE_HOST_AND_DEVICE)
#include <fftw3.h>
#include "KokkosFFT_utils.hpp"
static_assert(sizeof(fftwf_complex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(fftwf_complex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(fftw_complex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(fftw_complex) <= alignof(Kokkos::complex<double>));
#endif

namespace KokkosFFT {
namespace Impl {
using FFTDirectionType = int;

// Unused
template <typename ExecutionSpace>
using FFTInfoType = int;

/// \brief A class that wraps cufft for RAII
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedCufftPlanType {
  cufftHandle m_plan;

  ScopedCufftPlanType() { cufftCreate(&m_plan); }
  ~ScopedCufftPlanType() { cufftDestroy(m_plan); }

  cufftHandle &plan() { return m_plan; }
};

#if defined(ENABLE_HOST_AND_DEVICE)
enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };

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

/// \brief A class that wraps fftw_plan and fftwf_plan for RAII
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedFFTWPlanType {
  using floating_point_type = KokkosFFT::Impl::base_floating_point_type<T1>;
  using plan_type =
      std::conditional_t<std::is_same_v<floating_point_type, float>, fftwf_plan,
                         fftw_plan>;
  plan_type m_plan;
  bool m_is_created = false;

  ScopedFFTWPlanType() {}
  ~ScopedFFTWPlanType() {
    cleanup_threads<floating_point_type>();
    if constexpr (std::is_same_v<floating_point_type, float>) {
      if (m_is_created) fftwf_destroy_plan(m_plan);
    } else {
      if (m_is_created) fftw_destroy_plan(m_plan);
    }
  }

  const plan_type &plan() const { return m_plan; }

  template <typename InScalarType, typename OutScalarType>
  void create(const ExecutionSpace &exec_space, int rank, const int *n,
              int howmany, InScalarType *in, const int *inembed, int istride,
              int idist, OutScalarType *out, const int *onembed, int ostride,
              int odist, [[maybe_unused]] int sign, unsigned flags) {
    init_threads<floating_point_type>(exec_space);

    constexpr auto type =
        KokkosFFT::Impl::transform_type<ExecutionSpace, T1, T2>::type();

    if constexpr (type == KokkosFFT::Impl::FFTWTransformType::R2C) {
      m_plan =
          fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist,
                                  out, onembed, ostride, odist, flags);
    } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::D2Z) {
      m_plan =
          fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride, idist,
                                 out, onembed, ostride, odist, flags);
    } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::C2R) {
      m_plan =
          fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
                                  out, onembed, ostride, odist, flags);
    } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::Z2D) {
      m_plan =
          fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride, idist,
                                 out, onembed, ostride, odist, flags);
    } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::C2C) {
      m_plan =
          fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
                              out, onembed, ostride, odist, sign, flags);
    } else if constexpr (type == KokkosFFT::Impl::FFTWTransformType::Z2Z) {
      m_plan = fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
                                  out, onembed, ostride, odist, sign, flags);
    }
    m_is_created = true;
  }

 private:
  template <typename T>
  void init_threads([[maybe_unused]] const ExecutionSpace &exec_space) {
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    int nthreads = exec_space.concurrency();

    if constexpr (std::is_same_v<T, float>) {
      fftwf_init_threads();
      fftwf_plan_with_nthreads(nthreads);
    } else {
      fftw_init_threads();
      fftw_plan_with_nthreads(nthreads);
    }
#endif
  }

  template <typename T>
  void cleanup_threads() {
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::is_same_v<T, float>) {
      fftwf_cleanup_threads();
    } else {
      fftw_cleanup_threads();
    }
#endif
  }
};

template <typename ExecutionSpace, typename T1, typename T2>
struct FFTPlanType {
  using fftw_plan_type  = ScopedFFTWPlanType<ExecutionSpace, T1, T2>;
  using cufft_plan_type = ScopedCufftPlanType<ExecutionSpace, T1, T2>;
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
  using type = ScopedCufftPlanType<ExecutionSpace, T1, T2>;
};

template <typename ExecutionSpace>
auto direction_type(Direction direction) {
  return direction == Direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE;
}
#endif
}  // namespace Impl
}  // namespace KokkosFFT

#endif
