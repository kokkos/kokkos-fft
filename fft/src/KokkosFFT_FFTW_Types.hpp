// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_FFTW_TYPES_HPP
#define KOKKOSFFT_FFTW_TYPES_HPP

#include <fftw3.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

// Check the size of complex type
static_assert(sizeof(fftwf_complex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(fftwf_complex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(fftw_complex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(fftw_complex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {

enum class FFTWTransformType { R2C, D2Z, C2R, Z2D, C2C, Z2Z };

// Define fft transform types
template <typename ExecutionSpace, typename T1, typename T2>
struct fftw_transform_type {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename ExecutionSpace, typename T1, typename T2>
struct fftw_transform_type<ExecutionSpace, T1, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::R2C
                                                  : FFTWTransformType::D2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct fftw_transform_type<ExecutionSpace, Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T2, float>
                                                  ? FFTWTransformType::C2R
                                                  : FFTWTransformType::Z2D;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename ExecutionSpace, typename T1, typename T2>
struct fftw_transform_type<ExecutionSpace, Kokkos::complex<T1>,
                           Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::C2C
                                                  : FFTWTransformType::Z2Z;
  static constexpr FFTWTransformType type() { return m_type; };
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

  plan_type &plan() const { return m_plan; }

  template <typename InScalarType, typename OutScalarType>
  void create(const ExecutionSpace &exec_space, int rank, const int *n,
              int howmany, InScalarType *in, const int *inembed, int istride,
              int idist, OutScalarType *out, const int *onembed, int ostride,
              int odist, [[maybe_unused]] int sign, unsigned flags) {
    init_threads<floating_point_type>(exec_space);

    constexpr auto type = fftw_transform_type<ExecutionSpace, T1, T2>::type();

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

}  // namespace Impl
}  // namespace KokkosFFT

#endif
