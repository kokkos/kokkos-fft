// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_FFTW_TYPES_HPP
#define KOKKOSFFT_FFTW_TYPES_HPP

#include <mutex>
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
template <typename T1, typename T2>
struct fftw_transform_type {
  static_assert(std::is_same_v<T1, T2>,
                "Real to real transform is unavailable");
};

template <typename T1, typename T2>
struct fftw_transform_type<T1, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::R2C
                                                  : FFTWTransformType::D2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename T1, typename T2>
struct fftw_transform_type<Kokkos::complex<T1>, T2> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T2, float>
                                                  ? FFTWTransformType::C2R
                                                  : FFTWTransformType::Z2D;
  static constexpr FFTWTransformType type() { return m_type; };
};

template <typename T1, typename T2>
struct fftw_transform_type<Kokkos::complex<T1>, Kokkos::complex<T2>> {
  static_assert(std::is_same_v<T1, T2>,
                "T1 and T2 should have the same precision");
  static constexpr FFTWTransformType m_type = std::is_same_v<T1, float>
                                                  ? FFTWTransformType::C2C
                                                  : FFTWTransformType::Z2Z;
  static constexpr FFTWTransformType type() { return m_type; };
};

/// \brief A class that wraps fftw_plan and fftwf_plan for RAII
template <typename ExecutionSpace, typename T1, typename T2>
struct ScopedFFTWPlan {
 private:
  using floating_point_type = KokkosFFT::Impl::base_floating_point_type<T1>;
  using plan_type =
      std::conditional_t<std::is_same_v<floating_point_type, float>, fftwf_plan,
                         fftw_plan>;
  plan_type m_plan;
  const int m_local_id;

 public:
  template <typename InScalarType, typename OutScalarType>
  ScopedFFTWPlan(const ExecutionSpace &exec_space, int rank, const int *n,
                 int howmany, InScalarType *in, const int *inembed, int istride,
                 int idist, OutScalarType *out, const int *onembed, int ostride,
                 int odist, [[maybe_unused]] int sign, unsigned flags)
      : m_local_id(global_id()) {
    init_threads(exec_space);
    constexpr auto type = fftw_transform_type<T1, T2>::type();
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
  }

  ~ScopedFFTWPlan() noexcept {
    cleanup_threads();
    if constexpr (std::is_same_v<plan_type, fftwf_plan>) {
      fftwf_destroy_plan(m_plan);
    } else {
      fftw_destroy_plan(m_plan);
    }
  }

  ScopedFFTWPlan()                                  = delete;
  ScopedFFTWPlan(const ScopedFFTWPlan &)            = delete;
  ScopedFFTWPlan &operator=(const ScopedFFTWPlan &) = delete;
  ScopedFFTWPlan &operator=(ScopedFFTWPlan &&)      = delete;
  ScopedFFTWPlan(ScopedFFTWPlan &&)                 = delete;

  plan_type plan() const noexcept { return m_plan; }

 private:
  static int global_id() {
    static int global_id = 0;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    return global_id++;
  }

  static void init_threads([[maybe_unused]] const ExecutionSpace &exec_space) {
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::is_same_v<ExecutionSpace,
                                 Kokkos::DefaultHostExecutionSpace>) {
      int nthreads = exec_space.concurrency();

      if constexpr (std::is_same_v<plan_type, fftwf_plan>) {
        if (m_local_id == 0) fftwf_init_threads();
        fftwf_plan_with_nthreads(nthreads);
      } else {
        if (m_local_id == 0) fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
      }
    }
#endif
  }

  static void cleanup_threads() {
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::is_same_v<ExecutionSpace,
                                 Kokkos::DefaultHostExecutionSpace>) {
      if constexpr (std::is_same_v<plan_type, fftwf_plan>) {
        if (m_local_id == 0) fftwf_cleanup_threads();
      } else {
        if (m_local_id == 0) fftw_cleanup_threads();
      }
    }
#endif
  }
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
