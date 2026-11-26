// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_FFTW_TYPES_HPP
#define KOKKOSFFT_FFTW_TYPES_HPP

#include <concepts>
#include <fftw3.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Convert_Types.hpp"
#include "KokkosFFT_Traits.hpp"

// Check the size of complex type
static_assert(sizeof(fftwf_complex) == sizeof(Kokkos::complex<float>));
static_assert(alignof(fftwf_complex) <= alignof(Kokkos::complex<float>));

static_assert(sizeof(fftw_complex) == sizeof(Kokkos::complex<double>));
static_assert(alignof(fftw_complex) <= alignof(Kokkos::complex<double>));

namespace KokkosFFT {
namespace Impl {

/// \brief A class that wraps fftw_plan and fftwf_plan for RAII
/// \tparam ExecutionSpace The execution space for the FFT plan
/// \tparam InValueType The value type of the input view
/// \tparam OutValueType The value type of the output view
template <typename ExecutionSpace, typename InValueType, typename OutValueType>
struct ScopedFFTWPlan {
  static_assert(
      have_same_base_floating_point_type_v<InValueType, OutValueType>,
      "ScopedFFTWPlan: input and output must have the same precision");

  //! Transform kind deduced from the input/output value types
  static constexpr TransformKind transform_kind =
      transform_kind_v<InValueType, OutValueType>;

 private:
  using floating_point_type = base_floating_point_type<InValueType>;
  using plan_type = std::conditional_t<std::same_as<floating_point_type, float>,
                                       fftwf_plan, fftw_plan>;
  plan_type m_plan;

 public:
  template <typename InScalarType, typename OutScalarType>
  ScopedFFTWPlan(const ExecutionSpace &exec_space, int rank, const int *n,
                 int howmany, InScalarType *in, const int *inembed, int istride,
                 int idist, OutScalarType *out, const int *onembed, int ostride,
                 int odist, Direction direction) {
    init_threads(exec_space);

    unsigned flags = FFTW_ESTIMATE;
    if constexpr (transform_kind == TransformKind::C2C) {
      auto sign =
          (direction == Direction::forward) ? FFTW_FORWARD : FFTW_BACKWARD;
      if constexpr (std::same_as<floating_point_type, float>) {
        m_plan =
            fftwf_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
                                out, onembed, ostride, odist, sign, flags);
      } else {
        m_plan =
            fftw_plan_many_dft(rank, n, howmany, in, inembed, istride, idist,
                               out, onembed, ostride, odist, sign, flags);
      }
    } else if constexpr (transform_kind == TransformKind::R2C) {
      if constexpr (std::same_as<floating_point_type, float>) {
        m_plan =
            fftwf_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                                    idist, out, onembed, ostride, odist, flags);
      } else {
        m_plan =
            fftw_plan_many_dft_r2c(rank, n, howmany, in, inembed, istride,
                                   idist, out, onembed, ostride, odist, flags);
      }
    } else if constexpr (transform_kind == TransformKind::C2R) {
      if constexpr (std::same_as<floating_point_type, float>) {
        m_plan =
            fftwf_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                                    idist, out, onembed, ostride, odist, flags);
      } else {
        m_plan =
            fftw_plan_many_dft_c2r(rank, n, howmany, in, inembed, istride,
                                   idist, out, onembed, ostride, odist, flags);
      }
    } else {
      static_assert(always_false<InValueType, OutValueType>,
                    "ScopedFFTWPlan: unsupported transform kind");
    }
  }

  ~ScopedFFTWPlan() noexcept {
    Kokkos::Profiling::ScopedRegion region("KokkosFFT::cleanup_plan[TPL_fftw]");
    if constexpr (std::same_as<floating_point_type, float>) {
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

  /// \brief Return the workspace size in Byte
  /// For FFTW, we do not need to manage workspace separately
  /// \return the workspace size in Byte
  std::size_t workspace_size() const noexcept { return 0; }

  template <typename WorkViewType>
  void set_work_area(const WorkViewType & /*work*/) const {}

 private:
  void init_threads([[maybe_unused]] const ExecutionSpace &exec_space) {
#if defined(KOKKOS_ENABLE_OPENMP) || defined(KOKKOS_ENABLE_THREADS)
    if constexpr (std::same_as<ExecutionSpace,
                               Kokkos::DefaultHostExecutionSpace>) {
      int nthreads = exec_space.concurrency();

      if constexpr (std::same_as<floating_point_type, float>) {
        fftwf_init_threads();
        fftwf_plan_with_nthreads(nthreads);
      } else {
        fftw_init_threads();
        fftw_plan_with_nthreads(nthreads);
      }
    }
#endif
  }
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
