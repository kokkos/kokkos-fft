// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_FFTW_TYPES_HPP
#define KOKKOSFFT_FFTW_TYPES_HPP

#include <concepts>
#include <fftw3.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>
#include "KokkosFFT_Asserts.hpp"
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

  //! fftw_alignment_of of the arrays used to create this plan, compared
  //! against the execution arrays by check_alignment()
  int m_in_alignment = 0, m_out_alignment = 0;

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
      static_assert(always_false_v<InValueType, OutValueType>,
                    "ScopedFFTWPlan: unsupported transform kind");
    }

    // Record the alignment of the planning arrays: FFTW plans embed the
    // SIMD alignment of the arrays they are created on, and the new-array
    // execute functions require the arrays passed at execution time to be
    // as aligned (see check_alignment()).
    m_in_alignment  = in ? alignment_of(in) : 0;
    m_out_alignment = out ? alignment_of(out) : 0;
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

  /// \brief Check the FFTW alignment rule for the arrays about to be
  /// passed to the new-array execute functions, and throw a clear error
  /// instead of the undefined behaviour (measured to segfault) FFTW would
  /// otherwise give.
  ///
  /// FFTW plans embed the SIMD alignment of the arrays used to create
  /// them, and executing on less-aligned arrays is undefined. The
  /// opposite direction is safe: a plan created on misaligned arrays
  /// uses non-SIMD codelets and runs on arrays of any alignment. Only
  /// the crashing direction is rejected (see PR #572).
  ///
  /// \param[in] in The input array of the execution
  /// \param[in] out The output array of the execution
  void check_alignment(const void *in, const void *out) const {
    check_alignment_impl(m_in_alignment, in, "input");
    check_alignment_impl(m_out_alignment, out, "output");
  }

 private:
  /// \brief fftw_alignment_of of an array, in the units of the plan's
  /// precision: float plans vectorize on floats, so their alignment is
  /// measured by fftwf_alignment_of, double plans by fftw_alignment_of.
  /// Both sides of every comparison made by check_alignment_impl() go
  /// through this same function, so the verdict matches what the plan's
  /// SIMD codelets assume.
  static int alignment_of(const void *ptr) {
    if constexpr (std::same_as<floating_point_type, float>) {
      return fftwf_alignment_of(
          reinterpret_cast<float *>(const_cast<void *>(ptr)));
    } else {
      return fftw_alignment_of(
          reinterpret_cast<double *>(const_cast<void *>(ptr)));
    }
  }

  static void check_alignment_impl(int plan_alignment, const void *ptr,
                                  const char *which) {
    // a null array (plans made from null Views, #580) has no alignment
    // to check, and fftw_alignment_of(nullptr) would be meaningless
    if (ptr == nullptr) {
      return;
    }

    int exec_alignment = alignment_of(ptr);
    KOKKOSFFT_THROW_IF(
        plan_alignment == 0 && exec_alignment != 0,
        "executing an FFTW plan on a less aligned array than the one used "
        "to create it is not allowed: this plan was created on an aligned " +
            std::string(which) + " array (fftw_alignment_of == 0), but the " +
            "execution " + which + " array is misaligned (fftw_alignment_of == " +
            std::to_string(exec_alignment) + "). FFTW plans embed the SIMD " +
            "alignment of the planning arrays; reuse a plan only with Views " +
            "at least as aligned as those used to create it");
  }
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
