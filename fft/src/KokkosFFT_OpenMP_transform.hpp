#ifndef __KOKKOSFFT_OPENMP_TRANSFORM_HPP__
#define __KOKKOSFFT_OPENMP_TRANSFORM_HPP__

#include <fftw3.h>

namespace KokkosFFT {
  template <typename PlanType>
  void _exec(PlanType plan, [[maybe_unused]] float* idata, [[maybe_unused]] fftwf_complex* odata, [[maybe_unused]] int direction) {
    fftwf_execute(plan);
  }

  template <typename PlanType>
  void _exec(PlanType plan, [[maybe_unused]] double* idata, [[maybe_unused]] fftw_complex* odata, [[maybe_unused]] int direction) {
    fftw_execute(plan);
  }

  template <typename PlanType>
  void _exec(PlanType plan, [[maybe_unused]] fftwf_complex* idata, [[maybe_unused]] float* odata, [[maybe_unused]] int direction) {
    fftwf_execute(plan);
  }

  template <typename PlanType>
  void _exec(PlanType plan, [[maybe_unused]] fftw_complex* idata, [[maybe_unused]] double* odata, [[maybe_unused]] int direction) {
    fftw_execute(plan);
  }

  template <typename PlanType>
  void _exec(PlanType plan, [[maybe_unused]] fftwf_complex* idata, [[maybe_unused]] fftwf_complex* odata, [[maybe_unused]] int direction) {
    fftwf_execute(plan);
  }

  template <typename PlanType>
  void _exec(PlanType plan, [[maybe_unused]] fftw_complex* idata, [[maybe_unused]] fftw_complex* odata, [[maybe_unused]] int direction) {
    fftw_execute(plan);
  }
};

#endif