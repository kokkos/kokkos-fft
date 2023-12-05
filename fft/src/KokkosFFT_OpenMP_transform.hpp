#ifndef __KOKKOSFFT_OPENMP_TRANSFORM_HPP__
#define __KOKKOSFFT_OPENMP_TRANSFORM_HPP__

#include <fftw3.h>

namespace KokkosFFT {
  template <typename PlanType>
  void _exec(PlanType plan, float* idata, fftwf_complex* odata, [[maybe_unused]] int direction) {
    fftwf_execute_dft_r2c(plan, idata, odata);
  }

  template <typename PlanType>
  void _exec(PlanType plan, double* idata, fftw_complex* odata, [[maybe_unused]] int direction) {
    fftw_execute_dft_r2c(plan, idata, odata);
  }

  template <typename PlanType>
  void _exec(PlanType plan, fftwf_complex* idata, float* odata, [[maybe_unused]] int direction) {
    fftwf_execute_dft_c2r(plan, idata, odata);
  }

  template <typename PlanType>
  void _exec(PlanType plan, fftw_complex* idata, double* odata, [[maybe_unused]] int direction) {
    fftw_execute_dft_c2r(plan, idata, odata);
  }

  template <typename PlanType>
  void _exec(PlanType plan, fftwf_complex* idata, fftwf_complex* odata, [[maybe_unused]] int direction) {
    fftwf_execute_dft(plan, idata, odata);
  }

  template <typename PlanType>
  void _exec(PlanType plan, fftw_complex* idata, fftw_complex* odata, [[maybe_unused]] int direction) {
    fftw_execute_dft(plan, idata, odata);
  }
};

#endif