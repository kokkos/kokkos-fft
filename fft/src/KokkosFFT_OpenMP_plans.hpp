#ifndef KOKKOSFFT_OPENMP_PLANS_HPP
#define KOKKOSFFT_OPENMP_PLANS_HPP

#include <numeric>
#include "KokkosFFT_OpenMP_types.hpp"
#include "KokkosFFT_layouts.hpp"

namespace KokkosFFT {
  template <typename ExecutionSpace, typename T>
  void _init_threads(const ExecutionSpace& exec_space) {
    int nthreads = exec_space.concurrency();

    if constexpr (std::is_same_v<T, float>) {
      fftwf_init_threads();
      fftwf_plan_with_nthreads(nthreads);
    } else {
      fftw_init_threads();
      fftw_plan_with_nthreads(nthreads);
    }
  }

  // ND transform
  template <typename ExecutionSpace, typename PlanType, typename InViewType, typename OutViewType>
  auto _create(const ExecutionSpace& exec_space, PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType sign) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    _init_threads<ExecutionSpace, real_type_t<in_value_type>>(exec_space);

    const int rank = InViewType::rank();
    const int axis = -1;
    const int howmany = 1;
    constexpr auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents] = get_extents(in, out, axis);
    int idist = std::accumulate(in_extents.begin(), in_extents.end(), 1, std::multiplies<>());
    int odist = std::accumulate(out_extents.begin(), out_extents.end(), 1, std::multiplies<>());
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());

    auto* idata = reinterpret_cast<typename fft_data_type<in_value_type>::type*>(in.data());
    auto* odata = reinterpret_cast<typename fft_data_type<out_value_type>::type*>(out.data());
    int istride = 1, ostride = 1;

    if constexpr(type == TransformType::R2C) {
      plan = fftwf_plan_many_dft_r2c(rank,
                                     fft_extents.data(),
                                     howmany,
                                     idata,
                                     in_extents.data(),
                                     istride,
                                     idist,
                                     odata,
                                     out_extents.data(),
                                     ostride,
                                     odist,
                                     FFTW_ESTIMATE
                                    );
    } else if constexpr(type == TransformType::D2Z) {
      plan = fftw_plan_many_dft_r2c(rank,
                                    fft_extents.data(),
                                    howmany,
                                    idata,
                                    in_extents.data(),
                                    istride,
                                    idist,
                                    odata,
                                    out_extents.data(),
                                    ostride,
                                    odist,
                                    FFTW_ESTIMATE
                                   );
    } else if constexpr(type == TransformType::C2R) {
      plan = fftwf_plan_many_dft_c2r(rank,
                                     fft_extents.data(),
                                     howmany,
                                     idata,
                                     in_extents.data(),
                                     istride,
                                     idist,
                                     odata,
                                     out_extents.data(),
                                     ostride,
                                     odist,
                                     FFTW_ESTIMATE
                                    );
    } else if constexpr(type == TransformType::Z2D) {
      plan = fftw_plan_many_dft_c2r(rank,
                                    fft_extents.data(),
                                    howmany,
                                    idata,
                                    in_extents.data(),
                                    istride,
                                    idist,
                                    odata,
                                    out_extents.data(),
                                    ostride,
                                    odist,
                                    FFTW_ESTIMATE
                                   );
    } else if constexpr(type == TransformType::C2C) {
      plan = fftwf_plan_many_dft(rank,
                                 fft_extents.data(),
                                 howmany,
                                 idata,
                                 in_extents.data(),
                                 istride,
                                 idist,
                                 odata,
                                 out_extents.data(),
                                 ostride,
                                 odist,
                                 sign,
                                 FFTW_ESTIMATE
                                );
    } else if constexpr(type == TransformType::Z2Z) {
      plan = fftw_plan_many_dft(rank,
                                fft_extents.data(),
                                howmany,
                                idata,
                                in_extents.data(),
                                istride,
                                idist,
                                odata,
                                out_extents.data(),
                                ostride,
                                odist,
                                sign,
                                FFTW_ESTIMATE
                               );
    }

    return fft_size;
  }

  // batched transform, over ND Views
  template <typename ExecutionSpace, typename PlanType, typename InViewType, typename OutViewType, std::size_t fft_rank=1>
  auto _create(const ExecutionSpace& exec_space, PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType sign, axis_type<fft_rank> axes) {
    static_assert(Kokkos::is_view<InViewType>::value,
                  "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                  "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(InViewType::rank() >= fft_rank,
                  "KokkosFFT::_create: Rank of View must be larger than Rank of FFT.");
    const int rank = fft_rank;

    _init_threads<ExecutionSpace, real_type_t<in_value_type>>(exec_space);

    constexpr auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents, howmany] = get_extents_batched(in, out, axes);
    int idist = std::accumulate(in_extents.begin(), in_extents.end(), 1, std::multiplies<>());
    int odist = std::accumulate(out_extents.begin(), out_extents.end(), 1, std::multiplies<>());
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());

    auto* idata = reinterpret_cast<typename fft_data_type<in_value_type>::type*>(in.data());
    auto* odata = reinterpret_cast<typename fft_data_type<out_value_type>::type*>(out.data());

    // For the moment, considering the contiguous layout only
    int istride = 1, ostride = 1;

    if constexpr(type == TransformType::R2C) {
      plan = fftwf_plan_many_dft_r2c(rank,
                                     fft_extents.data(),
                                     howmany,
                                     idata,
                                     in_extents.data(),
                                     istride,
                                     idist,
                                     odata,
                                     out_extents.data(),
                                     ostride,
                                     odist,
                                     FFTW_ESTIMATE
                                    );
    } else if constexpr(type == TransformType::D2Z) {
      plan = fftw_plan_many_dft_r2c(rank,
                                    fft_extents.data(),
                                    howmany,
                                    idata,
                                    in_extents.data(),
                                    istride,
                                    idist,
                                    odata,
                                    out_extents.data(),
                                    ostride,
                                    odist,
                                    FFTW_ESTIMATE
                                   );
    } else if constexpr(type == TransformType::C2R) {
      plan = fftwf_plan_many_dft_c2r(rank,
                                     fft_extents.data(),
                                     howmany,
                                     idata,
                                     in_extents.data(),
                                     istride,
                                     idist,
                                     odata,
                                     out_extents.data(),
                                     ostride,
                                     odist,
                                     FFTW_ESTIMATE
                                    );
    } else if constexpr(type == TransformType::Z2D) {
      plan = fftw_plan_many_dft_c2r(rank,
                                    fft_extents.data(),
                                    howmany,
                                    idata,
                                    in_extents.data(),
                                    istride,
                                    idist,
                                    odata,
                                    out_extents.data(),
                                    ostride,
                                    odist,
                                    FFTW_ESTIMATE
                                   );
    } else if constexpr(type == TransformType::C2C) {
      plan = fftwf_plan_many_dft(rank,
                                 fft_extents.data(),
                                 howmany,
                                 idata,
                                 in_extents.data(),
                                 istride,
                                 idist,
                                 odata,
                                 out_extents.data(),
                                 ostride,
                                 odist,
                                 sign,
                                 FFTW_ESTIMATE
                                );
    } else if constexpr(type == TransformType::Z2Z) {
      plan = fftw_plan_many_dft(rank,
                                fft_extents.data(),
                                howmany,
                                idata,
                                in_extents.data(),
                                istride,
                                idist,
                                odata,
                                out_extents.data(),
                                ostride,
                                odist,
                                sign,
                                FFTW_ESTIMATE
                               );
    }

    return fft_size;
  }

  template <typename T>
  void _destroy(typename FFTPlanType<T>::type& plan) {
    if constexpr (std::is_same_v<T, float>) {
      fftwf_destroy_plan(plan);
    } else {
      fftw_destroy_plan(plan);
    }
  }
};

#endif