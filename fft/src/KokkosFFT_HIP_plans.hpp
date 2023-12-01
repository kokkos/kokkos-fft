#ifndef __KOKKOSFFT_HIP_PLANS_HPP__
#define __KOKKOSFFT_HIP_PLANS_HPP__

#include <numeric>
#include "KokkosFFT_HIP_types.hpp"
#include "KokkosFFT_layouts.hpp"

namespace KokkosFFT {
  // 1D transform 
  template <typename PlanType, typename InViewType, typename OutViewType,
            std::enable_if_t<InViewType::rank()==1, std::nullptr_t> = nullptr>
  auto _create(PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType direction) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftCreate failed");

    const int batch = 1;
    const int axis = 0;

    auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents] = get_extents(in, out, axis);
    const int nx = fft_extents.at(0);
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());
    
    hipfft_rt = hipfftPlan1d(&plan, nx, type, batch);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftPlan1d failed");
    return fft_size;
  }

  // 2D transform 
  template <typename PlanType, typename InViewType, typename OutViewType,
            std::enable_if_t<InViewType::rank()==2, std::nullptr_t> = nullptr>
  auto _create(PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType direction) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    
    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftCreate failed");

    const int axis = 0;
    auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents] = get_extents(in, out, axis);
    const int nx = fft_extents.at(0), ny = fft_extents.at(1);
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());

    hipfft_rt = hipfftPlan2d(&plan, nx, ny, type);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftPlan2d failed");
    return fft_size;
  }

  // 3D transform 
  template <typename PlanType, typename InViewType, typename OutViewType,
            std::enable_if_t<InViewType::rank()==3, std::nullptr_t> = nullptr>
  auto _create(PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType direction) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftCreate failed");

    const int batch = 1;
    const int axis = 0;

    auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents] = get_extents(in, out, axis);
    
    const int nx = fft_extents.at(0), ny = fft_extents.at(1), nz = fft_extents.at(2);
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());

    hipfft_rt = hipfftPlan3d(&plan, nx, ny, nz, type);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftPlan3d failed");
    return fft_size;
  }

  // ND transform 
  template <typename PlanType, typename InViewType, typename OutViewType,
            std::enable_if_t< std::isgreater(InViewType::rank(), 3), std::nullptr_t> = nullptr>
  auto _create(PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType direction) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftCreate failed");

    const int rank = InViewType::rank();
    const int batch = 1;
    const int axis = 0;
    auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents] = get_extents(in, out, axis);
    int idist = std::accumulate(in_extents.begin(), in_extents.end(), 1, std::multiplies<>());
    int odist = std::accumulate(out_extents.begin(), out_extents.end(), 1, std::multiplies<>());
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());
    
    hipfft_rt = hipfftPlanMany(
                 &plan, 
                 rank,
                 fft_extents.data(),
                 nullptr,
                 1,
                 idist,
                 nullptr,
                 1,
                 odist,
                 type,
                 batch);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftPlanMany failed");
    return fft_size;
  }

  // batched transform, over ND Views
  template <typename PlanType, typename InViewType, typename OutViewType, std::size_t fft_rank=1>
  auto _create(PlanType& plan, const InViewType& in, const OutViewType& out, [[maybe_unused]] FFTDirectionType direction, axis_type<fft_rank> axes) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_create: OutViewType is not a Kokkos::View.");
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(InViewType::rank() >= fft_rank,
                  "KokkosFFT::_create: Rank of View must be larger than Rank of FFT.");
    const int rank = fft_rank;
    constexpr auto type = transform_type<in_value_type, out_value_type>::type();
    auto [in_extents, out_extents, fft_extents, howmany] = get_extents_batched(in, out, axes);
    int idist = std::accumulate(in_extents.begin(), in_extents.end(), 1, std::multiplies<>());
    int odist = std::accumulate(out_extents.begin(), out_extents.end(), 1, std::multiplies<>());
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());

    auto* idata = reinterpret_cast<typename fft_data_type<in_value_type>::type*>(in.data());
    auto* odata = reinterpret_cast<typename fft_data_type<out_value_type>::type*>(out.data());

    // For the moment, considering the contiguous layout only
    int istride = 1, ostride = 1;

    hipfftResult hipfft_rt = hipfftCreate(&plan);
    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftCreate failed");

    hipfft_rt = hipfftPlanMany(
                 &plan, 
                 rank,
                 fft_extents.data(),
                 in_extents.data(),
                 istride,
                 idist,
                 out_extents.data(),
                 ostride,
                 odist,
                 type,
                 howmany);

    if(hipfft_rt != HIPFFT_SUCCESS)
      throw std::runtime_error("hipfftPlan failed");
    return fft_size;
  }

  template <typename T>
  void _destroy(typename FFTPlanType<T>::type& plan) {
    hipfftDestroy(plan);
  }
};

#endif