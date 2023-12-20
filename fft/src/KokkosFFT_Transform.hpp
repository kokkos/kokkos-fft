#ifndef KOKKOSFFT_TRANSFORM_HPP
#define KOKKOSFFT_TRANSFORM_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_normalization.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_Plans.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
  using default_device = Kokkos::Cuda;
  #include "KokkosFFT_Cuda_transform.hpp"
#elif defined(KOKKOS_ENABLE_HIP)
  using default_device = Kokkos::HIP;
  #include "KokkosFFT_HIP_transform.hpp"
#elif defined(KOKKOS_ENABLE_OPENMP)
  using default_device = Kokkos::OpenMP;
  #include "KokkosFFT_OpenMP_transform.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
  using default_device = Kokkos::Threads;
  #include "KokkosFFT_OpenMP_transform.hpp"
#else
  using default_device = Kokkos::Serial;
  #include "KokkosFFT_OpenMP_transform.hpp"
#endif

// General Transform Interface
namespace KokkosFFT {
namespace Impl {
  template <typename ExecutionSpace, typename PlanType, typename InViewType, typename OutViewType>
  void _fft(const ExecutionSpace& exec_space, PlanType& plan, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_fft: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::_fft: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<in_value_type>::type*>(in.data());
    auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<out_value_type>::type*>(out.data());

    KokkosFFT::Impl::_exec(plan.plan(), idata, odata, KOKKOS_FFT_FORWARD);
    KokkosFFT::Impl::normalize(exec_space, out, KOKKOS_FFT_FORWARD, norm, plan.fft_size());
  }

  template <typename ExecutionSpace, typename PlanType, typename InViewType, typename OutViewType>
  void _ifft(const ExecutionSpace& exec_space, PlanType& plan, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::_ifft: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::_ifft: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    auto* idata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<in_value_type>::type*>(in.data());
    auto* odata = reinterpret_cast<typename KokkosFFT::Impl::fft_data_type<out_value_type>::type*>(out.data());

    KokkosFFT::Impl::_exec(plan.plan(), idata, odata, KOKKOS_FFT_BACKWARD);
    KokkosFFT::Impl::normalize(exec_space, out, KOKKOS_FFT_BACKWARD, norm, plan.fft_size());
  }
} // namespace Impl
} // namespace KokkosFFT

namespace KokkosFFT {
  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void fft(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, int axis=-1) {
    static_assert(Kokkos::is_view<InViewType>::value,
                  "KokkosFFT::fft: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                  "KokkosFFT::fft: OutViewType is not a Kokkos::View.");

    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename InViewType::memory_space>::accessible,
                                 "KokkosFFT::fft: execution_space cannot access data in InViewType"
    );

    static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename OutViewType::memory_space>::accessible,
                                 "KokkosFFT::fft: execution_space cannot access data in OutViewType"
    );

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_FORWARD, axis);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

    } else {
      KokkosFFT::Impl::_fft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void ifft(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, int axis=-1) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::ifft: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::ifft: OutViewType is not a Kokkos::View.");

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_BACKWARD, axis);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());

    } else {
      KokkosFFT::Impl::_ifft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void rfft(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, int axis=-1) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::rfft: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::rfft: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(std::is_floating_point<in_value_type>::value,
                  "KokkosFFT::rfft: InViewType must be real");
    static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                  "KokkosFFT::rfft: OutViewType must be complex");

    fft(exec_space, in, out, norm, axis);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void irfft(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, int axis=-1) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::irfft: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::irfft: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                  "KokkosFFT::irfft: InViewType must be complex");
    static_assert(std::is_floating_point<out_value_type>::value,
                  "KokkosFFT::irfft: OutViewType must be real");

    ifft(exec_space, in, out, norm, axis);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void fft2(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, axis_type<2> axes={-2, -1}) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::fft2: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::fft2: OutViewType is not a Kokkos::View.");

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_FORWARD, axes);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
    } else {
      KokkosFFT::Impl::_fft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void ifft2(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, axis_type<2> axes={-2, -1}) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::ifft2: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::ifft2: OutViewType is not a Kokkos::View.");

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_BACKWARD, axes);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
    } else {
      KokkosFFT::Impl::_ifft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void rfft2(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, axis_type<2> axes={-2, -1}) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::rfft2: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::rfft2: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(std::is_floating_point<in_value_type>::value,
                  "KokkosFFT::rfft2: InViewType must be real");
    static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                  "KokkosFFT::rfft2: OutViewType must be complex");

    fft2(exec_space, in, out, norm, axes);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void irfft2(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD, axis_type<2> axes={-2, -1}) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::irfft2: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::irfft2: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                  "KokkosFFT::irfft2: InViewType must be complex");
    static_assert(std::is_floating_point<out_value_type>::value,
                  "KokkosFFT::irfft2: OutViewType must be real");

    ifft2(exec_space, in, out, norm, axes);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void fftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::fftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::fftn: OutViewType is not a Kokkos::View.");

    // Create a default sequence of axes {-rank, -(rank-1), ..., -1}
    constexpr std::size_t rank = InViewType::rank();
    constexpr int start = -static_cast<int>(rank);
    axis_type<rank> axes = KokkosFFT::Impl::index_sequence<rank>(start);

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_FORWARD, axes);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
    } else {
      KokkosFFT::Impl::_fft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType, std::size_t DIM=1>
  void fftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, axis_type<DIM> axes, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::fftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::fftn: OutViewType is not a Kokkos::View.");

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_FORWARD, axes);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_fft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
    } else {
      KokkosFFT::Impl::_fft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void ifftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::ifftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::ifftn: OutViewType is not a Kokkos::View.");

    // Create a default sequence of axes {-rank, -(rank-1), ..., -1}
    constexpr std::size_t rank = InViewType::rank();
    constexpr int start = -static_cast<int>(rank);
    axis_type<rank> axes = KokkosFFT::Impl::index_sequence<rank>(start);

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_BACKWARD, axes);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
    } else {
      KokkosFFT::Impl::_ifft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType, std::size_t DIM=1>
  void ifftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, axis_type<DIM> axes, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::ifftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::ifftn: OutViewType is not a Kokkos::View.");

    KokkosFFT::Impl::Plan plan(exec_space, in, out, KOKKOS_FFT_BACKWARD, axes);
    if(plan.is_transpose_needed()) {
      InViewType in_T;
      OutViewType out_T;

      KokkosFFT::Impl::transpose(exec_space, in, in_T, plan.map());
      KokkosFFT::Impl::transpose(exec_space, out, out_T, plan.map());

      KokkosFFT::Impl::_ifft(exec_space, plan, in_T, out_T, norm);

      KokkosFFT::Impl::transpose(exec_space, out_T, out, plan.map_inv());
    } else {
      KokkosFFT::Impl::_ifft(exec_space, plan, in, out, norm);
    }
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void rfftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::rfftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::rfftn: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(std::is_floating_point<in_value_type>::value,
                  "KokkosFFT::rfftn: InViewType must be real");
    static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                  "KokkosFFT::rfftn: OutViewType must be complex");

    fftn(exec_space, in, out, norm);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType, std::size_t DIM=1>
  void rfftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, axis_type<DIM> axes, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::rfftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::rfftn: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(std::is_floating_point<in_value_type>::value,
                  "KokkosFFT::rfftn: InViewType must be real");
    static_assert(KokkosFFT::Impl::is_complex<out_value_type>::value,
                  "KokkosFFT::rfftn: OutViewType must be complex");

    fftn(exec_space, in, out, axes, norm);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType>
  void irfftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::irfftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::irfftn: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                  "KokkosFFT::irfftn: InViewType must be complex");
    static_assert(std::is_floating_point<out_value_type>::value,
                  "KokkosFFT::irfftn: OutViewType must be real");

    ifftn(exec_space, in, out, norm);
  }

  template <typename ExecutionSpace, typename InViewType, typename OutViewType, std::size_t DIM=1>
  void irfftn(const ExecutionSpace& exec_space, const InViewType& in, OutViewType& out, axis_type<DIM> axes, KokkosFFT::Normalization norm=KokkosFFT::Normalization::BACKWARD) {
    static_assert(Kokkos::is_view<InViewType>::value,
                "KokkosFFT::irfftn: InViewType is not a Kokkos::View.");
    static_assert(Kokkos::is_view<OutViewType>::value,
                "KokkosFFT::irfftn: OutViewType is not a Kokkos::View.");

    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;

    static_assert(KokkosFFT::Impl::is_complex<in_value_type>::value,
                  "KokkosFFT::irfftn: InViewType must be complex");
    static_assert(std::is_floating_point<out_value_type>::value,
                  "KokkosFFT::irfftn: OutViewType must be real");

    ifftn(exec_space, in, out, axes, norm);
  }
};

#endif