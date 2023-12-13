#ifndef KOKKOSFFT_PLANS_HPP
#define KOKKOSFFT_PLANS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_transpose.hpp"
#include "KokkosFFT_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA)
  using default_device = Kokkos::Cuda;
  #include "KokkosFFT_Cuda_plans.hpp"
#elif defined(KOKKOS_ENABLE_HIP)
  using default_device = Kokkos::HIP;
  #include "KokkosFFT_HIP_plans.hpp"
#elif defined(KOKKOS_ENABLE_OPENMP)
  using default_device = Kokkos::OpenMP;
  #include "KokkosFFT_OpenMP_plans.hpp"
#elif defined(KOKKOS_ENABLE_THREADS)
  using default_device = Kokkos::Threads;
  #include "KokkosFFT_OpenMP_plans.hpp"
#else
  using default_device = Kokkos::Serial;
  #include "KokkosFFT_OpenMP_plans.hpp"
#endif

namespace KokkosFFT {
  template <typename InViewType, typename OutViewType, std::size_t DIM = 1>
  class Plan {
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    using float_type = real_type_t<in_value_type>;
    using fft_plan_type = typename FFTPlanType< float_type >::type;
    using fft_size_type = std::size_t;
    using map_type = axis_type<InViewType::rank()>;
    using nonConstInViewType = std::remove_cv_t<InViewType>;
    using nonConstOutViewType = std::remove_cv_t<OutViewType>;

    fft_plan_type m_plan;
    fft_size_type m_fft_size;
    map_type m_map, m_map_inv;
    bool m_is_transpose_needed;

    // Only used when transpose needed
    nonConstInViewType m_in_T;
    nonConstOutViewType m_out_T;

  public:
    explicit Plan(InViewType& in, OutViewType& out) : m_fft_size(1), m_is_transpose_needed(false) {
      // Available only for R2C or C2R
      // For C2C, direction should be given by an user
      static_assert(Kokkos::is_view<InViewType>::value,
                    "KokkosFFT::Plan: InViewType is not a Kokkos::View.");
      static_assert(Kokkos::is_view<OutViewType>::value,
                    "KokkosFFT::Plan: OutViewType is not a Kokkos::View.");

      using in_value_type = typename InViewType::non_const_value_type;
      using out_value_type = typename OutViewType::non_const_value_type;
      constexpr auto type = transform_type<in_value_type, out_value_type>::type();
      FFTDirectionType direction;
      if constexpr (type == KOKKOS_FFT_R2C || type == KOKKOS_FFT_D2Z) {
        direction = KOKKOS_FFT_FORWARD;
      } else if constexpr (type == KOKKOS_FFT_C2R || type == KOKKOS_FFT_Z2D) {
        direction = KOKKOS_FFT_BACKWARD;
      } else {
        throw std::runtime_error("direction not specified for Complex to Complex transform");
      }

      m_fft_size = _create(m_plan, in, out, direction);
    }

    explicit Plan(InViewType& in, OutViewType& out, axis_type<DIM> axes) : m_fft_size(1), m_is_transpose_needed(false) {
      // Available only for R2C or C2R
      // For C2C, direction should be given by an user
      static_assert(Kokkos::is_view<InViewType>::value,
                    "KokkosFFT::Plan: InViewType is not a Kokkos::View.");
      static_assert(Kokkos::is_view<OutViewType>::value,
                    "KokkosFFT::Plan: OutViewType is not a Kokkos::View.");

      using in_value_type = typename InViewType::non_const_value_type;
      using out_value_type = typename OutViewType::non_const_value_type;
      constexpr auto type = transform_type<in_value_type, out_value_type>::type();
      FFTDirectionType direction;
      if constexpr (type == KOKKOS_FFT_R2C || type == KOKKOS_FFT_D2Z) {
        direction = KOKKOS_FFT_FORWARD;
      } else if constexpr (type == KOKKOS_FFT_C2R || type == KOKKOS_FFT_Z2D) {
        direction = KOKKOS_FFT_BACKWARD;
      } else {
        throw std::runtime_error("direction not specified for Complex to Complex transform");
      }

      m_fft_size = _create(m_plan, in, out, direction, axes);
    }

    explicit Plan(InViewType& in, OutViewType& out, FFTDirectionType direction) : m_fft_size(1), m_is_transpose_needed(false) {
      static_assert(Kokkos::is_view<InViewType>::value,
                    "KokkosFFT::Plan: InViewType is not a Kokkos::View.");
      static_assert(Kokkos::is_view<OutViewType>::value,
                    "KokkosFFT::Plan: OutViewType is not a Kokkos::View.");

      /* Apply FFT over entire axes or along inner most directions */
      m_fft_size = _create(m_plan, in, out, direction);
    }

    explicit Plan(InViewType& in, OutViewType& out, FFTDirectionType direction, int axis) : m_fft_size(1), m_is_transpose_needed(false) {
      static_assert(Kokkos::is_view<InViewType>::value,
                    "KokkosFFT::Plan: InViewType is not a Kokkos::View.");
      static_assert(Kokkos::is_view<OutViewType>::value,
                    "KokkosFFT::Plan: OutViewType is not a Kokkos::View.");

      std::tie(m_map, m_map_inv) = KokkosFFT::get_map_axes(in, axis);
      m_is_transpose_needed = KokkosFFT::is_transpose_needed(m_map);
      m_fft_size = _create(m_plan, in, out, direction, axis_type<1>{axis});
    }

    explicit Plan(InViewType& in, OutViewType& out, FFTDirectionType direction, axis_type<DIM> axes) : m_fft_size(1), m_is_transpose_needed(false) {
      static_assert(Kokkos::is_view<InViewType>::value,
                    "KokkosFFT::Plan: InViewType is not a Kokkos::View.");
      static_assert(Kokkos::is_view<OutViewType>::value,
                    "KokkosFFT::Plan: OutViewType is not a Kokkos::View.");

      std::tie(m_map, m_map_inv) = KokkosFFT::get_map_axes(in, axes);
      m_is_transpose_needed = KokkosFFT::is_transpose_needed(m_map);
      m_fft_size = _create(m_plan, in, out, direction, axes);
    }

    ~Plan() {
      _destroy<float_type>(m_plan);
    }

    fft_plan_type plan() const { return m_plan; }
    fft_size_type fft_size() const { return m_fft_size; }
    bool is_transpose_needed() const { return m_is_transpose_needed; }
    map_type map() const { return m_map; }
    map_type map_inv() const { return m_map_inv; }
    nonConstInViewType& in_T() { return m_in_T; }
    nonConstOutViewType& out_T() { return m_out_T; }
  };
};

#endif