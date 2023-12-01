#ifndef __KOKKOSFFT_NORMALIZATION_HPP__
#define __KOKKOSFFT_NORMALIZATION_HPP__

#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
  enum class FFT_Normalization {
    FORWARD,
    BACKWARD,
    ORTHO
  };

  template <typename ViewType, typename T>
  void _normalize(ViewType& inout, const T coef) {
    std::size_t size = inout.size();
    auto* data = inout.data();

    Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>{0, size},
      KOKKOS_LAMBDA(const int& i) { data[i] *= coef; }
    );
  }

  template <typename ViewType>
  auto _coefficients(const ViewType& inout, FFTDirectionType direction, FFT_Normalization normalization, std::size_t fft_size) {
    using value_type = real_type_t<typename ViewType::value_type>;
    value_type coef = 1;

    switch (normalization) {
    case FFT_Normalization::FORWARD:
      coef = direction == KOKKOS_FFT_FORWARD
             ? static_cast<value_type>(1) / static_cast<value_type>(fft_size)
             : 1;
      break;
    case FFT_Normalization::BACKWARD:
      coef = direction == KOKKOS_FFT_BACKWARD
             ? static_cast<value_type>(1) / static_cast<value_type>(fft_size)
             : 1;
      break;
    case FFT_Normalization::ORTHO:
      coef = static_cast<value_type>(1) / Kokkos::sqrt(static_cast<value_type>(fft_size));
      break;
    };
    return coef;
  }

  template <typename ViewType>
  void normalize(ViewType& inout, FFTDirectionType direction, FFT_Normalization normalization, std::size_t fft_size) {
    auto coef = _coefficients(inout, direction, normalization, fft_size);
    _normalize(inout, coef);
  }
};

#endif