#ifndef KOKKOSFFT_NORMALIZATION_HPP
#define KOKKOSFFT_NORMALIZATION_HPP

#include <tuple>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
  enum class Normalization {
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
  auto _coefficients(const ViewType& inout, FFTDirectionType direction, Normalization normalization, std::size_t fft_size) {
    using value_type = real_type_t<typename ViewType::non_const_value_type>;
    value_type coef = 1;
    bool to_normalize = false;

    switch (normalization) {
    case Normalization::FORWARD:
      if(direction == KOKKOS_FFT_FORWARD) {
        coef = static_cast<value_type>(1) / static_cast<value_type>(fft_size);
        to_normalize = true;
      }

      break;
    case Normalization::BACKWARD:
      if(direction == KOKKOS_FFT_BACKWARD) {
        coef = static_cast<value_type>(1) / static_cast<value_type>(fft_size);
        to_normalize = true;
      }

      break;
    case Normalization::ORTHO:
      coef = static_cast<value_type>(1) / Kokkos::sqrt(static_cast<value_type>(fft_size));
      to_normalize = true;

      break;
    };
    return std::tuple<value_type, bool> ({coef, to_normalize});
  }

  template <typename ViewType>
  void normalize(ViewType& inout, FFTDirectionType direction, Normalization normalization, std::size_t fft_size) {
    auto [coef, to_normalize] = _coefficients(inout, direction, normalization, fft_size);
    if(to_normalize) _normalize(inout, coef);
  }
};

#endif