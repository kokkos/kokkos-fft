#ifndef KOKKOSFFT_NORMALIZATION_HPP
#define KOKKOSFFT_NORMALIZATION_HPP

#include <tuple>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_utils.hpp"

namespace KokkosFFT {
namespace Impl {
template <typename ExecutionSpace, typename ViewType, typename T>
void _normalize(const ExecutionSpace& exec_space, ViewType& inout,
                const T coef) {
  std::size_t size = inout.size();
  auto* data       = inout.data();

  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, size),
      KOKKOS_LAMBDA(const int& i) { data[i] *= coef; });
}

template <typename ViewType>
auto _coefficients(const ViewType& inout, Direction direction,
                   Normalization normalization, std::size_t fft_size) {
  using value_type =
      KokkosFFT::Impl::real_type_t<typename ViewType::non_const_value_type>;
  value_type coef                    = 1;
  [[maybe_unused]] bool to_normalize = false;

  switch (normalization) {
    case Normalization::forward:
      if (direction == Direction::Forward) {
        coef = static_cast<value_type>(1) / static_cast<value_type>(fft_size);
        to_normalize = true;
      }

      break;
    case Normalization::backward:
      if (direction == Direction::Backward) {
        coef = static_cast<value_type>(1) / static_cast<value_type>(fft_size);
        to_normalize = true;
      }

      break;
    case Normalization::ortho:
      coef = static_cast<value_type>(1) /
             Kokkos::sqrt(static_cast<value_type>(fft_size));
      to_normalize = true;

      break;
    default: // No normalization
      break;
  };
  return std::tuple<value_type, bool>({coef, to_normalize});
}

template <typename ExecutionSpace, typename ViewType>
void normalize(const ExecutionSpace& exec_space, ViewType& inout,
               Direction direction, Normalization normalization,
               std::size_t fft_size) {
  auto [coef, to_normalize] =
      _coefficients(inout, direction, normalization, fft_size);
  if (to_normalize) _normalize(exec_space, inout, coef);
}

inline auto swap_direction(Normalization normalization) {
  Normalization new_direction = Normalization::none;
  switch (normalization) {
    case Normalization::forward: new_direction = Normalization::backward; break;
    case Normalization::backward: new_direction = Normalization::forward; break;
    case Normalization::ortho: new_direction = Normalization::ortho; break;
    default: break;
  };
  return new_direction;
}
}  // namespace Impl
}  // namespace KokkosFFT

#endif