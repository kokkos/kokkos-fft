// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_NORMALIZATION_HPP
#define KOKKOSFFT_NORMALIZATION_HPP

#include <array>
#include <cmath>
#include <numeric>
#include <tuple>
#include "KokkosFFT_common_types.hpp"
#include "KokkosFFT_traits.hpp"

namespace KokkosFFT {
namespace Impl {

/// \brief Internal function to normalize the input/output view
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam ViewType: The type of input/output view
/// \tparam T: The type of normalization coefficient
///
/// \param[in] exec_space Kokkos execution space
/// \param[in,out] inout Input/output view to be normalized
/// \param[in] coef Normalization coefficient
template <typename ExecutionSpace, typename ViewType, typename T>
void normalize_impl(const ExecutionSpace& exec_space, ViewType& inout,
                    const T coef) {
  std::size_t size = inout.size();
  auto* data       = inout.data();

  Kokkos::parallel_for(
      "KokkosFFT::normalize",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, size),
      KOKKOS_LAMBDA(std::size_t i) { data[i] *= coef; });
}

/// \brief Internal function to compute the normalization coefficient 1/N
/// First try to compute N and compute 1/N, if it overflows we use the
/// following formula
/// N = N0 * N1 * ... = prod_{i} N_i
/// log N = log (N0 * N1 * ...) = sum_{i} log N_i
/// 1/N = exp(-1 * log N) = exp(-1 * sum_{i} log N_i)
/// \tparam RealType: The floating point precision type for normalization
/// \tparam ContainerType: The type of container for extents
///
/// \param[in] extents The extents of the input/output view
/// \return The normalization coefficient 1/N
template <typename RealType, typename ContainerType>
constexpr RealType one_over_N(const ContainerType& extents) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*extents.begin())>>;
  static_assert(std::is_floating_point_v<RealType>,
                "one_over_N: RealType must be a floating_point type");
  static_assert(std::is_integral_v<value_type>,
                "one_over_N: value_type must be an integral type.");
  bool overflow = false;
  value_type N  = 1;
  for (auto e : extents) {
    if (N > std::numeric_limits<value_type>::max() / e) {
      overflow = true;
      break;
    }
    N *= e;
  }
  if (!overflow) {
    return RealType{1} / static_cast<RealType>(N);
  } else {
    RealType log_N =
        std::accumulate(extents.begin(), extents.end(), RealType{0},
                        [](RealType acc, value_type e) {
                          return acc + std::log(static_cast<RealType>(e));
                        });
    return std::exp(-log_N);
  }
}

/// \brief Internal function to compute the normalization coefficient 1/sqrt(N)
/// First try to compute N and compute 1/sqrt(N), if it overflows we use the
/// following formula
/// N = N0 * N1 * ... = prod_{i} N_i
/// log N = log (N0 * N1 * ...) = sum_{i} log N_i
/// log (1/sqrt(N)) = log (N^{-1/2}) = -1/2 * log N
/// 1/sqrt(N) = exp(-1/2 * log N) = exp(-1/2 * sum_{i} log N_i)
/// \tparam RealType: The floating point precision type for normalization
/// \tparam ContainerType: The type of container for extents
///
/// \param[in] extents The extents of the input/output view
/// \return The normalization coefficient 1/sqrt(N)
template <typename RealType, typename ContainerType>
constexpr RealType one_over_sqrt_N(const ContainerType& extents) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*extents.begin())>>;
  static_assert(std::is_floating_point_v<RealType>,
                "one_over_sqrt_N: RealType must be a floating_point type");
  static_assert(std::is_integral_v<value_type>,
                "one_over_sqrt_N: value_type must be an integral type.");
  bool overflow = false;
  value_type N  = 1;
  for (auto e : extents) {
    if (N > std::numeric_limits<value_type>::max() / e) {
      overflow = true;
      break;
    }
    N *= e;
  }

  if (!overflow) {
    return RealType{1} / std::sqrt(static_cast<RealType>(N));
  } else {
    RealType log_N =
        std::accumulate(extents.begin(), extents.end(), RealType{0},
                        [](RealType acc, value_type e) {
                          return acc + std::log(static_cast<RealType>(e));
                        });

    return std::exp(-RealType{0.5} * log_N);
  }
}

/// \brief Get the normalization coefficient and whether to normalize
///
/// \tparam RealType: The floating point precision type for normalization
/// \tparam ContainerType: The type of container for extents
///
/// \param[in] direction The direction of the FFT
/// \param[in] normalization The normalization type
/// \param[in] extents The extents of the FFT
/// \return The normalization coefficient and whether to normalize
template <typename RealType, typename ContainerType>
auto get_coefficients(Direction direction, Normalization normalization,
                      const ContainerType& extents) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*extents.begin())>>;
  static_assert(std::is_floating_point_v<RealType>,
                "get_coefficients: RealType must be a floating_point type");
  static_assert(std::is_integral_v<value_type>,
                "get_coefficients: value_type must be an integral type.");
  RealType coef     = 1;
  bool to_normalize = false;

  switch (normalization) {
    case Normalization::forward:
      if (direction == Direction::forward) {
        coef         = one_over_N<RealType>(extents);
        to_normalize = true;
      }

      break;
    case Normalization::backward:
      if (direction == Direction::backward) {
        coef         = one_over_N<RealType>(extents);
        to_normalize = true;
      }

      break;
    case Normalization::ortho:
      coef         = one_over_sqrt_N<RealType>(extents);
      to_normalize = true;

      break;
    default:  // No normalization
      break;
  };
  return std::make_pair(coef, to_normalize);
}

/// \brief 1/N normalization for FFTs
///
/// \tparam RealType: The floating point precision type for normalization
/// \tparam ExecutionSpace: The type of Kokkos execution space
/// \tparam ViewType: The type of input/output view
/// \tparam ContainerType: The type of container for extents
///
/// \param[in] exec_space The execution space
/// \param[in,out] inout The input/output view
/// \param[in] direction The direction of the FFT
/// \param[in] normalization The normalization type
/// \param[in] fft_extents The extents of the FFT
template <typename RealType, typename ExecutionSpace, typename ViewType,
          typename ContainerType>
void normalize(const ExecutionSpace& exec_space, ViewType& inout,
               Direction direction, Normalization normalization,
               const ContainerType& fft_extents) {
  static_assert(KokkosFFT::Impl::is_operatable_view_v<ExecutionSpace, ViewType>,
                "normalize: View value type must be float, double, "
                "Kokkos::Complex<float>, or Kokkos::Complex<double>. "
                "Layout must be either LayoutLeft or LayoutRight. "
                "The data in ViewType must be accessible from ExecutionSpace.");
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*fft_extents.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "normalize: ContainerType must contain integral type.");
  auto [coef, to_normalize] =
      get_coefficients<RealType>(direction, normalization, fft_extents);
  if (to_normalize) normalize_impl(exec_space, inout, coef);
}

/// \brief Get the opposite direction used in hfft and ihfft
///
/// \param[in] normalization Normalization type
/// \return The opposite direction
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
