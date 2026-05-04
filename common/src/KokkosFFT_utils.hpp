// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Traits.hpp"

namespace KokkosFFT {
namespace Impl {

/// \brief Converts an axis in [-rank, rank-1] to [0, rank-1]
/// \tparam IntType The integer type used for axis
///
/// \param[in] axis The axis to be converted
/// \param[in] rank The rank of the view
/// \return The converted axis
/// \throws a runtime_error if axis is out of range
template <typename IntType>
IntType convert_negative_axis(IntType axis, std::size_t rank) {
  static_assert(std::is_integral_v<IntType>,
                "convert_negative_axis: IntType must be an integral type.");

  const IntType irank = static_cast<IntType>(rank);
  if constexpr (std::is_signed_v<IntType>) {
    KOKKOSFFT_THROW_IF(axis < -irank || axis >= irank,
                       "Axis must be in [-rank, rank-1]");

    return axis < 0 ? irank + axis : axis;
  } else {
    KOKKOSFFT_THROW_IF(axis >= irank, "Axis must be in [0, rank-1]");
    return axis;
  }
}

/// \brief Converts axes in [-rank, rank-1] to [0, rank-1]
/// \tparam IntType The integer type used for axis
/// \tparam DIM The dimensionality of the axes
///
/// \param[in] axes The axes to be converted
/// \param[in] rank The rank of the view
/// \return The converted axes
/// \throws a runtime_error if any axis is out of range
template <typename IntType, std::size_t DIM>
auto convert_negative_axes(const std::array<IntType, DIM>& axes,
                           std::size_t rank) {
  static_assert(std::is_integral_v<IntType>,
                "convert_negative_axes: IntType must be an integral type.");
  std::array<IntType, DIM> non_negative_axes{};

  const IntType irank        = static_cast<IntType>(rank);
  auto convert_negative_axis = [irank](IntType axis) -> IntType {
    if constexpr (std::is_signed_v<IntType>) {
      KOKKOSFFT_THROW_IF(axis < -irank || axis >= irank,
                         "All axes must be in [-rank, rank-1]");
      return axis < 0 ? irank + axis : axis;
    } else {
      KOKKOSFFT_THROW_IF(axis >= irank, "All axes must be in [0, rank-1]");
      return axis;
    }
  };

  std::transform(axes.begin(), axes.end(), non_negative_axes.begin(),
                 convert_negative_axis);

  return non_negative_axes;
}

template <typename IntType, std::size_t N, IntType start>
constexpr std::array<IntType, N> index_sequence() {
  static_assert(std::is_integral_v<IntType>,
                "index_sequence: IntType must be an integral type.");
  std::array<IntType, N> sequence{};
  for (std::size_t i = 0; i < N; ++i) {
    sequence[i] = start + static_cast<IntType>(i);
  }
  return sequence;
}

template <typename ElementType>
inline std::vector<ElementType> arange(const ElementType start,
                                       const ElementType stop,
                                       const ElementType step = 1) {
  const size_t length = ceil((stop - start) / step);
  std::vector<ElementType> result;
  ElementType delta = (stop - start) / length;

  // thrust::sequence
  for (std::size_t i = 0; i < length; i++) {
    ElementType value = start + delta * i;
    result.push_back(value);
  }
  return result;
}

template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_lvalue(
    std::array<T, N>& a, std::index_sequence<Is...>) {
  return {{a[Is]...}};
}
template <typename T, std::size_t N, std::size_t... Is>
constexpr Kokkos::Array<std::remove_cv_t<T>, N> to_array_rvalue(
    std::array<T, N>&& a, std::index_sequence<Is...>) {
  return {{std::move(a[Is])...}};
}

template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>& a) {
  return to_array_lvalue(a, std::make_index_sequence<N>());
}
template <typename T, std::size_t N>
constexpr Kokkos::Array<T, N> to_array(std::array<T, N>&& a) {
  return to_array_rvalue(std::move(a), std::make_index_sequence<N>());
}

template <typename T>
T safe_multiply(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    if constexpr (std::is_signed_v<T>) {
      // Check special cases with std::numeric_limits<T>::min()
      if ((a == std::numeric_limits<T>::min() && b < 0) ||
          (b == std::numeric_limits<T>::min() && a < 0)) {
        throw std::overflow_error("Integer multiplication overflow");
      }

      if (a > 0) {
        if ((b > 0 && a > std::numeric_limits<T>::max() / b) ||
            (b < 0 && b < std::numeric_limits<T>::min() / a)) {
          throw std::overflow_error("Integer multiplication overflow");
        }
      } else if (a < 0) {  // a < 0
        if ((b > 0 && a < std::numeric_limits<T>::min() / b) ||
            (b < 0 && a > std::numeric_limits<T>::max() / b)) {
          throw std::overflow_error("Integer multiplication overflow");
        }
      }
    } else {
      // nvcc warns a pointless comparison of unsigned integer with zero
      if (b != 0 && a > std::numeric_limits<T>::max() / b) {
        throw std::overflow_error("Unsigned integer multiplication overflow");
      }
    }
  }
  return a * b;
}

template <typename ContainerType>
auto total_size(const ContainerType& values) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*values.begin())>>;

  value_type init = 1;
  static_assert(std::is_integral_v<value_type>,
                "total_size: Container value type must be an integral type");
  return std::accumulate(
      values.begin(), values.end(), init,
      [](value_type a, value_type b) { return safe_multiply(a, b); });
}

/// \brief Helper to convert the base integral type of a container to another
/// integral type
/// \tparam To The target integral type
/// \tparam ContainerType The container type, must be either one of std::array
/// or std::vector
///
/// \param[in] src The source container
/// \return A new container with the same type as src but with base integral
/// type converted to To
template <typename To, typename ContainerType,
          std::enable_if_t<is_std_vector_v<ContainerType> ||
                               is_std_array_v<ContainerType>,
                           std::nullptr_t> = nullptr>
auto convert_base_int_type(const ContainerType& src) {
  using From = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_integral_v<From>,
                "convert_base_int_type: Container value type must be an "
                "integral type");
  static_assert(std::is_integral_v<To>,
                "convert_base_int_type: To must be an integral type");

  if constexpr (std::is_same_v<From, To>) {
    return src;  // no conversion needed
  } else {
    // Check convertibility
    static_assert(std::is_convertible_v<From, To>,
                  "convert_base_int_type: From must be convertible to To");

    auto safe_conversion = [](const From v) -> To {
      // Check for overflow or underflow
      constexpr auto to_min = std::numeric_limits<To>::min();
      constexpr auto to_max = std::numeric_limits<To>::max();

      if constexpr (std::is_signed_v<From> && std::is_signed_v<To>) {
        // Both are signed
        if (v < to_min || v > to_max) {
          throw std::overflow_error(
              "convert_base_int_type: Signed-to-signed overflow");
        }
      } else if constexpr (std::is_signed_v<From> && std::is_unsigned_v<To>) {
        // From is signed, To is unsigned
        if (v < 0) {
          throw std::overflow_error(
              "convert_base_int_type: negative value cannot be converted to "
              "unsigned type");
        }

        if (static_cast<std::make_unsigned_t<From>>(v) > to_max) {
          throw std::overflow_error(
              "convert_base_int_type: Signed-to-unsigned overflow");
        }
      } else if constexpr (std::is_unsigned_v<From> && std::is_signed_v<To>) {
        // From is unsigned, To is signed
        if (v > static_cast<std::make_unsigned_t<To>>(to_max)) {
          throw std::overflow_error(
              "convert_base_int_type: Unsigned-to-signed overflow");
        }
      } else {
        // Both are unsigned
        if (v > to_max) {
          throw std::overflow_error(
              "convert_base_int_type: Unsigned-to-unsigned overflow");
        }
      }

      return static_cast<To>(v);
    };

    if constexpr (is_std_vector_v<ContainerType>) {
      // Handle std::vector
      std::vector<To> dst(src.size());
      std::transform(src.begin(), src.end(), dst.begin(), safe_conversion);
      return dst;
    } else if constexpr (is_std_array_v<ContainerType>) {
      // Handle arrays: std::array
      constexpr std::size_t N = std::tuple_size_v<ContainerType>;
      std::array<To, N> dst{};
      std::transform(src.begin(), src.end(), dst.begin(), safe_conversion);
      return dst;
    }
  }
}

/// \brief Returns a reversed copy of the input container
/// \tparam Container The container type, must support begin() and end()
/// \param[in,out] c The input container
/// \return A reversed copy of the input container
template <typename Container>
auto reversed(Container&& c) {
  // Perfect-forward into a new variable (copy or move automatically)
  auto copy = std::forward<Container>(c);

  std::reverse(copy.begin(), copy.end());
  return copy;
}

/// \brief Convert a std::array to std::vector
/// \tparam ArrayType The type of the std::array
/// \param[in, out] arr The input std::array
/// \return A std::vector containing the elements of the input array
template <typename ArrayType>
auto to_vector(ArrayType&& arr) {
  using array_type = std::decay_t<ArrayType>;
  static_assert(KokkosFFT::Impl::is_std_array_v<array_type>,
                "to_vector: Input type must be a std::array");

  using value_type        = typename array_type::value_type;
  constexpr std::size_t N = std::tuple_size_v<array_type>;

  if constexpr (std::is_rvalue_reference_v<ArrayType&&>) {
    // Move elements from the rvalue array
    std::vector<value_type> vec;
    vec.reserve(N);
    std::move(arr.begin(), arr.end(), std::back_inserter(vec));
    return vec;
  } else {
    // Copy elements from the lvalue array
    std::vector<value_type> vec(arr.begin(), arr.end());
    return vec;
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
