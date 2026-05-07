// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Asserts.hpp"
#include "KokkosFFT_Common_Types.hpp"
#include "KokkosFFT_Traits.hpp"

namespace KokkosFFT {
namespace Impl {
/// \brief Generate a compile-time sequence of integral values starting from a
/// specified value
/// Example usage:
/// \code{.cpp}
/// std::array<int, 5> sequence = index_sequence<int, 5, 1>();
/// \endcode
/// This will generate a std::array containing the values {1, 2, 3, 4, 5}
///
/// \tparam IntType The integral type of the sequence values
/// \tparam N The number of values in the sequence
/// \tparam start The starting value of the sequence
///
/// \return A std::array containing the generated sequence of integral values
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

/// \brief Generate values starting from a specified value with a specified step
/// size excluding the stop value.
/// Example usage:
/// \code{.cpp}
/// std::vector<int> sequence = arange(0, 10, 2);
/// \endcode
/// This will generate a std::vector containing the values {0, 2, 4, 6, 8}
///
/// \tparam ElementType The arithmetic type of the sequence values
/// \param[in] start The starting value of the sequence
/// \param[in] stop The stopping value of the sequence (exclusive)
/// \param[in] step The step size between consecutive values (default is 1)
/// \return A std::vector containing the generated sequence values
/// \throws std::runtime_error if step is zero, if span or step is not finite,
/// if the computed number of generated elements would overflow `std::size_t`,
/// or if generating the next integral value would overflow `ElementType`
template <typename ElementType>
std::vector<ElementType> arange(const ElementType start, const ElementType stop,
                                const ElementType step = 1) {
  static_assert(std::is_arithmetic_v<ElementType>,
                "arange: ElementType must be an arithmetic type");
  KOKKOSFFT_THROW_IF(step == 0, "step must be non-zero");

  std::vector<ElementType> result;
  // Unsigned integers require special treatment: step is always positive
  if constexpr (std::is_integral_v<ElementType> &&
                std::is_unsigned_v<ElementType>) {
    if (start >= stop) {
      return result;
    }
  } else {
    // For signed integers and floating-point types
    if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
      return result;
    }
  }

  const long double span =
      static_cast<long double>(stop) - static_cast<long double>(start);
  const long double stride = static_cast<long double>(step);
  KOKKOSFFT_THROW_IF(!std::isfinite(span) || !std::isfinite(stride),
                     "span and step must be finite");

  const long double n = std::ceil(span / stride);
  KOKKOSFFT_THROW_IF(
      n > static_cast<long double>(std::numeric_limits<std::size_t>::max()),
      "sequence size overflow");

  const std::size_t count = static_cast<std::size_t>(n);
  result.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    result.push_back(start + static_cast<ElementType>(i) * step);
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

/// \brief Compute the total size by multiplying all elements in the container
/// Example usage:
/// \code{.cpp}
/// std::vector<int> dims = {2, 3, 4};
/// auto size = total_size(dims); // size will be 24
/// \endcode
/// \tparam ContainerType The container type, must support begin() and end()
/// \param[in] values The input container of integral values
/// \return The total size computed by multiplying all elements in the container
/// \throws std::overflow_error if multiplication overflows
template <typename ContainerType>
auto total_size(const ContainerType& values) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*values.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "total_size: Container value type must be an integral type");

  auto safe_multiply = [](value_type a, value_type b) -> value_type {
    if constexpr (std::is_signed_v<value_type>) {
      // Check special cases with std::numeric_limits<value_type>::min()
      if ((a == std::numeric_limits<value_type>::min() && b < 0) ||
          (b == std::numeric_limits<value_type>::min() && a < 0)) {
        throw std::overflow_error("Integer multiplication overflow");
      }
      if (a > 0) {
        if ((b > 0 && a > std::numeric_limits<value_type>::max() / b) ||
            (b < 0 && b < std::numeric_limits<value_type>::min() / a)) {
          throw std::overflow_error("Integer multiplication overflow");
        }
      } else if (a < 0) {  // a < 0
        if ((b > 0 && a < std::numeric_limits<value_type>::min() / b) ||
            (b < 0 && a > std::numeric_limits<value_type>::max() / b)) {
          throw std::overflow_error("Integer multiplication overflow");
        }
      }
    } else {
      // nvcc warns a pointless comparison of unsigned integer with zero
      if (b != 0 && a > std::numeric_limits<value_type>::max() / b) {
        throw std::overflow_error("Unsigned integer multiplication overflow");
      }
    }
    return a * b;
  };

  value_type init = 1;
  return std::accumulate(values.begin(), values.end(), init, safe_multiply);
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
