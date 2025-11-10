// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UTILS_HPP
#define KOKKOSFFT_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <limits>
#include "KokkosFFT_asserts.hpp"
#include "KokkosFFT_traits.hpp"
#include "KokkosFFT_common_types.hpp"

namespace KokkosFFT {
namespace Impl {

template <typename ScalarType1, typename ScalarType2>
bool are_aliasing(const ScalarType1* ptr1, const ScalarType2* ptr2) {
  return (static_cast<const void*>(ptr1) == static_cast<const void*>(ptr2));
}

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
  std::array<IntType, DIM> non_negative_axes = {};
  try {
    for (std::size_t i = 0; i < axes.size(); i++) {
      auto axis = axes.at(i);
      auto non_negative_axis =
          KokkosFFT::Impl::convert_negative_axis(axis, rank);
      non_negative_axes.at(i) = non_negative_axis;
    }
  } catch (std::runtime_error& e) {
    if constexpr (std::is_signed_v<IntType>) {
      KOKKOSFFT_THROW_IF(true, "All axes must be in [-rank, rank-1]");
    } else {
      KOKKOSFFT_THROW_IF(true, "All axes must be in [0, rank-1]");
    }
  }

  return non_negative_axes;
}

template <typename ContainerType, typename ValueType>
bool is_found(const ContainerType& values, const ValueType value) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_same_v<value_type, ValueType>,
                "is_found: Container value type must match ValueType");
  return std::find(values.begin(), values.end(), value) != values.end();
}

template <typename ContainerType>
bool has_duplicate_values(const ContainerType& values) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  std::set<value_type> set_values(values.begin(), values.end());
  return set_values.size() < values.size();
}

template <typename ContainerType, typename IntType>
bool is_out_of_range_value_included(const ContainerType& values, IntType max) {
  static_assert(
      std::is_integral_v<IntType>,
      "is_out_of_range_value_included: IntType must be an integral type");
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_same_v<value_type, IntType>,
                "is_out_of_range_value_included: Container value type must "
                "match IntType");
  if constexpr (std::is_signed_v<value_type>) {
    KOKKOSFFT_THROW_IF(
        std::any_of(values.begin(), values.end(),
                    [](value_type value) { return value < 0; }),
        "is_out_of_range_value_included: values must be non-negative");
  }
  return std::any_of(values.begin(), values.end(),
                     [max](value_type value) { return value >= max; });
}

template <
    typename ViewType, template <typename, std::size_t> class ArrayType,
    typename IntType, std::size_t DIM = 1,
    std::enable_if_t<Kokkos::is_view_v<ViewType> && std::is_integral_v<IntType>,
                     std::nullptr_t> = nullptr>
bool are_valid_axes(const ViewType& /*view*/,
                    const ArrayType<IntType, DIM>& axes) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "are_valid_axes: ViewType must be a Kokkos::View");
  static_assert(std::is_integral_v<IntType>,
                "are_valid_axes: IntType must be an integral type");
  static_assert(
      DIM >= 1 && DIM <= ViewType::rank(),
      "are_valid_axes: the Rank of FFT axes must be between 1 and View rank");

  // Convert the input axes to be in the range of [0, rank-1]
  // int type is chosen for consistency with the rest of the code
  // the axes are defined with int type
  std::array<IntType, DIM> non_negative_axes = {};

  // In case axis is out of range, 'convert_negative_axes' will throw an
  // runtime_error and we will return false. Without runtime_error, it is
  // ensured that the 'non_negative_axes' are in the range of [0, rank-1]
  try {
    non_negative_axes = convert_negative_axes(axes, ViewType::rank());
  } catch (std::runtime_error& e) {
    return false;
  }

  bool is_valid = !KokkosFFT::Impl::has_duplicate_values(non_negative_axes);
  return is_valid;
}

/// \brief Check if transpose is needed or not
/// If a map is contiguous and in ascending order (e.g. {0, 1, 2}),
/// we do not need transpose
/// \tparam IndexType The integer type used for map
/// \tparam DIM The dimensionality of the axes
///
/// \param[in] map The map used for permutation
template <typename IndexType, std::size_t DIM>
bool is_transpose_needed(const std::array<IndexType, DIM>& map) {
  static_assert(std::is_integral_v<IndexType>,
                "is_transpose_needed: IndexType must be an integral type.");
  std::array<IndexType, DIM> contiguous_map;
  std::iota(contiguous_map.begin(), contiguous_map.end(), 0);
  return map != contiguous_map;
}

template <typename ContainerType, typename ValueType>
std::size_t get_index(const ContainerType& values, const ValueType value) {
  using value_type = KokkosFFT::Impl::base_container_value_type<ContainerType>;
  static_assert(std::is_same_v<value_type, ValueType>,
                "get_index: Container value type must match ValueType");
  auto it = std::find(values.begin(), values.end(), value);
  KOKKOSFFT_THROW_IF(it == values.end(), "value is not included in values");
  return it - values.begin();
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

template <typename ExecutionSpace, typename InViewType, typename OutViewType>
void conjugate(const ExecutionSpace& exec_space, const InViewType& in,
               OutViewType& out) {
  static_assert(
      KokkosFFT::Impl::are_operatable_views_v<ExecutionSpace, InViewType,
                                              OutViewType>,
      "conjugate: InViewType and OutViewType must have the same base floating "
      "point "
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and the "
      "same rank. ExecutionSpace must be accessible to the data in InViewType "
      "and OutViewType.");

  using out_value_type = typename OutViewType::non_const_value_type;
  static_assert(KokkosFFT::Impl::is_complex_v<out_value_type>,
                "conjugate: OutViewType must be complex");
  std::size_t size = in.size();
  out              = OutViewType("out", in.layout());

  auto* in_data  = in.data();
  auto* out_data = out.data();

  Kokkos::parallel_for(
      "KokkosFFT::conjugate",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(
          exec_space, 0, size),
      KOKKOS_LAMBDA(std::size_t i) { out_data[i] = Kokkos::conj(in_data[i]); });
}

template <typename ViewType>
auto extract_extents(const ViewType& view) {
  static_assert(Kokkos::is_view_v<ViewType>,
                "extract_extents: ViewType is not a Kokkos::View.");
  constexpr std::size_t rank = ViewType::rank();
  std::array<std::size_t, rank> extents;
  for (std::size_t i = 0; i < rank; i++) {
    extents.at(i) = view.extent(i);
  }
  return extents;
}

template <typename Layout, typename IndexType, std::size_t N>
Layout create_layout(const std::array<IndexType, N>& extents) {
  static_assert(std::is_integral_v<IndexType>,
                "create_layout: IndexType must be an integral type");
  static_assert(std::is_same_v<Layout, Kokkos::LayoutLeft> ||
                    std::is_same_v<Layout, Kokkos::LayoutRight>,
                "create_layout: Layout must be either Kokkos::LayoutLeft or "
                "Kokkos::LayoutRight.");
  Layout layout;
  std::copy_n(extents.begin(), N, layout.dimension);
  return layout;
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

// \brief Helper to compute strides from extents.
// The extents are computed from a LayoutRight View.
// The computed strides can be considered as view strides
// in the reversed order.
//
// Examples:
// v0 (n0) -> (v0.stride(0)) or (1)
// v1 (n0, n1) -> (v1.stride(1), v1.stride(0)) or (1, n1)
// v2 (n0, n1, n2) -> (v2.stride(2), v2.stride(1), v2.stride(0))
//                 or (1, n2, n2 * n1)
/// \tparam ContainerType The container type, must be either one of std::array
/// or std::vector
/// \param[in] extents The extents of the data
/// \return strides computed from the input data
template <typename ContainerType,
          std::enable_if_t<is_std_vector_v<ContainerType> ||
                               is_std_array_v<ContainerType>,
                           std::nullptr_t> = nullptr>
auto compute_strides(const ContainerType& extents) {
  using index_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*extents.begin())>>;
  static_assert(std::is_integral_v<index_type>,
                "compute_strides: index_type must be an integral type.");
  KOKKOSFFT_THROW_IF(
      total_size(extents) <= 0,
      "compute_strides: total size of the extents must not be 0");
  ContainerType strides = extents, reversed_extents = extents;
  std::reverse(reversed_extents.begin(), reversed_extents.end());

  strides.at(0) = 1;
  for (std::size_t i = 1; i < reversed_extents.size(); i++) {
    strides.at(i) = reversed_extents.at(i - 1) * strides.at(i - 1);
  }
  return strides;
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

// Copied from Kokkos_Layout.hpp
// Since this is not publicly exposed, we re-implement it here
// to avoid dependency on Kokkos implementation details
template <typename... Layout>
struct layout_iterate_type_selector {
  static_assert(true,
                "layout_iterate_type_selector: Layout must be one of "
                "LayoutLeft, LayoutRight, LayoutStride");
};

template <>
struct layout_iterate_type_selector<Kokkos::LayoutRight> {
  static const Kokkos::Iterate outer_iteration_pattern = Kokkos::Iterate::Right;
  static const Kokkos::Iterate inner_iteration_pattern = Kokkos::Iterate::Right;
};

template <>
struct layout_iterate_type_selector<Kokkos::LayoutLeft> {
  static const Kokkos::Iterate outer_iteration_pattern = Kokkos::Iterate::Left;
  static const Kokkos::Iterate inner_iteration_pattern = Kokkos::Iterate::Left;
};

template <>
struct layout_iterate_type_selector<Kokkos::LayoutStride> {
  static const Kokkos::Iterate outer_iteration_pattern =
      Kokkos::Iterate::Default;
  static const Kokkos::Iterate inner_iteration_pattern =
      Kokkos::Iterate::Default;
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
