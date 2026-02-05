// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_DISTRIBUTED_CONTAINER_ANALYSES_HPP
#define KOKKOSFFT_DISTRIBUTED_CONTAINER_ANALYSES_HPP

#include <algorithm>
#include <Kokkos_Core.hpp>
#include <KokkosFFT.hpp>

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

/// \brief Count the number of components that are not equal to one in a
/// container
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \param[in] values The container of values
/// \return The number of components that are not equal to one
template <typename ContainerType>
auto count_non_ones(const ContainerType& values) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*values.begin())>>;

  static_assert(
      std::is_integral_v<value_type>,
      "count_non_ones: Container value type must be an integral type");
  return std::count_if(values.cbegin(), values.cend(),
                       [](value_type val) { return val != 1; });
}

/// \brief Extract the different indices between two arrays
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \param[in] b The second array
/// \return A vector of indices where the arrays differ
template <typename ContainerType>
auto extract_different_indices(const ContainerType& a, const ContainerType& b) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*a.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "extract_different_indices: Container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(a.size() != b.size(),
                     "Containers must have the same size.");

  std::vector<std::size_t> diffs;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a.at(i) != b.at(i)) {
      diffs.push_back(i);
    }
  }
  return diffs;
}

/// \brief Extract the different values between two arrays
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \param[in] b The second array
/// \return A set of values where the arrays differ
template <typename ContainerType>
auto extract_different_value_set(const ContainerType& a,
                                 const ContainerType& b) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*a.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "extract_different_indices: Container value type must be an "
                "integral type");
  KOKKOSFFT_THROW_IF(a.size() != b.size(),
                     "Containers must have the same size.");

  std::vector<value_type> diffs;
  for (std::size_t i = 0; i < a.size(); ++i) {
    diffs.push_back(a.at(i));
    diffs.push_back(b.at(i));
  }
  return std::set<value_type>(diffs.begin(), diffs.end());
}

/// \brief Extract the indices where the values are not ones
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \param[in] b The second array
/// \return A vector containing the indices where the value from a or b is not
/// one
template <typename ContainerType>
auto extract_non_one_indices(const ContainerType& a, const ContainerType& b) {
  using value_type = std::remove_cv_t<std::remove_reference_t<decltype(a[0])>>;
  static_assert(
      std::is_integral_v<value_type>,
      "extract_non_one_indices: Container value type must be an integral type");
  KOKKOSFFT_THROW_IF(a.size() != b.size(),
                     "Containers must have the same size.");

  std::vector<std::size_t> non_one_indices;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1 || b[i] != 1) {
      non_one_indices.push_back(i);
    }
  }
  return non_one_indices;
}

/// \brief Extract the non-one values
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] a The first array
/// \return A vector of non-one values
template <typename ContainerType>
auto extract_non_one_values(const ContainerType& a) {
  using value_type = std::remove_cv_t<std::remove_reference_t<decltype(a[0])>>;
  static_assert(
      std::is_integral_v<value_type>,
      "extract_non_one_values: Container value type must be an integral type");
  std::vector<value_type> non_ones;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != 1) {
      non_ones.push_back(a[i]);
    }
  }
  return non_ones;
}

/// \brief Check if the non-one elements are identical
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \param[in] non_ones The vector of non-one elements
/// \return True if the non-one elements are identical, false otherwise
/// Note: This function assumes that the size of non_ones is 2
template <typename ContainerType>
bool has_identical_non_ones(const ContainerType& non_ones) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(*non_ones.begin())>>;
  static_assert(std::is_integral_v<value_type>,
                "has_identical_non_ones: Container value type must be an "
                "integral type");

  // If there are less than 2 non-one elements, return false
  return non_ones.size() == 2 && (*(non_ones.cbegin()) == *(non_ones.cend()));
}

/// \brief Swap two elements in an array and return a new array
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \tparam iType The type of the index in the array
///
/// \param[in] arr The array to be swapped
/// \param[in] i The index of the first element to be swapped
/// \param[in] j The index of the second element to be swapped
/// \return A new array with the elements swapped at indices i and j
template <typename ContainerType, typename iType>
ContainerType swap_elements(const ContainerType& arr, iType i, iType j) {
  static_assert(std::is_integral_v<iType>,
                "swap_elements: Index type must be an integral type");
  ContainerType result = arr;
  std::swap(result.at(i), result.at(j));
  return result;
}

/// \brief Merge two topologies into one if they are convertible pencils
/// Examples
/// convertible: in_topology = {1, px, py} and out_topology = {px, 1, py}
/// in_topology can be converted into out_topology by exchaning 0th and 1st dims
/// non-convertible: in_topology = {1, px, py} and out_topology = {py, 1, px}
/// in_topology can not be converted into out_topology by a single exchange
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
///
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \return The merged topology
/// \throws std::runtime_error if the topologies do not have the same size or
/// are not convertible pencils
template <typename ContainerType>
auto merge_topology(const ContainerType& in_topology,
                    const ContainerType& out_topology) {
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  KOKKOSFFT_THROW_IF(in_size != out_size,
                     "Input and output topologies must have the same size.");

  if (in_size == 1) return in_topology;

  auto mismatched_extents = [](ContainerType in_topology,
                               ContainerType out_topology) -> std::string {
    std::string message;
    message = "Input and output topologies must differ exactly two positions: ";
    message += "in_topology (";
    message += std::to_string(in_topology.at(0));
    for (std::size_t r = 1; r < in_topology.size(); r++) {
      message += ",";
      message += std::to_string(in_topology.at(r));
    }
    message += "), ";
    message += "out_topology (";
    message += std::to_string(out_topology.at(0));
    for (std::size_t r = 1; r < out_topology.size(); r++) {
      message += ",";
      message += std::to_string(out_topology.at(r));
    }
    message += ")";
    return message;
  };

  // Check if two topologies are two convertible pencils
  auto diff_indices = extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(diff_indices.size() != 2,
                     mismatched_extents(in_topology, out_topology));

  ContainerType merged_topology = in_topology;
  for (std::size_t i = 0; i < in_topology.size(); i++) {
    merged_topology.at(i) = std::max(in_topology.at(i), out_topology.at(i));
  }
  return merged_topology;
}

/// \brief Get the non-one extent where the two topologies differ
/// In practice, compare the merged topology with one of the input
/// For example, if the two topologies are (1, p0, 1) and (p0, 1, 1),
/// the merged topology is (p0, p0, 1). The two topologies differ at the first
/// position, and the non-one extent is p0
///
/// \tparam ContainerType The type of the container (e.g., std::array,
/// std::vector)
/// \param[in] in_topology The input topology
/// \param[in] out_topology The output topology
/// \return The non-one extent where the two topologies differ. If both
/// topologies are ones, returns 1
/// \throws std::runtime_error if the topologies do not differ at exactly one
/// position
template <typename ContainerType>
auto diff_topology(const ContainerType& in_topology,
                   const ContainerType& out_topology) {
  using value_type =
      std::remove_cv_t<std::remove_reference_t<decltype(in_topology.at(0))>>;
  auto in_size  = KokkosFFT::Impl::total_size(in_topology);
  auto out_size = KokkosFFT::Impl::total_size(out_topology);

  if (in_size == 1 && out_size == 1) return value_type(1);

  auto diff_indices = extract_different_indices(in_topology, out_topology);
  KOKKOSFFT_THROW_IF(
      diff_indices.size() != 1,
      "Input and output topologies must differ at exactly one position.");
  auto diff_idx = diff_indices.at(0);

  // Returning the non-one extent
  return std::max(in_topology.at(diff_idx), out_topology.at(diff_idx));
}

}  // namespace Impl
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
