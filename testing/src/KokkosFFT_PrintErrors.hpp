// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_PRINT_ERRORS_HPP
#define KOKKOSFFT_PRINT_ERRORS_HPP

#include <tuple>
#include <map>
#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Concepts.hpp"

namespace KokkosFFT {
namespace Testing {
namespace Impl {

/// \brief Sorts error information based on global indices. This function
/// processes error data from Kokkos error views and organizes them into a map
/// where the key is the global index and the value is a tuple containing a
/// vector of additional index information, the error value from the first view,
/// and the error value from the second view.
///
/// \tparam AErrorViewType The type of the Kokkos view storing error values from
/// the first data set.
/// \tparam BErrorViewType The type of the Kokkos view storing error values from
/// the second data set.
/// \tparam CountViewType The type of the Kokkos view storing index information
/// for each error.
/// \param a_error [in] A Kokkos view containing error values from the first
/// set.
/// \param b_error [in] A Kokkos view containing error values from the second
/// set.
/// \param loc_error [in] A Kokkos view containing index/location information
/// for each error.
/// \param verbose [in] How many elements to be returned (default: 3)
/// \return A std::map where the key is the global index and the value is a
/// tuple consisting of a vector of additional indices, the corresponding error
/// value from the first view, and the error value from the second view.
template <typename AErrorViewType, typename BErrorViewType,
          typename CountViewType>
auto sort_errors(const AErrorViewType &a_error, const BErrorViewType &b_error,
                 const CountViewType &loc_error,
                 const std::size_t verbose = 3) {
  // Key: global idx
  // Value: tuple (vector of error idx, a, b)
  using a_value_type     = typename AErrorViewType::non_const_value_type;
  using b_value_type     = typename BErrorViewType::non_const_value_type;
  using iType            = typename CountViewType::non_const_value_type;
  using coord_type       = std::vector<iType>;
  using error_value_type = std::tuple<coord_type, a_value_type, b_value_type>;

  auto h_a_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a_error);
  auto h_b_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b_error);
  auto h_loc_error =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), loc_error);

  using error_map_type = std::map<iType, error_value_type>;
  error_map_type error_map;
  const std::size_t nb_errors         = h_a_error.extent(0);
  const std::size_t nb_errors_verbose = std::min(nb_errors, verbose);
  const std::size_t rank              = h_loc_error.extent(1);

  for (std::size_t err = 0; err < nb_errors_verbose; ++err) {
    iType global_idx = h_loc_error(err, 0);  // global idx -> key

    coord_type loc;
    for (std::size_t d = 1; d < rank; ++d) {
      loc.push_back(h_loc_error(err, d));
    }

    error_map[global_idx] =
        error_value_type({loc, h_a_error(err), h_b_error(err)});
  }

  return error_map;
}

/// \brief Converts sorted error information into a formatted string.
/// This function iterates over an error map containing error information and
/// builds a formatted string reporting each error's location, along with the
/// actual and expected error values and their difference.
///
/// \tparam ErrorMapType The type of the error map containing error information.
/// \param error_map [in] A constant reference to a map that stores error
/// details.
/// \param labelA [in] A label for the first error value (default is "actual").
/// \param labelB [in] A label for the second error value (default is
/// "expected").
/// \return A std::string containing the formatted error report.
template <typename ErrorMapType>
auto print_errors(const ErrorMapType &error_map,
                  const std::string &labelA = "actual",
                  const std::string &labelB = "expected") {
  using TupleType = std::remove_reference_t<
      decltype(std::declval<ErrorMapType>().begin()->second)>;
  using ReferenceValueType = typename std::tuple_element<2, TupleType>::type;

  std::stringstream ss;
  ss << std::fixed << std::setprecision(sizeof(ReferenceValueType) * 2);
  ss << "Mismatched elements (by indices):\n";

  // Loop over error_map and print the error information
  for (const auto &error : error_map) {
    const auto &loc = std::get<0>(error.second);
    const auto &a   = std::get<1>(error.second);
    const auto &b   = std::get<2>(error.second);

    auto diff = Kokkos::fabs(a - b);

    ss << "  Index (";
    for (std::size_t i = 0; i < loc.size(); ++i) {
      ss << loc[i];
      if (i < loc.size() - 1) {
        ss << ", ";
      }
    }
    ss << "): " << labelA + " " << a << " vs " << labelB + " " << b
       << " (diff=" << diff << ")\n";
  }

  // Convert the stringstream data to a std::string
  std::string data = ss.str();

  // Check if the string is non-empty and its last character is a newline
  if (!data.empty() && data.back() == '\n') {
    data.pop_back();  // Remove the last character
  }

  return data;
}

}  // namespace Impl
}  // namespace Testing
}  // namespace KokkosFFT

#endif
