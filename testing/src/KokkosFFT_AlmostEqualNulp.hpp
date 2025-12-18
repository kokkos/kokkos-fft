// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ALMOST_EQUAL_NULP_HPP
#define KOKKOSFFT_ALMOST_EQUAL_NULP_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Concepts.hpp"
#include "KokkosFFT_Ulps.hpp"
#include "KokkosFFT_CountErrors.hpp"
#include "KokkosFFT_PrintErrors.hpp"

namespace KokkosFFT {
namespace Testing {
namespace Impl {
/// \brief Compares two Kokkos views element-wise and checks if they are close
///        within specified n units in the last place (ULPs).
///        AViewType and BViewType must have the same rank,
///        non_const_value_type, and execution_space;
///
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
///
/// \param[out] listener The testing match result listener.
/// \param[in] actual The actual Kokkos view.
/// \param[in] expected The expected (reference) Kokkos view.
/// \param[in] nulp The maximum allowed difference in ULPs for the
/// numbers to be considered equal
/// \param[in] max_displayed_errors How many elements to be reported
/// (default: 3)
template <KokkosView AViewType, KokkosView BViewType>
  requires(std::is_same_v<typename AViewType::execution_space,
                          typename BViewType::execution_space> &&
           std::is_same_v<typename AViewType::non_const_value_type,
                          typename BViewType::non_const_value_type> &&
           (AViewType::rank() == BViewType::rank()))
inline bool almost_equal_nulp_impl(testing::MatchResultListener* listener,
                                   const AViewType& actual,
                                   const BViewType& expected, std::size_t nulp,
                                   std::size_t max_displayed_errors) {
  const std::size_t rank = actual.rank();
  for (std::size_t i = 0; i < rank; i++) {
    if (actual.extent(i) != expected.extent(i)) {
      *listener << actual.label() << ".extent(" << i
                << ") != " << expected.label() << ".extent(" << i << ")";
      return false;
    }
  }
  using ExecutionSpace = typename AViewType::execution_space;
  ExecutionSpace exec_space;

  KokkosFFT::Testing::Impl::UlpsComparisonOp op(nulp);

  std::size_t errors =
      KokkosFFT::Testing::Impl::count_errors(exec_space, actual, expected, op);
  if (errors == 0) return true;

  auto [a_val, e_val, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      exec_space, actual, expected, errors, op);

  exec_space.fence();
  auto error_map = KokkosFFT::Testing::Impl::sort_errors(
      a_val, e_val, loc_error, max_displayed_errors);
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  *listener << error_str;
  return false;
}
}  // namespace Impl

MATCHER_P3(almost_equal_nulp, expected, nulp, max_displayed_errors, "") {
  return Impl::almost_equal_nulp_impl(result_listener, arg, expected, nulp,
                                      max_displayed_errors);
}

MATCHER_P2(almost_equal_nulp, expected, nulp, "") {
  return Impl::almost_equal_nulp_impl(result_listener, arg, expected, nulp, 3);
}

MATCHER_P(almost_equal_nulp, expected, "") {
  return Impl::almost_equal_nulp_impl(result_listener, arg, expected, 1, 3);
}

}  // namespace Testing
}  // namespace KokkosFFT

#endif
