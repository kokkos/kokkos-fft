// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ALLCLOSE_HPP
#define KOKKOSFFT_ALLCLOSE_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Concepts.hpp"
#include "KokkosFFT_AlmostEqual.hpp"
#include "KokkosFFT_CountErrors.hpp"
#include "KokkosFFT_PrintErrors.hpp"

namespace KokkosFFT {
namespace Testing {
namespace Impl {
/// \brief Compares two Kokkos views element-wise and checks if they are close
///        within specified relative and absolute tolerances.
///        AViewType and BViewType must have the same rank,
///        non_const_value_type, and execution_space;
///
/// \tparam AViewType The type of the first Kokkos view.
/// \tparam BViewType The type of the second Kokkos view.
///
/// \param[out] listener The testing match result listener.
/// \param[in] actual The actual Kokkos view.
/// \param[in] expected The expected (reference) Kokkos view.
/// \param[in] rtol Relative tolerance for comparing the view elements
/// (default 1.e-5).
/// \param[in] atol Absolute tolerance for comparing the view elements
/// (default 1.e-8).
/// \param[in] max_displayed_errors How many elements to be reported
/// (default: 3)
template <KokkosView AViewType, KokkosView BViewType>
  requires(std::is_same_v<typename AViewType::execution_space,
                          typename BViewType::execution_space> &&
           std::is_same_v<typename AViewType::non_const_value_type,
                          typename BViewType::non_const_value_type> &&
           (AViewType::rank() == BViewType::rank()))
inline bool allclose_impl(testing::MatchResultListener* listener,
                          const AViewType& actual, const BViewType& expected,
                          double rtol, double atol,
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

  KokkosFFT::Testing::Impl::AlmostEqualOp op(rtol, atol);

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

MATCHER_P4(allclose, expected, rtol, atol, max_displayed_errors, "") {
  return Impl::allclose_impl(result_listener, arg, expected, rtol, atol,
                             max_displayed_errors);
}

MATCHER_P3(allclose, expected, rtol, atol, "") {
  return Impl::allclose_impl(result_listener, arg, expected, rtol, atol, 3);
}

MATCHER_P2(allclose, expected, rtol, "") {
  return Impl::allclose_impl(result_listener, arg, expected, rtol, 1.0e-8, 3);
}

MATCHER_P(allclose, expected, "") {
  return Impl::allclose_impl(result_listener, arg, expected, 1.0e-5, 1.0e-8, 3);
}

}  // namespace Testing
}  // namespace KokkosFFT

#endif
