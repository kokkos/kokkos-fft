// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ALLCLOSE_HPP
#define KOKKOSFFT_ALLCLOSE_HPP

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Concepts.hpp"
#include "KokkosFFT_CountErrors.hpp"
#include "KokkosFFT_PrintErrors.hpp"

namespace KokkosFFT {
namespace Testing {

MATCHER_P5(allclose, exec_space, expected, rtol, atol, verbose, "") {
  const std::size_t rank = arg.rank();
  for (std::size_t i = 0; i < rank; i++) {
    if (arg.extent(i) != expected.extent(i)) {
      *result_listener << arg.label() + ".extent(" + std::to_string(i) + ") != "
                       << arg.label() + ".extent(" + std::to_string(i) + ")";
      return false;
    }
  }

  std::size_t errors = KokkosFFT::Testing::Impl::count_errors(
      exec_space, arg, expected, rtol, atol);
  if (errors == 0) return true;

  auto [a_val, e_val, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      exec_space, arg, expected, errors, rtol, atol);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_val, e_val, loc_error, verbose);
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  *result_listener << error_str;
  return false;
}

MATCHER_P4(allclose, exec_space, expected, rtol, atol, "") {
  const std::size_t rank = arg.rank();
  for (std::size_t i = 0; i < rank; i++) {
    if (arg.extent(i) != expected.extent(i)) {
      *result_listener << arg.label() + ".extent(" + std::to_string(i) + ") != "
                       << arg.label() + ".extent(" + std::to_string(i) + ")";
      return false;
    }
  }

  std::size_t errors = KokkosFFT::Testing::Impl::count_errors(
      exec_space, arg, expected, rtol, atol);
  if (errors == 0) return true;

  auto [a_val, e_val, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      exec_space, arg, expected, errors, rtol, atol);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_val, e_val, loc_error);
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  *result_listener << error_str;
  return false;
}

MATCHER_P3(allclose, exec_space, expected, rtol, "") {
  double atol            = 1.0e-8;
  const std::size_t rank = arg.rank();
  for (std::size_t i = 0; i < rank; i++) {
    if (arg.extent(i) != expected.extent(i)) {
      *result_listener << arg.label() + ".extent(" + std::to_string(i) + ") != "
                       << arg.label() + ".extent(" + std::to_string(i) + ")";
      return false;
    }
  }

  std::size_t errors = KokkosFFT::Testing::Impl::count_errors(
      exec_space, arg, expected, rtol, atol);
  if (errors == 0) return true;

  auto [a_val, e_val, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      exec_space, arg, expected, errors, rtol, atol);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_val, e_val, loc_error);
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  *result_listener << error_str;
  return false;
}

MATCHER_P2(allclose, exec_space, expected, "") {
  const double rtol = 1.0e-5, atol = 1.0e-8;
  const std::size_t rank = arg.rank();
  for (std::size_t i = 0; i < rank; i++) {
    if (arg.extent(i) != expected.extent(i)) {
      *result_listener << arg.label() + ".extent(" + std::to_string(i) + ") != "
                       << arg.label() + ".extent(" + std::to_string(i) + ")";
      return false;
    }
  }

  std::size_t errors = KokkosFFT::Testing::Impl::count_errors(
      exec_space, arg, expected, rtol, atol);
  if (errors == 0) return true;

  auto [a_val, e_val, loc_error] = KokkosFFT::Testing::Impl::find_errors(
      exec_space, arg, expected, errors, rtol, atol);

  auto error_map =
      KokkosFFT::Testing::Impl::sort_errors(a_val, e_val, loc_error);
  std::string error_str = KokkosFFT::Testing::Impl::print_errors(error_map);

  *result_listener << error_str;
  return false;
}

}  // namespace Testing
}  // namespace KokkosFFT

#endif
