// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_ASSERTS_HPP
#define KOKKOSFFT_ASSERTS_HPP

#include <stdexcept>
#include <sstream>
#include <string_view>

#define KOKKOSFFT_STATIC_ASSERT_VIEWS_ARE_OPERATABLE(expr, name)             \
  static_assert(                                                             \
      (expr), name                                                           \
      ": InViewType and OutViewType must have the same base floating point " \
      "type (float/double), the same layout (LayoutLeft/LayoutRight), and "  \
      "the same rank. The data in InViewType and OutViewType must be "       \
      "accessible from ExecutionSpace.")

#if defined(__cpp_lib_source_location) && __cpp_lib_source_location >= 201907L
#include <source_location>
#define KOKKOSFFT_THROW_IF(expression, msg)                           \
  KokkosFFT::Impl::check_precondition(                                \
      (expression), msg, std::source_location::current().file_name(), \
      std::source_location::current().line(),                         \
      std::source_location::current().function_name(),                \
      std::source_location::current().column())
#else
#define KOKKOSFFT_THROW_IF(expression, msg)                                  \
  KokkosFFT::Impl::check_precondition((expression), msg, __FILE__, __LINE__, \
                                      __FUNCTION__)
#endif

namespace KokkosFFT {
namespace Impl {

inline std::stringstream error_info(const char* file_name, int line,
                                    const char* function_name,
                                    const int column = -1) {
  std::stringstream ss("file: ");
  if (column == -1) {
    // For C++ 17
    ss << file_name << '(' << line << ") `" << function_name << "`: ";
  } else {
    // For C++ 20 and later
    ss << file_name << '(' << line << ':' << column << ") `" << function_name
       << "`: ";
  }
  return ss;
}

inline void check_precondition(const bool expression,
                               const std::string_view& msg,
                               const char* file_name, int line,
                               const char* function_name,
                               const int column = -1) {
  // Quick return if possible
  if (!expression) return;
  auto ss = error_info(file_name, line, function_name, column);
  ss << msg << '\n';
  throw std::runtime_error(ss.str());
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
