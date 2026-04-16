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

/// \brief Generates a formatted error message with source location information.
///
/// Produces a stringstream containing file name, line number, and function
/// name, used internally for exceptions thrown by KokkosFFT.
///
/// Example:
/// \code{.cpp}
/// auto ss = error_info("example.cpp", 10, "main");
/// \endcode
///
/// \param[in] file_name The name of the source file where the error occurs
/// \param[in] line The line number in the source file where the error occurs
/// \param[in] function_name The name of the function where the error occurs
/// \param[in] column The column number in the source file where the error
/// occurs (default: -1)
/// \return A stringstream containing the formatted error message
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

/// \brief Checks a precondition and throws an exception if it is not met.
///
/// When \p expression evaluates to true, throws a std::runtime_error with
/// detailed source location information and the provided error message.
///
/// Example:
/// \code{.cpp}
/// KOKKOSFFT_THROW_IF(true, "An error occurred");
/// // throws std::runtime_error with message:
/// // "file: example.cpp(10) `main`: An error occurred"
/// \endcode
///
/// \param[in] expression The boolean expression representing the precondition
/// to check
/// \param[in] msg The error message to include in the exception
/// \param[in] file_name The name of the source file where the error occurs
/// \param[in] line The line number in the source file where the error occurs
/// \param[in] function_name The name of the function where the error occurs
/// \param[in] column The column number in the source file where the error
/// occurs (default: -1)
/// \throws std::runtime_error if \p expression is true
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

/// \brief Checks the return status of an FFT library call and throws on
/// failure.
///
/// Validates the status returned by an FFT library call (cuFFT, hipFFT, or
/// rocFFT) and throws a std::runtime_error with detailed error information if
/// it does not indicate success.
///
/// Example:
/// \code{.cpp}
/// KOKKOSFFT_CHECK_CUFFT_CALL(cufftPlan1d(&plan, n, CUFFT_R2C, batch));
/// // On failure, throws std::runtime_error with message:
/// // "file: example.cpp(10) `main`:
/// //  cufftPlan1d(&plan, n, CUFFT_R2C, batch) failed with error code 1
/// (CUFFT_INVALID_PLAN)"
/// \endcode
///
/// \tparam FFTStatusType The type of the FFT library status code
/// \tparam ResultToStringFunc The type of the function converting a status code
/// to a string
///
/// \param[in] status The return status of the FFT library call
/// \param[in] command_name The string representation of the FFT library call
/// \param[in] status_success The status value indicating a successful call
/// \param[in] result_to_string A function that converts the status to a string
/// \param[in] file_name The name of the source file where the error occurs
/// \param[in] line The line number in the source file where the error occurs
/// \param[in] function_name The name of the function where the error occurs
/// \param[in] column The column number in the source file where the error
/// occurs (default: -1)
/// \throws std::runtime_error if \p status does not equal \p status_success
template <typename FFTStatusType, typename ResultToStringFunc>
void check_fft_call(FFTStatusType status, const char* command_name,
                    FFTStatusType status_success,
                    ResultToStringFunc result_to_string, const char* file_name,
                    int line, const char* function_name,
                    const int column = -1) {
  if (status != status_success) {
    auto ss = error_info(file_name, line, function_name, column);
    ss << "\n"
       << command_name << " failed with error code " << status << " ("
       << result_to_string(status) << ")\n";
    throw std::runtime_error(ss.str());
  }
}

}  // namespace Impl
}  // namespace KokkosFFT

#endif
