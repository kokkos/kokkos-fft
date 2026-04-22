// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_UNARYOPS_HPP
#define KOKKOSFFT_UNARYOPS_HPP

#include <Kokkos_Core.hpp>
#include "KokkosFFT_Traits.hpp"

namespace KokkosFFT {
namespace Impl {

/// \brief Functor to compute the conjugate of a complex number.
struct Conjugate {
  /// \brief Applies the conjugate operation to the input value if it is
  /// complex, otherwise returns the input value unchanged. \tparam T The type
  /// of the input value. \param[in] x The input value. \return The conjugate of
  /// the input value if it is complex, otherwise the input value unchanged.
  template <typename T>
  KOKKOS_INLINE_FUNCTION auto operator()(const T& x) const {
    return is_complex_v<T> ? Kokkos::conj(x) : x;
  }
};

/// \brief Functor to add a scalar to each element of a view.
/// \tparam ScalarType1 The type of the scalar to be added.
template <typename ScalarType1>
struct Add {
  static_assert(is_admissible_value_type_v<ScalarType1>,
                "Add: ScalarType1 must be an admissible value type.");
  ScalarType1 m_scalar;

  /// \brief Constructor for the Add functor.
  /// \param[in] scalar The scalar value to be added to each element of the view
  Add(const ScalarType1& scalar) : m_scalar(scalar) {}

  /// \brief Applies the addition operation to the input value.
  /// \tparam ScalarType2 The type of the input value.
  /// \param[in] x The input value.
  /// \return The result of adding the scalar to the input value.
  template <typename ScalarType2>
  KOKKOS_INLINE_FUNCTION auto operator()(const ScalarType2& x) const {
    static_assert(is_admissible_value_type_v<ScalarType2>,
                  "Add: ScalarType2 must be an admissible value type.");
    return x + m_scalar;
  }
};

/// \brief Functor to subtract a scalar from each element of a view.
/// \tparam ScalarType1 The type of the scalar to be subtracted.
template <typename ScalarType1>
struct Subtract {
  static_assert(is_admissible_value_type_v<ScalarType1>,
                "Subtract: ScalarType1 must be an admissible value type.");
  ScalarType1 m_scalar;
  /// \brief Constructor for the Subtract functor.
  /// \param[in] scalar The scalar value to be subtracted from each element of
  /// the view.
  Subtract(const ScalarType1& scalar) : m_scalar(scalar) {}

  /// \brief Applies the subtraction operation to the input value.
  /// \tparam ScalarType2 The type of the input value.
  /// \param[in] x The input value.
  /// \return The result of subtracting the scalar from the input value.
  template <typename ScalarType2>
  KOKKOS_INLINE_FUNCTION auto operator()(const ScalarType2& x) const {
    static_assert(is_admissible_value_type_v<ScalarType2>,
                  "Subtract: ScalarType2 must be an admissible value type.");
    return x - m_scalar;
  }
};

/// \brief Functor to multiply each element of a view by a scalar.
/// \tparam ScalarType1 The type of the scalar to be multiplied.
template <typename ScalarType1>
struct Multiply {
  static_assert(is_admissible_value_type_v<ScalarType1>,
                "Multiply: ScalarType1 must be an admissible value type.");
  ScalarType1 m_scalar;

  /// \brief Constructor for the Multiply functor.
  /// \param[in] scalar The scalar value to be multiplied with each element of
  /// the view.
  Multiply(const ScalarType1& scalar) : m_scalar(scalar) {}

  /// \brief Applies the multiplication operation to the input value.
  /// \tparam ScalarType2 The type of the input value.
  /// \param[in] x The input value.
  /// \return The result of multiplying the input value by the scalar.
  template <typename ScalarType2>
  KOKKOS_INLINE_FUNCTION auto operator()(const ScalarType2& x) const {
    static_assert(is_admissible_value_type_v<ScalarType2>,
                  "Multiply: ScalarType2 must be an admissible value type.");
    return x * m_scalar;
  }
};

/// \brief Functor to divide each element of a view by a scalar.
/// \tparam ScalarType1 The type of the scalar to be divided.
template <typename ScalarType1>
struct Divide {
  static_assert(is_admissible_value_type_v<ScalarType1>,
                "Divide: ScalarType1 must be an admissible value type.");
  ScalarType1 m_scalar;

  /// \brief Constructor for the Divide functor.
  /// \param[in] scalar The scalar value to divide each element of the view by.
  Divide(const ScalarType1& scalar) : m_scalar(scalar) {}

  /// \brief Applies the division operation to the input value.
  /// \tparam ScalarType2 The type of the input value.
  /// \param[in] x The input value.
  /// \return The result of dividing the input value by the scalar.
  template <typename ScalarType2>
  KOKKOS_INLINE_FUNCTION auto operator()(const ScalarType2& x) const {
    static_assert(is_admissible_value_type_v<ScalarType2>,
                  "Divide: ScalarType2 must be an admissible value type.");
    return x / m_scalar;
  }
};

}  // namespace Impl
}  // namespace KokkosFFT

#endif
