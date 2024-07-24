// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef KOKKOSFFT_COMMON_TYPES_HPP
#define KOKKOSFFT_COMMON_TYPES_HPP

namespace KokkosFFT {
//! Type to specify transform axis
template <std::size_t DIM>
using axis_type = std::array<int, DIM>;

//! Type to specify new shape
template <std::size_t DIM>
using shape_type = std::array<std::size_t, DIM>;

//! Tag to specify when and how to normalize
enum class Normalization {
  //! 1/n scaling for forward transform
  forward,
  //! 1/n scaling for backward transform
  backward,
  //! 1/sqrt(n) scaling for both direction
  ortho,
  //! No scaling
  none
};

//! Tag to specify FFT direction
enum class Direction {
  //! Forward FFT
  forward,
  //! Inverse FFT
  backward,
};

//! Maximum FFT dimension allowed in KokkosFFT
constexpr std::size_t MAX_FFT_DIM = 3;

}  // namespace KokkosFFT

#endif