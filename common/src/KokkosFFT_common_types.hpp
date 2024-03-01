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

}  // namespace KokkosFFT

#endif