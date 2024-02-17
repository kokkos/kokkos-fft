#ifndef KOKKOSFFT_COMMON_TYPES_HPP
#define KOKKOSFFT_COMMON_TYPES_HPP

namespace KokkosFFT {
// Define type to specify transform axis
template <std::size_t DIM>
using axis_type = std::array<int, DIM>;

// Define type to specify new shape
template <std::size_t DIM>
using shape_type = std::array<std::size_t, DIM>;

// Tag to specify when and how to normalize
enum class Normalization { forward, backward, ortho, none };

// Tag to specify FFT direction
enum class Direction {
  forward,
  backward,
};

}  // namespace KokkosFFT

#endif