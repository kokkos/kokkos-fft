#ifndef KOKKOSFFT_LAYOUTS_HPP
#define KOKKOSFFT_LAYOUTS_HPP


#include <vector>
#include <tuple>
#include <iostream>
#include <numeric>
#include "KokkosFFT_default_types.hpp"
#include "KokkosFFT_utils.hpp"
#include "KokkosFFT_transpose.hpp"

namespace KokkosFFT {
  /* Input and output extents exposed to the fft library
     i.e extents are converted into Layout Right
  */
  template <typename InViewType, typename OutViewType, std::size_t DIM=1>
  auto get_extents(InViewType& in, OutViewType& out, axis_type<DIM> _axes) {
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    using array_layout_type = typename InViewType::array_layout;

    // index map after transpose over axis
    auto [map, map_inv] = get_map_axes(in, _axes);

    constexpr std::size_t rank = InViewType::rank;
    int inner_most_axis = std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft> ? 0 : rank - 1;

    std::vector<int> in_extents, out_extents, fft_extents;
    for(std::size_t i=0; i<rank; i++) {
      auto _idx = map.at(i);
      in_extents.push_back(in.extent(_idx));
      out_extents.push_back(out.extent(_idx));

      // The extent for transform is always equal to the extent
      // of the extent of real type (R2C or C2R)
      // For C2C, the in and out extents are the same.
      // In the end, we can just use the largest extent among in and out extents.
      auto fft_extent = std::max(in.extent(_idx), out.extent(_idx));
      fft_extents.push_back(fft_extent);
    }

    if(std::is_floating_point<in_value_type>::value) {
      // Then R2C
      if(is_complex<out_value_type>::value) {
        assert( out_extents.at(inner_most_axis) == in.extent(inner_most_axis)/2+1 );
      } else {
        throw std::runtime_error("If the input type is real, the output type should be complex");
      }
    }

    if(std::is_floating_point<out_value_type>::value) {
      // Then C2R
      if(is_complex<in_value_type>::value) {
        assert( in_extents.at(inner_most_axis) == out.extent(inner_most_axis)/2+1 );
      } else {
        throw std::runtime_error("If the output type is real, the input type should be complex");
      }
    }

    if(std::is_same<array_layout_type, Kokkos::LayoutLeft>::value) {
      std::reverse(in_extents.begin(), in_extents.end());
      std::reverse(out_extents.begin(), out_extents.end());
      std::reverse(fft_extents.begin(), fft_extents.end());
    }

    return std::tuple< std::vector<int>, std::vector<int>, std::vector<int> >({in_extents, out_extents, fft_extents});
  }

  /* Input and output extents exposed to the fft library
     i.e extents are converted into Layout Right
  */
  template <typename InViewType, typename OutViewType, std::size_t DIM=1>
  auto get_extents_batched(InViewType& in, OutViewType& out, axis_type<DIM> _axes) {
    using in_value_type = typename InViewType::non_const_value_type;
    using out_value_type = typename OutViewType::non_const_value_type;
    using array_layout_type = typename InViewType::array_layout;

    // index map after transpose over axis
    auto [map, map_inv] = get_map_axes(in, _axes);

    static_assert(InViewType::rank() >= DIM,
                  "KokkosFFT::get_map_axes: Rank of View must be larger thane or equal to the Rank of FFT axes.");
    static_assert(DIM > 0,
                  "KokkosFFT::get_map_axes: Rank of FFT axes must be larger than or equal to 1.");

    constexpr std::size_t rank = InViewType::rank;
    int inner_most_axis = rank - 1;

    std::vector<int> _in_extents, _out_extents, _fft_extents;

    // Get extents for the inner most axes in LayoutRight
    // If we allow the FFT on the layoutLeft, this part should be modified
    for(std::size_t i=0; i<rank; i++) {
      auto _idx = map.at(i);
      _in_extents.push_back(in.extent(_idx));
      _out_extents.push_back(out.extent(_idx));

      // The extent for transform is always equal to the extent
      // of the extent of real type (R2C or C2R)
      // For C2C, the in and out extents are the same.
      // In the end, we can just use the largest extent among in and out extents.
      auto fft_extent = std::max(in.extent(_idx), out.extent(_idx));
      _fft_extents.push_back(fft_extent);
    }

    if(std::is_same<array_layout_type, Kokkos::LayoutLeft>::value) {
      std::reverse(_in_extents.begin(), _in_extents.end());
      std::reverse(_out_extents.begin(), _out_extents.end());
      std::reverse(_fft_extents.begin(), _fft_extents.end());
    }

    // Define subvectors starting from last - DIM
    // Dimensions relevant to FFTs
    std::vector<int> in_extents(_in_extents.end()-DIM, _in_extents.end());
    std::vector<int> out_extents(_out_extents.end()-DIM, _out_extents.end());
    std::vector<int> fft_extents(_fft_extents.end()-DIM, _fft_extents.end());

    int total_fft_size = std::accumulate(_fft_extents.begin(), _fft_extents.end(), 1, std::multiplies<>());
    int fft_size = std::accumulate(fft_extents.begin(), fft_extents.end(), 1, std::multiplies<>());
    int howmany = total_fft_size / fft_size;

    if(std::is_floating_point<in_value_type>::value) {
      // Then R2C
      if(is_complex<out_value_type>::value) {
        assert( out_extents.at(inner_most_axis) == in.extent(inner_most_axis)/2+1 );
      } else {
        throw std::runtime_error("If the input type is real, the output type should be complex");
      }
    }

    if(std::is_floating_point<out_value_type>::value) {
      // Then C2R
      if(is_complex<in_value_type>::value) {
        assert( in_extents.at(inner_most_axis) == out.extent(inner_most_axis)/2+1 );
      } else {
        throw std::runtime_error("If the output type is real, the input type should be complex");
      }
    }

    return std::tuple< std::vector<int>, std::vector<int>, std::vector<int>, int >({in_extents, out_extents, fft_extents, howmany});
  }

  /* Input and output extents exposed to the fft library
     i.e extents are converted into Layout Right
  */
  template <typename InViewType, typename OutViewType>
  auto get_extents(InViewType& in, OutViewType& out) {
    using array_layout_type = typename InViewType::array_layout;

    constexpr std::size_t rank = InViewType::rank;
    int inner_most_axis = std::is_same_v<array_layout_type, typename Kokkos::LayoutLeft> ? 0 : rank - 1;
    return get_extents(in, out, axis_type<1>{inner_most_axis});
  }

  /* Input and output extents exposed to the fft library
     i.e extents are converted into Layout Right
  */
  template <typename InViewType, typename OutViewType>
  auto get_extents(InViewType& in, OutViewType& out, int _axis) {
     return get_extents(in, out, axis_type<1>{_axis});
  }

  /* Input and output extents exposed to the fft library
     i.e extents are converted into Layout Right
  */
  template <typename InViewType, typename OutViewType>
  auto get_extents_batched(InViewType& in, OutViewType& out, int _axis) {
     return get_extents_batched(in, out, axis_type<1>{_axis});
  }
};

#endif