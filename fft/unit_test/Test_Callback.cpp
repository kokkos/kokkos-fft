// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
// using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
//                                     std::pair<float, Kokkos::LayoutRight>,
//                                     std::pair<double, Kokkos::LayoutLeft>,
//                                     std::pair<double, Kokkos::LayoutRight> >;
using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestCallback1D : public ::testing::Test {
  using float_type  = typename T::first_type;
  using layout_type = typename T::second_type;
};

struct Params {
  unsigned int original_size;
  unsigned int padded_size;
};

/*
template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION auto zero_pad_load_callback(void* dataIn, size_t
offset, void* callerInfo, void* sharedPointer) -> typename
KokkosFFT::Impl::fft_data_type<execution_space, T>::type { using data_type =
typename KokkosFFT::Impl::fft_data_type<execution_space, T>::type; auto*
callback_params = static_cast<Params*>(callerInfo); const data_type* in_data =
static_cast<const data_type*>(dataIn);

  // Zero-padding: return 0 for indices beyond original_size
  if (offset >= callback_params->original_size) {
    return static_cast<data_type>(0);
  }
  return in_data[offset];
}

template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION KokkosFFT::CallBackSymbolType<execution_space, T,
KokkosFFT::LoadCallback>::type d_load_callback_symbol =
zero_pad_load_callback<T>;
*/

KOKKOS_IMPL_DEVICE_FUNCTION cufftReal zero_pad_load_callback(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPointer) {
  using data_type          = cufftReal;
  auto* callback_params    = static_cast<Params*>(callerInfo);
  const data_type* in_data = static_cast<const data_type*>(dataIn);

  // Zero-padding: return 0 for indices beyond original_size
  if (offset >= callback_params->original_size) {
    return static_cast<data_type>(0);
  }
  return in_data[offset];
}

KOKKOS_IMPL_DEVICE_FUNCTION cufftCallbackLoadR d_load_callback_symbol =
    zero_pad_load_callback;

template <typename T, typename LayoutType>
void test_callback_1d() {
  const int n          = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  RealView1DType x("x", n);
  ComplexView1DType x_c("x_c", n / 2 + 1);
  ComplexView1DType x_cin("x_cin", n), x_cout("x_cout", n);

  // R2C plan
  execution_space exec;
  KokkosFFT::Plan plan_r2c_axis_0(exec, x, x_c, KokkosFFT::Direction::forward,
                                  /*axis=*/0);

  plan_r2c_axis_0.set_loadcallback(d_load_callback_symbol);
}
}  // namespace

TYPED_TEST_SUITE(TestCallback1D, test_types);

// Tests for plan constructiblility
TYPED_TEST(TestCallback1D, callback_1d) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_callback_1d<float_type, layout_type>();
}
