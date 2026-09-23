// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"
#include "KokkosFFT_Transform.hpp"

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

// A templated __device__ global (one symbol generically covering every T)
// does not compile -- see dev meeting notes 2026-09-18. kokkosfftReal/
// kokkosfftCallbackLoadR are backend-agnostic aliases (KokkosFFT_default_types.hpp)
// resolving to cufftReal/cufftCallbackLoadR or hipfftReal/hipfftCallbackLoadR
// depending on which backend is active, so this stays portable without
// naming a vendor type directly.
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftReal zero_pad_load_callback(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPointer) {
  using data_type          = kokkosfftReal;
  auto* callback_params    = static_cast<Params*>(callerInfo);
  const data_type* in_data = static_cast<const data_type*>(dataIn);

  // Zero-padding: return 0 for indices beyond original_size
  if (offset >= callback_params->original_size) {
    return static_cast<data_type>(0);
  }
  return in_data[offset];
}

KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadR d_load_callback_symbol =
    zero_pad_load_callback;

template <typename T, typename LayoutType>
void test_callback_1d() {
  const int original_size = 20;
  const int n              = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  RealView1DType x("x", n);
  ComplexView1DType x_c("x_c", n / 2 + 1);

  // Fill the whole buffer; indices >= original_size are only meaningful
  // because the callback is expected to zero them out.
  auto x_host = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) {
    x_host(i) = static_cast<T>(i + 1);
  }
  Kokkos::deep_copy(x, x_host);

  // R2C plan
  execution_space exec;
  KokkosFFT::Plan plan_r2c_axis_0(exec, x, x_c, KokkosFFT::Direction::forward,
                                  /*axis=*/0);

  Params params{static_cast<unsigned int>(original_size),
               static_cast<unsigned int>(n)};
  plan_r2c_axis_0.set_callback(d_load_callback_symbol, params);

  KokkosFFT::execute(plan_r2c_axis_0, x, x_c);
  Kokkos::fence();

  // Reference: same transform, no callback, on an input that is explicitly
  // zero-padded on the host instead of relying on the callback to do it.
  RealView1DType x_ref("x_ref", n);
  ComplexView1DType x_c_ref("x_c_ref", n / 2 + 1);
  auto x_ref_host = Kokkos::create_mirror_view(x_ref);
  for (int i = 0; i < n; ++i) {
    x_ref_host(i) = (i < original_size) ? x_host(i) : static_cast<T>(0);
  }
  Kokkos::deep_copy(x_ref, x_ref_host);

  KokkosFFT::Plan plan_ref(exec, x_ref, x_c_ref, KokkosFFT::Direction::forward,
                           /*axis=*/0);
  KokkosFFT::execute(plan_ref, x_ref, x_c_ref);
  Kokkos::fence();

  auto x_c_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_c);
  auto x_c_ref_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_c_ref);

  for (int i = 0; i < static_cast<int>(x_c_host.extent(0)); ++i) {
    EXPECT_NEAR(x_c_host(i).real(), x_c_ref_host(i).real(), 1e-3)
        << "mismatch at index " << i << " (real)";
    EXPECT_NEAR(x_c_host(i).imag(), x_c_ref_host(i).imag(), 1e-3)
        << "mismatch at index " << i << " (imag)";
  }
}
}  // namespace

TYPED_TEST_SUITE(TestCallback1D, test_types);

// Tests for plan constructiblility
TYPED_TEST(TestCallback1D, callback_1d) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_callback_1d<float_type, layout_type>();
}
