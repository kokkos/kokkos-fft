// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <gtest/gtest.h>
#include "KokkosFFT_Plans.hpp"
#include "KokkosFFT_Transform.hpp"
#include "KokkosFFT_Testing_Allclose.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

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

// Templated on T (float or double) so the zero-padding logic is written
// once rather than duplicated per precision. kokkosfftCallbackLoadR/LoadD
// are backend-agnostic aliases (KokkosFFT_default_types.hpp) resolving to
// cufftCallbackLoadR/D or hipfftCallbackLoadR/D depending on which backend
// is active, so this stays portable without naming a vendor type directly.
// A __device__ global itself can't be templated, so one concrete global per
// precision below just instantiates this shared function template.
template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION T zero_pad_load_callback(
    void* dataIn, size_t offset, void* callerInfo, void* sharedPointer) {
  auto* callback_params = static_cast<Params*>(callerInfo);
  const T* in_data       = static_cast<const T*>(dataIn);

  // Zero-padding: return 0 for indices beyond original_size
  if (offset >= callback_params->original_size) {
    return static_cast<T>(0);
  }
  return in_data[offset];
}

KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadR d_load_callback_symbol_fp32 =
    zero_pad_load_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadD d_load_callback_symbol_fp64 =
    zero_pad_load_callback<KokkosFFT::fft_data_type<double>>;

// Store-callback counterpart: only write within original_size, leaving the
// padding region alone. Normalization is handled by KokkosFFT's own
// Normalization argument to execute(), not by this callback.
//
// \warning cuFFT does not guarantee every output offset is routed through
// the store callback for every transform configuration (observed with C2R:
// cuFFT writes to some odd output offsets directly regardless of what the
// callback does). Only offset < original_size is a meaningful invariant to
// check; offsets the callback intentionally skips are not.
template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION void store_callback(void* dataOut, size_t offset,
                                                T element, void* callerInfo,
                                                void* sharedPointer) {
  auto* callback_params = static_cast<Params*>(callerInfo);
  T* out_data            = static_cast<T*>(dataOut);

  if (offset < callback_params->original_size) {
    out_data[offset] = element;
  }
}

KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreR d_store_callback_symbol_fp32 =
    store_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreD d_store_callback_symbol_fp64 =
    store_callback<KokkosFFT::fft_data_type<double>>;

template <typename T, typename LayoutType>
void test_load_callback_1d() {
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
  if constexpr (std::is_same_v<T, float>) {
    plan_r2c_axis_0.set_callback(d_load_callback_symbol_fp32, params);
  } else {
    plan_r2c_axis_0.set_callback(d_load_callback_symbol_fp64, params);
  }

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

  EXPECT_THAT(x_c, KokkosFFT::Testing::allclose(x_c_ref, 1.e-5, 1.e-12));
}

template <typename T, typename LayoutType>
void test_store_callback_1d() {
  const int original_size = 20;
  const int n              = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  ComplexView1DType in("in", n / 2 + 1);
  RealView1DType out("out", n);

  auto in_host = Kokkos::create_mirror_view(in);
  for (int i = 0; i < static_cast<int>(in_host.extent(0)); ++i) {
    in_host(i) = Kokkos::complex<T>(static_cast<T>(i + 1), static_cast<T>(0));
  }
  Kokkos::deep_copy(in, in_host);

  // C2R plan
  execution_space exec;
  KokkosFFT::Plan plan_c2r_axis_0(exec, in, out,
                                  KokkosFFT::Direction::backward,
                                  /*axis=*/0);

  Params params{static_cast<unsigned int>(original_size),
               static_cast<unsigned int>(n)};
  if constexpr (std::is_same_v<T, float>) {
    plan_c2r_axis_0.set_callback(d_store_callback_symbol_fp32, params);
  } else {
    plan_c2r_axis_0.set_callback(d_store_callback_symbol_fp64, params);
  }

  KokkosFFT::execute(plan_c2r_axis_0, in, out);
  Kokkos::fence();

  // Reference: same transform, no callback, at the same size. KokkosFFT's
  // own normalization runs after exec_plan either way, so both plans get
  // normalized identically regardless of the callback.
  RealView1DType out_ref("out_ref", n);
  KokkosFFT::Plan plan_ref(exec, in, out_ref, KokkosFFT::Direction::backward,
                           /*axis=*/0);
  KokkosFFT::execute(plan_ref, in, out_ref);
  Kokkos::fence();

  // Only offset < original_size is a meaningful invariant (see \warning on
  // store_callback above).
  auto out_sub     = Kokkos::subview(out, std::make_pair(0, original_size));
  auto out_ref_sub = Kokkos::subview(out_ref, std::make_pair(0, original_size));
  EXPECT_THAT(out_sub, KokkosFFT::Testing::allclose(out_ref_sub, 1.e-5, 1.e-12));
}

// End-to-end: forward R2C with the zero-padding load callback, then
// backward C2R with the cropping store callback, matching the actual
// motivating use case (zero-padding for non-periodic directions without
// explicit memory initialization -- see the parent repo's README) rather
// than exercising each callback in isolation.
template <typename T, typename LayoutType>
void test_load_store_roundtrip_1d() {
  const int original_size = 20;
  const int n              = 30;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  // Only the first original_size elements are meaningful; the rest of this
  // size-n buffer is left uninitialized -- the load callback zero-pads it
  // on the fly instead of requiring an explicit memset.
  RealView1DType x("x", n);
  ComplexView1DType x_c("x_c", n / 2 + 1);

  auto x_host = Kokkos::create_mirror_view(x);
  for (int i = 0; i < original_size; ++i) {
    x_host(i) = static_cast<T>(i + 1);
  }
  Kokkos::deep_copy(x, x_host);

  execution_space exec;
  Params params{static_cast<unsigned int>(original_size),
               static_cast<unsigned int>(n)};

  // Forward R2C with the zero-padding load callback.
  KokkosFFT::Plan plan_fwd(exec, x, x_c, KokkosFFT::Direction::forward,
                           /*axis=*/0);
  if constexpr (std::is_same_v<T, float>) {
    plan_fwd.set_callback(d_load_callback_symbol_fp32, params);
  } else {
    plan_fwd.set_callback(d_load_callback_symbol_fp64, params);
  }
  KokkosFFT::execute(plan_fwd, x, x_c);
  Kokkos::fence();

  // Backward C2R with the cropping store callback.
  RealView1DType x_out("x_out", n);
  KokkosFFT::Plan plan_bwd(exec, x_c, x_out, KokkosFFT::Direction::backward,
                           /*axis=*/0);
  if constexpr (std::is_same_v<T, float>) {
    plan_bwd.set_callback(d_store_callback_symbol_fp32, params);
  } else {
    plan_bwd.set_callback(d_store_callback_symbol_fp64, params);
  }
  KokkosFFT::execute(plan_bwd, x_c, x_out);
  Kokkos::fence();

  // Round trip should reconstruct the original signal over the region the
  // user actually cared about.
  auto x_out_sub = Kokkos::subview(x_out, std::make_pair(0, original_size));
  auto x_sub     = Kokkos::subview(x, std::make_pair(0, original_size));
  EXPECT_THAT(x_out_sub, KokkosFFT::Testing::allclose(x_sub, 1.e-5, 1.e-12));
}
}  // namespace

TYPED_TEST_SUITE(TestCallback1D, test_types);

TYPED_TEST(TestCallback1D, load_callback_1d) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_load_callback_1d<float_type, layout_type>();
}

TYPED_TEST(TestCallback1D, store_callback_1d) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_store_callback_1d<float_type, layout_type>();
}

TYPED_TEST(TestCallback1D, load_store_roundtrip_1d) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_load_store_roundtrip_1d<float_type, layout_type>();
}
