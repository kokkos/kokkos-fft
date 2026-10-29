// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <concepts>
#include <utility>

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosFFT_Plans.hpp"
#include "KokkosFFT_Transform.hpp"
#include "KokkosFFT_Testing_Allclose.hpp"

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using test_types      = ::testing::Types<std::pair<float, Kokkos::LayoutLeft>,
                                    std::pair<float, Kokkos::LayoutRight>,
                                    std::pair<double, Kokkos::LayoutLeft>,
                                    std::pair<double, Kokkos::LayoutRight>>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct TestCallback1D : public ::testing::Test {
  using float_type   = typename T::first_type;
  using complex_type = Kokkos::complex<float_type>;
  using layout_type  = typename T::second_type;
};

struct Params {
  unsigned int original_size;
};

// A __device__ global itself can't be templated, so the callbacks below are
// function templates, and one concrete global per precision and data type
// right under each of them just instantiates it. kokkosfftCallbackLoadR/D/C/Z
// and kokkosfftCallbackStoreR/D/C/Z are backend-agnostic aliases (defined per
// backend in KokkosFFT_Cuda_types.hpp and KokkosFFT_HIP_types.hpp) resolving
// to the cufft*/hipfft* callback typedefs depending on which backend is
// active, so this stays portable without naming a vendor type directly.

// Templated on T (float, double or their complex counterparts) so the
// zero-padding logic is written once rather than duplicated per precision
// and per data type. Zero-padding: return 0 for indices beyond original_size.
// T{} is zero for both the real types and the vendor complex types (float2,
// double2).
template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION T pad_load_callback(void* dataIn, size_t offset,
                                                void* callerInfo,
                                                void* /*sharedPointer*/) {
  auto* callback_params = static_cast<Params*>(callerInfo);
  const T* in_data      = static_cast<const T*>(dataIn);

  if (offset >= callback_params->original_size) {
    return T{};
  }
  return in_data[offset];
}

// Real input (R2C), then complex input (C2C)
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadR d_load_callback_symbol_fp32 =
    pad_load_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadD d_load_callback_symbol_fp64 =
    pad_load_callback<KokkosFFT::fft_data_type<double>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadC
    d_load_callback_symbol_complex_fp32 =
        pad_load_callback<KokkosFFT::fft_data_type<Kokkos::complex<float>>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadZ
    d_load_callback_symbol_complex_fp64 =
        pad_load_callback<KokkosFFT::fft_data_type<Kokkos::complex<double>>>;

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
KOKKOS_IMPL_DEVICE_FUNCTION void crop_store_callback(void* dataOut,
                                                     size_t offset, T element,
                                                     void* callerInfo,
                                                     void* /*sharedPointer*/) {
  auto* callback_params = static_cast<Params*>(callerInfo);
  T* out_data           = static_cast<T*>(dataOut);

  if (offset < callback_params->original_size) {
    out_data[offset] = element;
  }
}

// Real output (C2R), then complex output (C2C)
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreR
    d_store_callback_symbol_fp32 =
        crop_store_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreD
    d_store_callback_symbol_fp64 =
        crop_store_callback<KokkosFFT::fft_data_type<double>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreC
    d_store_callback_symbol_complex_fp32 =
        crop_store_callback<KokkosFFT::fft_data_type<Kokkos::complex<float>>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreZ
    d_store_callback_symbol_complex_fp64 =
        crop_store_callback<KokkosFFT::fft_data_type<Kokkos::complex<double>>>;

// Minimal scale callbacks, used only to exercise the one-load/one-store-per
// -plan precondition below (not the zero-padding use case above). A plan's
// load and store callback each get their own independent params struct, so
// scaling by two distinct factors and checking the combined result proves
// neither one's params leak into or get overwritten by the other's.
struct ScaleParams {
  float scale;
};

template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION T scale_load_callback(void* dataIn, size_t offset,
                                                  void* callerInfo,
                                                  void* /*sharedPointer*/) {
  auto* p = static_cast<ScaleParams*>(callerInfo);
  return static_cast<const T*>(dataIn)[offset] * static_cast<T>(p->scale);
}

KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadR d_scale_load_fp32 =
    scale_load_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadD d_scale_load_fp64 =
    scale_load_callback<KokkosFFT::fft_data_type<double>>;

template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION void scale_store_callback(void* dataOut,
                                                      size_t offset, T element,
                                                      void* callerInfo,
                                                      void* /*sharedPointer*/) {
  auto* p = static_cast<ScaleParams*>(callerInfo);
  element.x *= p->scale;
  element.y *= p->scale;
  static_cast<T*>(dataOut)[offset] = element;
}

KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreC d_scale_store_fp32 =
    scale_store_callback<KokkosFFT::fft_data_type<Kokkos::complex<float>>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreZ d_scale_store_fp64 =
    scale_store_callback<KokkosFFT::fft_data_type<Kokkos::complex<double>>>;

// Value of the i-th input sample: a ramp for real data, a ramp with a
// non-trivial imaginary part for complex data.
template <typename ValueType>
ValueType ramp_value(const int i) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  if constexpr (KokkosFFT::Impl::is_complex_v<ValueType>) {
    return ValueType(
        static_cast<float_type>(i + 1),
        static_cast<float_type>(i + 1) * static_cast<float_type>(0.5));
  } else {
    return static_cast<ValueType>(i + 1);
  }
}

// Value of the i-th bin of the half spectrum (n_in = n/2+1 bins) fed to the
// store tests. It is a valid half of a Hermitian spectrum: the imaginary part
// is non-zero except for bin 0 and, for even n, bin n/2, which must be real.
template <typename FloatType>
Kokkos::complex<FloatType> hermitian_bin_value(const int i, const int n) {
  const bool must_be_real = (i == 0) || (n % 2 == 0 && i == n / 2);
  return Kokkos::complex<FloatType>(
      static_cast<FloatType>(i + 1),
      must_be_real
          ? static_cast<FloatType>(0)
          : static_cast<FloatType>(i + 1) * static_cast<FloatType>(0.5));
}

// Attach the zero-padding load callback matching the data type (real or
// complex) and the precision of ValueType, the input type of the plan.
template <typename ValueType, typename PlanType>
void set_pad_load_callback(PlanType& plan, const Params& params) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  if constexpr (KokkosFFT::Impl::is_real_v<ValueType>) {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_load_callback_symbol_fp32, params);
    } else {
      plan.set_callback(d_load_callback_symbol_fp64, params);
    }
  } else {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_load_callback_symbol_complex_fp32, params);
    } else {
      plan.set_callback(d_load_callback_symbol_complex_fp64, params);
    }
  }
}

// Attach the cropping store callback matching the data type (real or
// complex) and the precision of ValueType, the output type of the plan.
template <typename ValueType, typename PlanType>
void set_crop_store_callback(PlanType& plan, const Params& params) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  if constexpr (KokkosFFT::Impl::is_real_v<ValueType>) {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_store_callback_symbol_fp32, params);
    } else {
      plan.set_callback(d_store_callback_symbol_fp64, params);
    }
  } else {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_store_callback_symbol_complex_fp32, params);
    } else {
      plan.set_callback(d_store_callback_symbol_complex_fp64, params);
    }
  }
}

// Zero-padding load callback on a transform of length n whose input has
// ValueType: real (R2C) or complex (C2C). Only the first original_size
// samples of the input are meaningful, the callback zeroes the others.
template <typename ValueType, typename LayoutType>
void test_load_callback_1d(const int n, const int original_size) {
  using float_type      = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<ValueType>;
  using InView1DType    = Kokkos::View<ValueType*, LayoutType, execution_space>;
  using OutView1DType =
      Kokkos::View<Kokkos::complex<float_type>*, LayoutType, execution_space>;
  const int n_out =
      static_cast<int>(KokkosFFT::Impl::extent_after_transform(n, is_R2C));

  InView1DType x("x", n);
  OutView1DType x_out("x_out", n_out);

  // Fill the whole buffer; indices >= original_size are only meaningful
  // because the callback is expected to zero them out.
  auto x_host = Kokkos::create_mirror_view(x);
  for (int i = 0; i < n; ++i) {
    x_host(i) = ramp_value<ValueType>(i);
  }
  Kokkos::deep_copy(x, x_host);

  // R2C or C2C plan
  execution_space exec;
  KokkosFFT::Plan plan(exec, x, x_out, KokkosFFT::Direction::forward,
                       /*axis=*/0);

  Params params{static_cast<unsigned int>(original_size)};
  set_pad_load_callback<ValueType>(plan, params);

  KokkosFFT::execute(plan, x, x_out);
  Kokkos::fence();

  // Reference: same transform, no callback, on an input that is explicitly
  // zero-padded on the host instead of relying on the callback to do it.
  InView1DType x_ref("x_ref", n);
  OutView1DType x_out_ref("x_out_ref", n_out);
  auto x_ref_host = Kokkos::create_mirror_view(x_ref);
  for (int i = 0; i < n; ++i) {
    x_ref_host(i) =
        (i < original_size) ? x_host(i) : ValueType(static_cast<float_type>(0));
  }
  Kokkos::deep_copy(x_ref, x_ref_host);

  KokkosFFT::Plan plan_ref(exec, x_ref, x_out_ref,
                           KokkosFFT::Direction::forward, /*axis=*/0);
  KokkosFFT::execute(plan_ref, x_ref, x_out_ref);
  Kokkos::fence();

  EXPECT_THAT(x_out, KokkosFFT::Testing::allclose(x_out_ref, 1.e-5, 1.e-12));
}

// Cropping store callback on a transform of length n whose output has
// ValueType: real (C2R) or complex (C2C).
template <typename ValueType, typename LayoutType>
void test_store_callback_1d(const int n, const int original_size) {
  using float_type      = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  constexpr bool is_C2R = KokkosFFT::Impl::is_real_v<ValueType>;
  using InView1DType =
      Kokkos::View<Kokkos::complex<float_type>*, LayoutType, execution_space>;
  using OutView1DType = Kokkos::View<ValueType*, LayoutType, execution_space>;
  const int n_in =
      static_cast<int>(KokkosFFT::Impl::extent_after_transform(n, is_C2R));

  InView1DType in("in", n_in);
  OutView1DType out("out", n);

  auto in_host = Kokkos::create_mirror_view(in);
  for (int i = 0; i < n_in; ++i) {
    // Valid for C2R (Hermitian half spectrum) and harmless for C2C
    in_host(i) = hermitian_bin_value<float_type>(i, n);
  }
  Kokkos::deep_copy(in, in_host);

  // C2R or C2C plan
  execution_space exec;
  KokkosFFT::Plan plan(exec, in, out, KokkosFFT::Direction::backward,
                       /*axis=*/0);

  Params params{static_cast<unsigned int>(original_size)};
  set_crop_store_callback<ValueType>(plan, params);

  KokkosFFT::execute(plan, in, out);
  Kokkos::fence();

  // Reference: same transform, no callback, at the same size. KokkosFFT's
  // own normalization runs after exec_plan either way, so both plans get
  // normalized identically regardless of the callback.
  OutView1DType out_ref("out_ref", n);
  KokkosFFT::Plan plan_ref(exec, in, out_ref, KokkosFFT::Direction::backward,
                           /*axis=*/0);
  KokkosFFT::execute(plan_ref, in, out_ref);
  Kokkos::fence();

  // Only offset < original_size is a meaningful invariant (see \warning on
  // crop_store_callback above).
  auto out_sub     = Kokkos::subview(out, std::make_pair(0, original_size));
  auto out_ref_sub = Kokkos::subview(out_ref, std::make_pair(0, original_size));
  EXPECT_THAT(out_sub,
              KokkosFFT::Testing::allclose(out_ref_sub, 1.e-5, 1.e-12));
}

// End-to-end: forward transform with the zero-padding load callback, then
// backward transform with the cropping store callback, matching the actual
// motivating use case (zero-padding for non-periodic directions without
// explicit memory initialization -- see the parent repo's README) rather
// than exercising each callback in isolation. ValueType is the type of the
// signal: real (R2C then C2R) or complex (C2C both ways).
template <typename ValueType, typename LayoutType>
void test_load_store_roundtrip_1d(const int n, const int original_size) {
  using float_type      = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  constexpr bool is_R2C = KokkosFFT::Impl::is_real_v<ValueType>;
  using View1DType      = Kokkos::View<ValueType*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<float_type>*, LayoutType, execution_space>;
  const int n_c =
      static_cast<int>(KokkosFFT::Impl::extent_after_transform(n, is_R2C));

  // Only the first original_size elements are meaningful; the rest of this
  // size-n buffer is left uninitialized -- the load callback zero-pads it
  // on the fly instead of requiring an explicit memset.
  View1DType x("x", n);
  ComplexView1DType x_c("x_c", n_c);

  auto x_host = Kokkos::create_mirror_view(x);
  for (int i = 0; i < original_size; ++i) {
    x_host(i) = ramp_value<ValueType>(i);
  }
  Kokkos::deep_copy(x, x_host);

  execution_space exec;
  Params params{static_cast<unsigned int>(original_size)};

  // Forward transform with the zero-padding load callback.
  KokkosFFT::Plan plan_fwd(exec, x, x_c, KokkosFFT::Direction::forward,
                           /*axis=*/0);
  set_pad_load_callback<ValueType>(plan_fwd, params);
  KokkosFFT::execute(plan_fwd, x, x_c);
  Kokkos::fence();

  // Backward transform with the cropping store callback.
  View1DType x_out("x_out", n);
  KokkosFFT::Plan plan_bwd(exec, x_c, x_out, KokkosFFT::Direction::backward,
                           /*axis=*/0);
  set_crop_store_callback<ValueType>(plan_bwd, params);
  KokkosFFT::execute(plan_bwd, x_c, x_out);
  Kokkos::fence();

  // Round trip should reconstruct the original signal over the region the
  // user actually cared about.
  auto x_out_sub = Kokkos::subview(x_out, std::make_pair(0, original_size));
  auto x_sub     = Kokkos::subview(x, std::make_pair(0, original_size));
  EXPECT_THAT(x_out_sub, KokkosFFT::Testing::allclose(x_sub, 1.e-5, 1.e-12));
}

// A plan may have one load and one store callback set independently, each
// exactly once; setting either a second time must throw, and a failed
// second attempt must not corrupt the callback(s) already registered.
template <typename T, typename LayoutType>
void test_callback_precondition_1d() {
  const int n          = 8;
  using RealView1DType = Kokkos::View<T*, LayoutType, execution_space>;
  using ComplexView1DType =
      Kokkos::View<Kokkos::complex<T>*, LayoutType, execution_space>;

  RealView1DType x("x", n);
  ComplexView1DType x_c("x_c", n / 2 + 1);

  auto x_host = Kokkos::create_mirror_view(x);
  T sum       = static_cast<T>(0);
  for (int i = 0; i < n; ++i) {
    x_host(i) = static_cast<T>(i + 1);
    sum += x_host(i);
  }
  Kokkos::deep_copy(x, x_host);

  execution_space exec;
  KokkosFFT::Plan plan(exec, x, x_c, KokkosFFT::Direction::forward,
                       /*axis=*/0);

  // A load callback may be set once...
  ScaleParams load_params{2.0f};
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_NO_THROW(plan.set_callback(d_scale_load_fp32, load_params));
  } else {
    EXPECT_NO_THROW(plan.set_callback(d_scale_load_fp64, load_params));
  }

  // ...but not twice, even with a different scale.
  ScaleParams other_load_params{99.0f};
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_THROW(plan.set_callback(d_scale_load_fp32, other_load_params),
                 std::runtime_error);
  } else {
    EXPECT_THROW(plan.set_callback(d_scale_load_fp64, other_load_params),
                 std::runtime_error);
  }

  // A store callback can still be set independently -- load and store
  // coexist on the same plan, each with its own params...
  ScaleParams store_params{10.0f};
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_NO_THROW(plan.set_callback(d_scale_store_fp32, store_params));
  } else {
    EXPECT_NO_THROW(plan.set_callback(d_scale_store_fp64, store_params));
  }

  // ...but, likewise, not twice.
  ScaleParams other_store_params{55.0f};
  if constexpr (std::is_same_v<T, float>) {
    EXPECT_THROW(plan.set_callback(d_scale_store_fp32, other_store_params),
                 std::runtime_error);
  } else {
    EXPECT_THROW(plan.set_callback(d_scale_store_fp64, other_store_params),
                 std::runtime_error);
  }

  // The failed re-attempts above must not have corrupted the plan: the
  // originally-registered load (x2) and store (x10) scales should still be
  // the ones in effect, applied together, independently.
  KokkosFFT::execute(plan, x, x_c);
  Kokkos::fence();

  auto x_c_host = Kokkos::create_mirror_view(x_c);
  Kokkos::deep_copy(x_c_host, x_c);

  // bin 0 of a forward R2C transform is the sum of the (scaled) inputs.
  T expected = sum * static_cast<T>(2.0) * static_cast<T>(10.0);
  EXPECT_NEAR(static_cast<double>(x_c_host(0).real()),
              static_cast<double>(expected), 1.e-3);
}
}  // namespace

TYPED_TEST_SUITE(TestCallback1D, test_types);

// Transform lengths: one even and one odd, each zero-padded from
// original_size. Vendor libraries handle them differently internally (for
// real transforms, an even length can be computed as a complex transform of
// half the length, an odd length cannot), which may also change how the
// callbacks are called.
constexpr int even_size     = 30;
constexpr int even_original = 20;
constexpr int odd_size      = 31;
constexpr int odd_original  = 21;

TYPED_TEST(TestCallback1D, load_r2c_callback_1d_even) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_load_callback_1d<float_type, layout_type>(even_size, even_original);
}

TYPED_TEST(TestCallback1D, load_r2c_callback_1d_odd) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_load_callback_1d<float_type, layout_type>(odd_size, odd_original);
}

TYPED_TEST(TestCallback1D, load_c2c_callback_1d_even) {
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  test_load_callback_1d<complex_type, layout_type>(even_size, even_original);
}

TYPED_TEST(TestCallback1D, load_c2c_callback_1d_odd) {
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  test_load_callback_1d<complex_type, layout_type>(odd_size, odd_original);
}

TYPED_TEST(TestCallback1D, store_c2r_callback_1d_even) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_store_callback_1d<float_type, layout_type>(even_size, even_original);
}

TYPED_TEST(TestCallback1D, store_c2r_callback_1d_odd) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_store_callback_1d<float_type, layout_type>(odd_size, odd_original);
}

TYPED_TEST(TestCallback1D, store_c2c_callback_1d_even) {
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  test_store_callback_1d<complex_type, layout_type>(even_size, even_original);
}

TYPED_TEST(TestCallback1D, store_c2c_callback_1d_odd) {
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  test_store_callback_1d<complex_type, layout_type>(odd_size, odd_original);
}

TYPED_TEST(TestCallback1D, load_store_roundtrip_r2c_1d_even) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_load_store_roundtrip_1d<float_type, layout_type>(even_size,
                                                        even_original);
}

TYPED_TEST(TestCallback1D, load_store_roundtrip_r2c_1d_odd) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_load_store_roundtrip_1d<float_type, layout_type>(odd_size, odd_original);
}

TYPED_TEST(TestCallback1D, load_store_roundtrip_c2c_1d_even) {
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  test_load_store_roundtrip_1d<complex_type, layout_type>(even_size,
                                                          even_original);
}

TYPED_TEST(TestCallback1D, load_store_roundtrip_c2c_1d_odd) {
  using complex_type = typename TestFixture::complex_type;
  using layout_type  = typename TestFixture::layout_type;
  test_load_store_roundtrip_1d<complex_type, layout_type>(odd_size,
                                                          odd_original);
}

TYPED_TEST(TestCallback1D, callback_precondition_1d) {
  using float_type  = typename TestFixture::float_type;
  using layout_type = typename TestFixture::layout_type;
  test_callback_precondition_1d<float_type, layout_type>();
}
