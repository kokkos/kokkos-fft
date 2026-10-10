#include "PerfTest_Utils.hpp"

// Many small 2-D / 3-D FFTs (small cubes), transformed over all the
// dimensions but the batch one. Batched serial / team plans against one
// KokkosFFT batched plan over the whole view. Double precision only (compile
// time, §9).
//
// Two layouts, with one batch a contiguous block of memory in both:
//   LayoutLeft : (n, n, nbatch), FFT axes (0, 1). KokkosFFT transposes the
//                view internally for these axes: its LayoutLeft plans need
//                the axes in reverse order to avoid that.
//   LayoutRight: (nbatch, n, n), FFT axes (1, 2). KokkosFFT transforms the
//                view as it is: this is the reference without a transpose.
// In both, the last FFT axis is the halved one for R2C.
namespace BatchedFFTBenchmark {
namespace {

/// \brief View dimension of the first FFT axis: after the batch dimension
/// for LayoutRight views
template <typename Layout>
inline constexpr int first_axis_v =
    std::is_same_v<Layout, Kokkos::LayoutRight> ? 1 : 0;

template <typename Layout>
using axes_2d_type =
    KokkosFFT::Batched::AxisTag<first_axis_v<Layout>, first_axis_v<Layout> + 1>;
template <typename Layout>
using axes_3d_type =
    KokkosFFT::Batched::AxisTag<first_axis_v<Layout>, first_axis_v<Layout> + 1,
                                first_axis_v<Layout> + 2>;

template <typename Layout>
KokkosFFT::axis_type<2> kokkosfft_axes_2d() {
  const int a = first_axis_v<Layout>;
  return {a, a + 1};
}

template <typename Layout>
KokkosFFT::axis_type<3> kokkosfft_axes_3d() {
  const int a = first_axis_v<Layout>;
  return {a, a + 1, a + 2};
}

template <typename Layout, Variant V>
void C2C_2D(benchmark::State &state) {
  using T = double;
  using ViewType =
      Kokkos::View<Kokkos::complex<T> ***, Layout, execution_space>;
  const std::size_t n      = state.range(0);
  const std::size_t nbatch = batch_count(n * n);
  auto x0                  = make_batched_view<ViewType>("x0", nbatch, n, n);
  auto x                   = make_batched_view<ViewType>("x", nbatch, n, n);
  auto y                   = make_batched_view<ViewType>("y", nbatch, n, n);
  Kokkos::Random_XorShift64_Pool<execution_space> pool(12345);
  Kokkos::fill_random(x0, pool, Kokkos::complex<T>(1, 1));

  const double bytes = 2.0 * x.size() * sizeof(Kokkos::complex<T>);
  const double flops = nbatch * c2c_flops(double(n) * n);
  if constexpr (V == Variant::KokkosFFT) {
    KokkosFFT::Plan<execution_space, ViewType, ViewType, 2> plan(
        execution_space(), x, y, KokkosFFT::Direction::forward,
        kokkosfft_axes_2d<Layout>());
    timed_loop(state, x0, x, bytes, flops,
               [&]() { KokkosFFT::execute(plan, x, y); });
  } else {
    KokkosFFT::Batched::Plan plan(plan_policy<V>(), x, y,
                                  axes_2d_type<Layout>{});
    run_batched(state, plan, x0, x, y, nbatch, bytes, flops);
  }
  state.counters["batch"] = static_cast<double>(nbatch);
}

template <typename Layout, Variant V>
void R2C_2D(benchmark::State &state) {
  using T            = double;
  using RealViewType = Kokkos::View<T ***, Layout, execution_space>;
  using ComplexViewType =
      Kokkos::View<Kokkos::complex<T> ***, Layout, execution_space>;
  const std::size_t n      = state.range(0);
  const std::size_t nbatch = batch_count(n * n);
  auto x0 = make_batched_view<RealViewType>("x0", nbatch, n, n);
  auto x  = make_batched_view<RealViewType>("x", nbatch, n, n);
  // The last FFT axis is the halved one
  auto y = make_batched_view<ComplexViewType>("y", nbatch, n, n / 2 + 1);
  Kokkos::Random_XorShift64_Pool<execution_space> pool(12345);
  Kokkos::fill_random(x0, pool, T(1));

  const double bytes =
      x.size() * sizeof(T) + y.size() * sizeof(Kokkos::complex<T>);
  const double flops = nbatch * r2c_flops(double(n) * n);
  if constexpr (V == Variant::KokkosFFT) {
    KokkosFFT::Plan<execution_space, RealViewType, ComplexViewType, 2> plan(
        execution_space(), x, y, KokkosFFT::Direction::forward,
        kokkosfft_axes_2d<Layout>());
    timed_loop(state, x0, x, bytes, flops,
               [&]() { KokkosFFT::execute(plan, x, y); });
  } else {
    KokkosFFT::Batched::Plan plan(plan_policy<V>(), x, y,
                                  axes_2d_type<Layout>{});
    run_batched(state, plan, x0, x, y, nbatch, bytes, flops);
  }
  state.counters["batch"] = static_cast<double>(nbatch);
}

template <typename Layout, Variant V>
void C2C_3D(benchmark::State &state) {
  using T = double;
  using ViewType =
      Kokkos::View<Kokkos::complex<T> ****, Layout, execution_space>;
  const std::size_t n      = state.range(0);
  const std::size_t nbatch = batch_count(n * n * n);
  auto x0                  = make_batched_view<ViewType>("x0", nbatch, n, n, n);
  auto x                   = make_batched_view<ViewType>("x", nbatch, n, n, n);
  auto y                   = make_batched_view<ViewType>("y", nbatch, n, n, n);
  Kokkos::Random_XorShift64_Pool<execution_space> pool(12345);
  Kokkos::fill_random(x0, pool, Kokkos::complex<T>(1, 1));

  const double bytes = 2.0 * x.size() * sizeof(Kokkos::complex<T>);
  const double flops = nbatch * c2c_flops(double(n) * n * n);
  if constexpr (V == Variant::KokkosFFT) {
    KokkosFFT::Plan<execution_space, ViewType, ViewType, 3> plan(
        execution_space(), x, y, KokkosFFT::Direction::forward,
        kokkosfft_axes_3d<Layout>());
    timed_loop(state, x0, x, bytes, flops,
               [&]() { KokkosFFT::execute(plan, x, y); });
  } else {
    KokkosFFT::Batched::Plan plan(plan_policy<V>(), x, y,
                                  axes_3d_type<Layout>{});
    run_batched(state, plan, x0, x, y, nbatch, bytes, flops);
  }
  state.counters["batch"] = static_cast<double>(nbatch);
}

constexpr std::initializer_list<int> lengths_2d = {8, 16, 32, 64};
constexpr std::initializer_list<int> lengths_3d = {8, 16, 32};
// Team-size scan: the largest slices, where the number of lines per pass (64
// and 1024) exceeds most team sizes
constexpr std::initializer_list<int> scan_lengths_2d = {64};
constexpr std::initializer_list<int> scan_lengths_3d = {32};

template <typename Layout>
const char *layout_name() {
  return std::is_same_v<Layout, Kokkos::LayoutRight> ? "LayoutRight"
                                                     : "LayoutLeft";
}

/// \brief Names: <case>/<layout>/double/<variant>/n:<n>[/team:<t>]
template <typename Layout, Variant V>
void register_nd() {
  const std::string suffix =
      std::string("/") + layout_name<Layout>() + "/double/" + variant_name<V>();
  apply_sizes<V>(
      benchmark::RegisterBenchmark("C2C_2D" + suffix, C2C_2D<Layout, V>),
      lengths_2d, scan_lengths_2d);
  apply_sizes<V>(
      benchmark::RegisterBenchmark("R2C_2D" + suffix, R2C_2D<Layout, V>),
      lengths_2d, scan_lengths_2d);
  apply_sizes<V>(
      benchmark::RegisterBenchmark("C2C_3D" + suffix, C2C_3D<Layout, V>),
      lengths_3d, scan_lengths_3d);
}

template <typename Layout>
void register_nd_all() {
  register_nd<Layout, Variant::Serial>();
  register_nd<Layout, Variant::Team>();
  register_nd<Layout, Variant::KokkosFFT>();
}

[[maybe_unused]] const bool registered_nd = [] {
  register_nd_all<Kokkos::LayoutLeft>();
  register_nd_all<Kokkos::LayoutRight>();
  return true;
}();

}  // namespace
}  // namespace BatchedFFTBenchmark
