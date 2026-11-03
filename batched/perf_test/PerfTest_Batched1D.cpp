#include "PerfTest_Utils.hpp"

// Many small 1-D FFTs: a (n, nbatch) LayoutLeft view transformed along axis 0,
// so every batch is a contiguous line. Batched serial / team plans against one
// KokkosFFT batched plan over the whole view.
namespace BatchedFFTBenchmark {
namespace {

template <typename T, Variant V>
void C2C_1D(benchmark::State &state) {
  using ViewType =
      Kokkos::View<Kokkos::complex<T> **, Kokkos::LayoutLeft, execution_space>;
  const std::size_t n      = state.range(0);
  const std::size_t nbatch = batch_count(n);
  ViewType x0("x0", n, nbatch), x("x", n, nbatch), y("y", n, nbatch);
  Kokkos::Random_XorShift64_Pool<execution_space> pool(12345);
  Kokkos::fill_random(x0, pool, Kokkos::complex<T>(1, 1));

  const double bytes = 2.0 * x.size() * sizeof(Kokkos::complex<T>);
  const double flops = nbatch * c2c_flops(n);
  if constexpr (V == Variant::KokkosFFT) {
    KokkosFFT::Plan<execution_space, ViewType, ViewType, 1> plan(
        execution_space(), x, y, KokkosFFT::Direction::forward, 0);
    timed_loop(state, x0, x, bytes, flops,
               [&]() { KokkosFFT::execute(plan, x, y); });
  } else {
    KokkosFFT::Batched::Plan plan(plan_policy<V>(), x, y,
                                  KokkosFFT::Batched::AxisTag<0>{});
    run_batched(state, plan, x0, x, y, nbatch, bytes, flops);
  }
  state.counters["batch"] = static_cast<double>(nbatch);
}

template <typename T, Variant V>
void R2C_1D(benchmark::State &state) {
  using RealViewType = Kokkos::View<T **, Kokkos::LayoutLeft, execution_space>;
  using ComplexViewType =
      Kokkos::View<Kokkos::complex<T> **, Kokkos::LayoutLeft, execution_space>;
  const std::size_t n      = state.range(0);
  const std::size_t nbatch = batch_count(n);
  RealViewType x0("x0", n, nbatch), x("x", n, nbatch);
  ComplexViewType y("y", n / 2 + 1, nbatch);
  Kokkos::Random_XorShift64_Pool<execution_space> pool(12345);
  Kokkos::fill_random(x0, pool, T(1));

  const double bytes =
      x.size() * sizeof(T) + y.size() * sizeof(Kokkos::complex<T>);
  const double flops = nbatch * r2c_flops(n);
  if constexpr (V == Variant::KokkosFFT) {
    KokkosFFT::Plan<execution_space, RealViewType, ComplexViewType, 1> plan(
        execution_space(), x, y, KokkosFFT::Direction::forward, 0);
    timed_loop(state, x0, x, bytes, flops,
               [&]() { KokkosFFT::execute(plan, x, y); });
  } else {
    KokkosFFT::Batched::Plan plan(plan_policy<V>(), x, y,
                                  KokkosFFT::Batched::AxisTag<0>{});
    run_batched(state, plan, x0, x, y, nbatch, bytes, flops);
  }
  state.counters["batch"] = static_cast<double>(nbatch);
}

/// \brief Lengths: powers of two from 8 to 1024, a mixed radix (100 = 4 * 25)
/// and a prime (97: generic radix)
constexpr std::initializer_list<int> lengths_1d = {8,   16,  32,   64,  128,
                                                   256, 512, 1024, 100, 97};

/// \brief Lengths of the team-size scan: few butterflies per stage (8), one
/// warp's worth (64) and many (1024)
constexpr std::initializer_list<int> scan_lengths_1d = {8, 64, 1024};

template <typename T>
const char *type_name() {
  return sizeof(T) == sizeof(float) ? "float" : "double";
}

template <typename T, Variant V>
void register_1d() {
  const std::string suffix =
      std::string("/") + type_name<T>() + "/" + variant_name<V>();
  apply_sizes<V>(benchmark::RegisterBenchmark("C2C_1D" + suffix, C2C_1D<T, V>),
                 lengths_1d, scan_lengths_1d);
  apply_sizes<V>(benchmark::RegisterBenchmark("R2C_1D" + suffix, R2C_1D<T, V>),
                 lengths_1d, scan_lengths_1d);
}

template <typename T>
void register_1d_all() {
  register_1d<T, Variant::Serial>();
  register_1d<T, Variant::Team>();
  register_1d<T, Variant::KokkosFFT>();
}

[[maybe_unused]] const bool registered_1d = [] {
  register_1d_all<float>();
  register_1d_all<double>();
  return true;
}();

}  // namespace
}  // namespace BatchedFFTBenchmark
