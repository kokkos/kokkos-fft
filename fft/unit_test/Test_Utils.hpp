// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <iomanip>

template <typename ExecutionSpace, typename AViewType, typename BViewType>
bool allclose(const ExecutionSpace& exec, const AViewType& a,
              const BViewType& b, double rtol = 1.e-5, double atol = 1.e-8) {
  constexpr std::size_t rank = AViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      "KokkosFFT::Test::allclose",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const std::size_t& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  return error == 0;
}

template <typename ExecutionSpace, typename ViewType, typename T>
void multiply(const ExecutionSpace& exec, ViewType& x, T a) {
  const auto n = x.size();
  auto* ptr_x  = x.data();

  Kokkos::parallel_for(
      "KokkosFFT::Test::multiply",
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>(exec,
                                                                          0, n),
      KOKKOS_LAMBDA(const std::size_t& i) { ptr_x[i] = ptr_x[i] * a; });
}

template <typename ViewType>
void display(ViewType& a) {
  auto label   = a.label();
  const auto n = a.size();

  auto h_a = Kokkos::create_mirror_view(a);
  Kokkos::deep_copy(h_a, a);
  auto* data = h_a.data();

  std::cout << std::scientific << std::setprecision(16) << std::flush;
  for (std::size_t i = 0; i < n; i++) {
    std::cout << label + "[" << i << "]: " << i << ", " << data[i] << std::endl;
  }
  std::cout << std::resetiosflags(std::ios_base::floatfield);
}

/// \brief Kokkos equivalent of fft1 with numpy
/// def fft1(x):
///    L = len(x)
///    phase = -2j * np.pi * (np.arange(L) / L)
///    phase = np.arange(L).reshape(-1, 1) * phase
///    return np.sum(x*np.exp(phase), axis=1)
///
/// \tparam ViewType: Input and output view type
///
/// \param in [in]: Input rank 1 view
/// \param out [out]: Output rank 1 view
template <typename ExecutionSpace, typename ViewType>
void fft1(const ExecutionSpace& exec, const ViewType& in, const ViewType& out) {
  using value_type      = typename ViewType::non_const_value_type;
  using real_value_type = KokkosFFT::Impl::base_floating_point_type<value_type>;

  static_assert(KokkosFFT::Impl::is_complex_v<value_type>,
                "fft1: ViewType must be complex");

  constexpr auto pi = Kokkos::numbers::pi_v<double>;
  const value_type I(0.0, 1.0);
  std::size_t L = in.size();

  Kokkos::parallel_for(
      "KokkosFFT::Test::fft1",
      Kokkos::TeamPolicy<ExecutionSpace>(exec, L, Kokkos::AUTO),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type&
              team_member) {
        const int j = team_member.league_rank();

        value_type sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, L),
            [&](const int i, value_type& lsum) {
              auto phase = -2 * I * pi * static_cast<real_value_type>(i) /
                           static_cast<real_value_type>(L);

              auto tmp_in = in(i);
              lsum +=
                  tmp_in * Kokkos::exp(static_cast<real_value_type>(j) * phase);
            },
            sum);

        out(j) = sum;
      });
}

/// \brief Kokkos equivalent of ifft1 with numpy
/// def ifft1(x):
///    L = len(x)
///    phase = 2j * np.pi * (np.arange(L) / L)
///    phase = np.arange(L).reshape(-1, 1) * phase
///    return np.sum(x*np.exp(phase), axis=1)
///
/// \tparam ViewType: Input and output view type
///
/// \param in [in]: Input rank 1 view
/// \param out [out]: Output rank 1 view
template <typename ExecutionSpace, typename ViewType>
void ifft1(const ExecutionSpace& exec, const ViewType& in,
           const ViewType& out) {
  using value_type      = typename ViewType::non_const_value_type;
  using real_value_type = KokkosFFT::Impl::base_floating_point_type<value_type>;

  static_assert(KokkosFFT::Impl::is_complex_v<value_type>,
                "ifft1: ViewType must be complex");

  constexpr auto pi = Kokkos::numbers::pi_v<double>;
  const value_type I(0.0, 1.0);
  std::size_t L = in.size();

  Kokkos::parallel_for(
      "KokkosFFT::Test::ifft1",
      Kokkos::TeamPolicy<ExecutionSpace>(exec, L, Kokkos::AUTO),
      KOKKOS_LAMBDA(
          const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type&
              team_member) {
        const int j = team_member.league_rank();

        value_type sum = 0;
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(team_member, L),
            [&](const int i, value_type& lsum) {
              auto phase = 2 * I * pi * static_cast<real_value_type>(i) /
                           static_cast<real_value_type>(L);

              auto tmp_in = in(i);
              lsum +=
                  tmp_in * Kokkos::exp(static_cast<real_value_type>(j) * phase);
            },
            sum);

        out(j) = sum;
      });
}

#endif
