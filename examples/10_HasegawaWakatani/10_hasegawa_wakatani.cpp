// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <memory>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
constexpr int TILE0 = 4;
constexpr int TILE1 = 32;
#else
constexpr int TILE0 = 4;
constexpr int TILE1 = 4;
#endif

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;

// \brief A struct that manages grid in wavenumber space
struct Grid {
  //! Grid in x direction (nkx * 2 + 1)
  View1D<double> m_kx;

  //! Grid in y direction (nky + 1)
  View1D<double> m_kyh;

  //! Grid in x and y direction (nky + 1, nkx * 2 + 1)
  View2D<double> m_ksq;

  Grid(int nx, int ny, double lx, double ly) {
    int nkx = (nx - 2) / 3, nky = (ny - 2) / 3;
    int nkx2 = nkx * 2 + 1, nkyh = nky + 1;
    m_kx  = View1D<double>("m_kx", nkx2);
    m_kyh = View1D<double>("m_kyh", nkyh);
    m_ksq = View2D<double>("m_ksq", nkyh, nkx2);

    auto h_kx  = Kokkos::create_mirror_view(m_kx);
    auto h_kyh = Kokkos::create_mirror_view(m_kyh);
    auto h_ksq = Kokkos::create_mirror_view(m_ksq);

    // [0, dkx, 2*dkx, ..., nkx * dkx, -nkx * dkx, ..., -dkx]
    for (int ikx = 0; ikx < nkx + 1; ikx++) {
      h_kx(ikx) = static_cast<double>(ikx) / lx;
    }

    for (int ikx = 1; ikx < nkx + 1; ikx++) {
      h_kx(nkx2 - ikx) = -static_cast<double>(ikx) / lx;
    }

    // [0, dky, 2*dky, ..., nky * dky]
    for (int iky = 0; iky < nkyh; iky++) {
      h_kyh(iky) = static_cast<double>(iky) / ly;
    }

    // kx**2 + ky**2
    for (int iky = 0; iky < nkyh; iky++) {
      for (int ikx = 0; ikx < nkx2; ikx++) {
        h_ksq(iky, ikx) = h_kx(ikx) * h_kx(ikx) + h_kyh(iky) * h_kyh(iky);
      }
    }

    Kokkos::deep_copy(m_kx, h_kx);
    Kokkos::deep_copy(m_kyh, h_kyh);
    Kokkos::deep_copy(m_ksq, h_ksq);
  }
};

// \brief A struct that manages physical quantities in wavenumber space
struct Variables {
  View3D<Kokkos::complex<double>> m_fk;
  View2D<Kokkos::complex<double>> m_pk;

  Variables(Grid& grid, double init_val = 0.001) {
    double random_number  = 1.0;
    constexpr int nb_vars = 2;
    const int nkx2 = grid.m_ksq.extent(1), nkyh = grid.m_ksq.extent(0);
    m_fk = View3D<Kokkos::complex<double>>("m_fk", nb_vars, nkyh, nkx2);
    m_pk = View2D<Kokkos::complex<double>>("m_pk", nkyh, nkx2);

    const Kokkos::complex<double> I(0.0, 1.0);  // Imaginary unit
    auto h_fk = Kokkos::create_mirror_view(m_fk);
    auto h_ksq =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_ksq);
    for (int iky = 0; iky < nkyh; iky++) {
      for (int ikx = 0; ikx < nkx2; ikx++) {
        h_fk(0, iky, ikx) = init_val / (1.0 + h_ksq(iky, ikx)) *
                            Kokkos::exp(I * 2.0 * M_PI * random_number);
        h_fk(1, iky, ikx) = -h_fk(0, iky, ikx) * h_ksq(iky, ikx);
      }
    }
    Kokkos::deep_copy(m_fk, h_fk);
  }
};

// \brief A class that manages the time integration using
// the 4th order Runge-Kutta method
template <typename ViewType>
class RK4th {
  using value_type     = typename ViewType::non_const_value_type;
  using float_type     = KokkosFFT::Impl::base_floating_point_type<value_type>;
  using BufferViewType = View1D<value_type>;

  const int m_order = 4;
  const float_type m_h;
  std::size_t m_array_size;
  BufferViewType m_y, m_k1, m_k2, m_k3;

 public:
  RK4th(const ViewType& y, float_type h) : m_h(h) {
    m_array_size = y.size();
    m_y          = BufferViewType("m_y", m_array_size);
    m_k1         = BufferViewType("m_k1", m_array_size);
    m_k2         = BufferViewType("m_k2", m_array_size);
    m_k3         = BufferViewType("m_k3", m_array_size);
  }

  auto order() { return m_order; }

  void advance(ViewType& dydt, ViewType& y, int step) {
    auto h                = m_h;
    auto* y_data          = y.data();
    const auto* dydt_data = dydt.data();
    if (step == 0) {
      auto* y_copy_data = m_y.data();
      auto* k1_data     = m_k1.data();
      Kokkos::parallel_for(
          "rk_step0",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const int& i) {
            y_copy_data[i] = y_data[i];
            k1_data[i]     = dydt_data[i] * h;
            y_data[i]      = y_copy_data[i] + k1_data[i] / 2.0;
          });
    } else if (step == 1) {
      const auto* y_copy_data = m_y.data();
      auto* k2_data           = m_k2.data();
      Kokkos::parallel_for(
          "rk_step1",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const int& i) {
            k2_data[i] = dydt_data[i] * h;
            y_data[i]  = y_copy_data[i] + k2_data[i] / 2.0;
          });
    } else if (step == 2) {
      const auto* y_copy_data = m_y.data();
      auto* k3_data           = m_k3.data();
      Kokkos::parallel_for(
          "rk_step2",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const int& i) {
            k3_data[i] = dydt_data[i] * h;
            y_data[i]  = y_copy_data[i] + k3_data[i];
          });
    } else if (step == 3) {
      const auto* y_copy_data = m_y.data();
      const auto* k1_data     = m_k1.data();
      const auto* k2_data     = m_k2.data();
      const auto* k3_data     = m_k3.data();
      Kokkos::parallel_for(
          "rk_step3",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const int& i) {
            auto tmp_k4 = dydt_data[i] * h;
            y_data[i]   = y_copy_data[i] + (k1_data[i] + 2.0 * k2_data[i] +
                                          2.0 * k3_data[i] + tmp_k4) /
                                             6.0;
          });
    } else {
      throw std::runtime_error("step should be 0, 1, 2, or 3");
    }
  }
};

// \brief Apply the reality condition in wavenumber space
// Force A to satisfy the following conditios
// A[i] == conj(A[-i]) and A[0] == 0
//
// \tparam ViewType The type of the view
// \tparam MaskViewType The type of the mask view
// \param view View to be modified (2 * nkx + 1) or (nvars, 2 * nkx + 1)
// \param mask The mask view [0, 1, 1, ..., 1] (nkx + 1)
template <typename ViewType, typename MaskViewType>
void realityCondition(const ViewType& view, const MaskViewType& mask) {
  if constexpr (ViewType::rank() == 1) {
    const int nk0 = (view.extent(0) - 1) / 2;
    Kokkos::parallel_for(
        "reality_condition",
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
            execution_space(), 0, nk0 + 1),
        KOKKOS_LAMBDA(int i0) {
          auto tmp_view = view(i0);
          view(i0)      = mask(i0) * tmp_view;

          int dst_idx   = i0 == 0 ? 0 : 2 * nk0 + 1 - i0;
          view(dst_idx) = mask(i0) * Kokkos::conj(tmp_view);
        });
  } else if constexpr (ViewType::rank() == 2) {
    const int nk0 = view.extent(0);
    const int nk1 = (view.extent(1) - 1) / 2;

    Kokkos::parallel_for(
        "reality_condition",
        Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
            execution_space(), 0, nk1 + 1),
        KOKKOS_LAMBDA(int i1) {
          for (int i0 = 0; i0 < nk0; i0++) {
            auto tmp_view = view(i0, i1);
            view(i0, i1)  = mask(i1) * tmp_view;

            int dst_idx       = i1 == 0 ? 0 : 2 * nk1 + 1 - i1;
            view(i0, dst_idx) = mask(i1) * Kokkos::conj(tmp_view);
          }
        });
  } else {
    throw std::runtime_error("rank should be 1 or 2");
  }
}

// \brief A class that solves the Hasegawa-Wakatani equation using
// spectral methods
class HasegawaWakatani {
  using OdeSolverType   = RK4th<View3D<Kokkos::complex<double>>>;
  using ForwardPlanType = KokkosFFT::Plan<execution_space, View3D<double>,
                                          View3D<Kokkos::complex<double>>, 2>;
  using BackwardPlanType =
      KokkosFFT::Plan<execution_space, View3D<Kokkos::complex<double>>,
                      View3D<double>, 2>;

  // MDRanges used in the kernels
  using range2D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<2, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile2D_type  = typename range2D_type::tile_type;
  using point2D_type = typename range2D_type::point_type;

  // MDRanges used in the kernels
  using range3D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Default, Kokkos::Iterate::Default>>;
  using tile3D_type  = typename range3D_type::tile_type;
  using point3D_type = typename range3D_type::point_type;

  using pair_type = std::pair<int, int>;

  std::unique_ptr<Grid> m_grid;
  std::unique_ptr<Variables> m_variables;
  std::unique_ptr<OdeSolverType> m_ode;
  std::unique_ptr<ForwardPlanType> m_forward_plan;
  std::unique_ptr<BackwardPlanType> m_backward_plan;

  View3D<Kokkos::complex<double>> m_dfkdt;
  View3D<Kokkos::complex<double>> m_nonlinear_k;
  View3D<Kokkos::complex<double>> m_forward_buffer;
  View3D<Kokkos::complex<double>> m_backward_buffer;
  View3D<Kokkos::complex<double>> m_ik_fg_all;

  View3D<double> m_dfgdx_all;
  View3D<double> m_conv;

  View1D<double> m_mask;
  View1D<double> m_adiabacity_factor;
  View2D<double> m_poisson_operator;

  const int m_nbiter;
  const double m_ca  = 3.0;
  const double m_nu  = 0.01;
  const double m_eta = 5.0;
  double m_norm_coef;
  int m_nkx2, m_nkyh, m_nx, m_ny;

 public:
  HasegawaWakatani(int nx, double lx, int nbiter, double dt)
      : m_nbiter(nbiter), m_nx(nx), m_ny(nx) {
    m_grid      = std::make_unique<Grid>(nx, nx, lx, lx);
    m_variables = std::make_unique<Variables>(*m_grid);
    m_ode       = std::make_unique<OdeSolverType>(m_variables->m_fk, dt);

    m_nkx2                = m_grid->m_ksq.extent(1);
    m_nkyh                = m_grid->m_ksq.extent(0);
    m_norm_coef           = static_cast<double>(m_nx * m_ny);
    const int nkx         = (m_nkx2 - 1) / 2;
    constexpr int nb_vars = 2;
    m_dfkdt =
        View3D<Kokkos::complex<double>>("m_dfkdt", nb_vars, m_nkyh, m_nkx2);
    m_nonlinear_k    = View3D<Kokkos::complex<double>>("m_nonlinear_k", nb_vars,
                                                    m_nkyh, m_nkx2);
    m_forward_buffer = View3D<Kokkos::complex<double>>(
        "m_forward_buffer", nb_vars, m_ny, m_nx / 2 + 1);
    m_backward_buffer = View3D<Kokkos::complex<double>>("m_backward_buffer", 6,
                                                        m_ny, m_nx / 2 + 1);
    m_ik_fg_all =
        View3D<Kokkos::complex<double>>("m_ik_fg_all", 6, m_nkyh, m_nkx2);
    m_dfgdx_all         = View3D<double>("m_dfgdx_all", 6, m_ny, m_nx);
    m_conv              = View3D<double>("m_conv", nb_vars, m_ny, m_nx);
    m_mask              = View1D<double>("m_mask", nkx);
    m_poisson_operator  = View2D<double>("m_poisson_operator", m_nkyh, m_nkx2);
    m_adiabacity_factor = View1D<double>("m_adiabacity_factor", m_nkyh);

    m_forward_plan = std::make_unique<ForwardPlanType>(
        execution_space(), m_conv, m_forward_buffer,
        KokkosFFT::Direction::forward, KokkosFFT::axis_type<2>({-2, -1}));
    m_backward_plan = std::make_unique<BackwardPlanType>(
        execution_space(), m_backward_buffer, m_dfgdx_all,
        KokkosFFT::Direction::backward, KokkosFFT::axis_type<2>({-2, -1}));
    auto h_mask              = Kokkos::create_mirror_view(m_mask);
    auto h_poisson_operator  = Kokkos::create_mirror_view(m_poisson_operator);
    auto h_adiabacity_factor = Kokkos::create_mirror_view(m_adiabacity_factor);
    auto h_ksq =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_grid->m_ksq);
    auto h_kyh =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_grid->m_kyh);
    for (int ikx = 1; ikx < nkx; ikx++) {
      h_mask(ikx) = 1.0;
    }

    for (int iky = 0; iky < m_nkyh; iky++) {
      h_adiabacity_factor(iky) = h_kyh(iky) * h_kyh(iky);
      for (int ikx = 0; ikx < m_nkx2; ikx++) {
        h_poisson_operator(iky, ikx) =
            (ikx == 0 && iky == 0) ? 0.0 : -1.0 / (h_ksq(iky, ikx));
      }
    }
    Kokkos::deep_copy(m_mask, h_mask);
    Kokkos::deep_copy(m_poisson_operator, h_poisson_operator);
    Kokkos::deep_copy(m_adiabacity_factor, h_adiabacity_factor);

    auto rhok = Kokkos::subview(m_variables->m_fk, 1, Kokkos::ALL, Kokkos::ALL);
    poisson(rhok, m_variables->m_pk);
    auto sub_fk = Kokkos::subview(m_variables->m_fk, Kokkos::ALL, 0,
                                  Kokkos::ALL);  // ky == 0 component
    auto sub_pk = Kokkos::subview(m_variables->m_pk, 0,
                                  Kokkos::ALL);  // ky == 0 component
    realityCondition(sub_fk, m_mask);
    realityCondition(sub_pk, m_mask);
  }
  ~HasegawaWakatani() = default;

  void run() {
    for (int i = 0; i < m_nbiter; i++) {
      solve();
    }
  }

  void solve() {
    for (int step = 0; step < m_ode->order(); step++) {
      vorticity(m_variables->m_fk, m_variables->m_pk, m_dfkdt);
      m_ode->advance(m_dfkdt, m_variables->m_fk, step);

      auto rhok =
          Kokkos::subview(m_variables->m_fk, 1, Kokkos::ALL, Kokkos::ALL);
      poisson(rhok, m_variables->m_pk);

      // ky == 0 component
      auto sub_fk =
          Kokkos::subview(m_variables->m_fk, Kokkos::ALL, 0, Kokkos::ALL);
      auto sub_pk = Kokkos::subview(m_variables->m_pk, 0, Kokkos::ALL);
      realityCondition(sub_fk, m_mask);
      realityCondition(sub_pk, m_mask);
    }
  }

  template <typename FViewType, typename PViewType, typename dFViewType>
  void vorticity(const FViewType& fk, const PViewType& pk,
                 const dFViewType& dfkdt) {
    poissonBracket(fk, pk, m_nonlinear_k);

    constexpr int nb_vars  = 2;
    auto nonlinear_k       = m_nonlinear_k;
    auto adiabacity_factor = m_adiabacity_factor;
    auto kyh               = m_grid->m_kyh;
    auto ksq               = m_grid->m_ksq;
    const Kokkos::complex<double> I(0.0, 1.0);  // Imaginary unit
    const double eta = m_eta, ca = m_ca, nu = m_nu;

    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_nkyh, m_nkx2}},
                       tile2D_type{{TILE0, TILE1}});

    Kokkos::parallel_for(
        "vorticity", range, KOKKOS_LAMBDA(int iky, int ikx) {
          auto tmp_pk                = pk(iky, ikx);
          auto tmp_kyh               = kyh(iky);
          auto tmp_adiabacity_factor = adiabacity_factor(iky);
          auto tmp_k4                = ksq(iky, ikx) * ksq(iky, ikx);
          for (int in = 0; in < nb_vars; in++) {
            dfkdt(in, iky, ikx) =
                -nonlinear_k(in, iky, ikx) - I * eta * tmp_kyh * tmp_pk -
                ca * tmp_adiabacity_factor * (fk(in, iky, ikx) - tmp_pk) -
                nu * fk(in, iky, ikx) * tmp_k4;
          }
        });
  }

  template <typename FViewType, typename GViewType, typename PViewType>
  void poissonBracket(const FViewType& fk, const GViewType& gk, PViewType& pk) {
    multiplication(fk, gk, m_ik_fg_all);
    backwardFFT(m_ik_fg_all, m_dfgdx_all);

    // Convolution in real space
    convolution(m_dfgdx_all, m_conv);

    // Forward FFT
    forwardFFT(m_conv, pk);

    // ky == 0 component
    auto sub_pk = Kokkos::subview(pk, Kokkos::ALL, 0, Kokkos::ALL);
    realityCondition(pk, m_mask);
  }

  template <typename FViewType, typename GViewType, typename FGViewType>
  void multiplication(const FViewType& fk, const GViewType& gk,
                      FGViewType& ik_fg_all) {
    auto ikx_f =
        Kokkos::subview(ik_fg_all, pair_type(0, 2), Kokkos::ALL, Kokkos::ALL);
    auto iky_f =
        Kokkos::subview(ik_fg_all, pair_type(2, 4), Kokkos::ALL, Kokkos::ALL);
    auto ikx_g = Kokkos::subview(ik_fg_all, 4, Kokkos::ALL, Kokkos::ALL);
    auto iky_g = Kokkos::subview(ik_fg_all, 5, Kokkos::ALL, Kokkos::ALL);
    auto kx    = m_grid->m_kx;
    auto kyh   = m_grid->m_kyh;

    const Kokkos::complex<double> I(0.0, 1.0);  // Imaginary unit
    constexpr int nb_vars = 2;

    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_nkyh, m_nkx2}},
                       tile2D_type{{TILE0, TILE1}});

    Kokkos::parallel_for(
        range, KOKKOS_LAMBDA(int iky, int ikx) {
          const auto tmp_ikx = I * kx(ikx);
          const auto tmp_iky = I * kyh(iky);
          for (int in = 0; in < nb_vars; in++) {
            const auto tmp_fk   = fk(in, iky, ikx);
            ikx_f(in, iky, ikx) = tmp_ikx * tmp_fk;
            iky_f(in, iky, ikx) = tmp_iky * tmp_fk;
          };
          const auto tmp_gk = gk(iky, ikx);
          ikx_g(iky, ikx)   = tmp_ikx * tmp_gk;
          iky_g(iky, ikx)   = tmp_iky * tmp_gk;
        });
  }

  template <typename InViewType, typename OutViewType>
  void forwardFFT(const InViewType& f, OutViewType& fk) {
    m_forward_plan->execute(f, m_forward_buffer);

    auto forward_buffer = m_forward_buffer;
    auto norm_coef      = m_norm_coef;
    int nkx2 = m_nkx2, nkx = (m_nkx2 - 1) / 2, ny = m_ny, nv = 2;
    range3D_type range(point3D_type{{0, 0, 0}},
                       point3D_type{{nv, m_nkyh, nkx + 1}},
                       tile3D_type{{2, TILE0, TILE1}});

    Kokkos::parallel_for(
        "FFT_unpack", range, KOKKOS_LAMBDA(int iv, int iky, int ikx) {
          fk(iv, iky, ikx) = forward_buffer(iv, iky, ikx) * norm_coef;

          int ikx_neg = nkx2 - ikx;
          int iky_neg = (ny - iky), iky_nonzero = iky;
          if (ikx == 0) {
            ikx_neg = 0;
          };
          if (iky == 0) {
            iky_neg     = ny - 1;
            iky_nonzero = 1;
          };
          fk(iv, iky_nonzero, ikx_neg) =
              Kokkos::conj(forward_buffer(iv, iky_neg, ikx)) * norm_coef;
        });
  }

  template <typename InViewType, typename OutViewType>
  void backwardFFT(const InViewType& fk, OutViewType& f) {
    auto backward_buffer = m_backward_buffer;
    Kokkos::deep_copy(backward_buffer, 0.0);
    int nkx2 = m_nkx2, nkx = (m_nkx2 - 1) / 2, ny = m_ny, nv = 6;
    range3D_type range(point3D_type{{0, 0, 0}},
                       point3D_type{{nv, m_nkyh, nkx + 1}},
                       tile3D_type{{6, TILE0, TILE1}});

    Kokkos::parallel_for(
        "FFT_pack", range, KOKKOS_LAMBDA(int iv, int iky, int ikx) {
          backward_buffer(iv, iky, ikx) = fk(iv, iky, ikx);
          int ikx_neg                   = nkx2 - ikx;
          int iky_neg = (ny - iky), iky_nonzero = iky;
          if (ikx == 0) {
            ikx_neg = 0;
          };
          if (iky == 0) {
            iky_neg     = ny - 1;
            iky_nonzero = 1;
          };
          backward_buffer(iv, iky_neg, ikx) =
              Kokkos::conj(fk(iv, iky_nonzero, ikx_neg));
        });

    m_backward_plan->execute(backward_buffer, f);
  }

  template <typename InViewType, typename OutViewType>
  void convolution(const InViewType& dfgdx_all, OutViewType& conv) {
    auto dfdx =
        Kokkos::subview(dfgdx_all, pair_type(0, 2), Kokkos::ALL, Kokkos::ALL);
    auto dfdy =
        Kokkos::subview(dfgdx_all, pair_type(2, 4), Kokkos::ALL, Kokkos::ALL);
    auto dgdx = Kokkos::subview(dfgdx_all, 4, Kokkos::ALL, Kokkos::ALL);
    auto dgdy = Kokkos::subview(dfgdx_all, 5, Kokkos::ALL, Kokkos::ALL);

    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_ny, m_nx}},
                       tile2D_type{{TILE0, TILE1}});

    constexpr int nb_vars = 2;
    Kokkos::parallel_for(
        "convolution", range, KOKKOS_LAMBDA(int iy, int ix) {
          const auto tmp_dgdx = dgdx(iy, ix);
          const auto tmp_dgdy = dgdy(iy, ix);
          for (int in = 0; in < nb_vars; in++) {
            const auto tmp_dfdx = dfdx(in, iy, ix);
            const auto tmp_dfdy = dfdy(in, iy, ix);
            conv(in, iy, ix)    = tmp_dfdx * tmp_dgdy - tmp_dfdy * tmp_dgdx;
          };
        });
  }

  template <typename InViewType, typename OutViewType>
  void poisson(const InViewType& rhok, OutViewType& phik) {
    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_nkyh, m_nkx2}},
                       tile2D_type{{TILE0, TILE1}});

    auto poisson_operator = m_poisson_operator;
    Kokkos::parallel_for(
        "poisson", range, KOKKOS_LAMBDA(int iky, int ikx) {
          phik(iky, ikx) = poisson_operator(iky, ikx) * rhok(iky, ikx);
        });
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int nx = 1024, nbiter = 100;
    double lx = 10.0, dt = 0.005;
    HasegawaWakatani model(nx, lx, nbiter, dt);
    Kokkos::Timer timer;
    model.run();
    Kokkos::fence();
    double seconds = timer.seconds();
    std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
