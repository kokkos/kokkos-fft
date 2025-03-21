// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <memory>
#include <random>
#include <filesystem>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include "io_utils.hpp"

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
constexpr int TILE0 = 4, TILE1 = 32;
#else
constexpr int TILE0 = 4, TILE1 = 4;
#endif

using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T>
using View1D = Kokkos::View<T*, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View2D = Kokkos::View<T**, Kokkos::LayoutRight, execution_space>;
template <typename T>
using View3D = Kokkos::View<T***, Kokkos::LayoutRight, execution_space>;

// \brief A class to represent the grid used in the Hasegawa-Wakatani model.
struct Grid {
  //! Grid in x direction (nkx * 2 + 1)
  View1D<double> m_kx;

  //! Grid in y direction (nky + 1)
  View1D<double> m_kyh;

  //! Grid in x and y direction (nky + 1, nkx * 2 + 1)
  View2D<double> m_ksq;

  // \brief Constructor of a Grid class
  // \param nx [in] Number of grid points in the x-direction.
  // \param ny [in] Number of grid points in the y-direction.
  // \param lx [in] Length of the domain in the x-direction.
  // \param ly [in] Length of the domain in the y-direction.
  Grid(int nx, int ny, double lx, double ly) {
    int nkx = (nx - 2) / 3, nky = (ny - 2) / 3;
    int nkx2 = nkx * 2 + 1, nky2 = nky * 2 + 1, nkyh = nky + 1;
    m_kx  = View1D<double>("m_kx", nkx2);
    m_kyh = View1D<double>("m_kyh", nkyh);
    m_ksq = View2D<double>("m_ksq", nkyh, nkx2);

    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    // [0, dkx, 2*dkx, ..., nkx * dkx, -nkx * dkx, ..., -dkx]
    double dkx = lx / static_cast<double>(nkx2);
    auto h_kx  = KokkosFFT::fftfreq(host_execution_space(), nkx2, dkx);

    // [0, dky, 2*dky, ..., nky * dky]
    double dky = ly / static_cast<double>(nky2);
    auto h_kyh = KokkosFFT::rfftfreq(host_execution_space(), nky2, dky);

    // kx**2 + ky**2
    auto h_ksq = Kokkos::create_mirror_view(m_ksq);
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

// \brief A class to represent the variables used in the Hasegawa-Wakatani
// model.
struct Variables {
  //! Density and vorticity field in Fourier space
  View3D<Kokkos::complex<double>> m_fk;

  //! Potential field in Fourier space
  View2D<Kokkos::complex<double>> m_pk;

  // \brief Constructor of a Variables class
  // \param grid [in] Grid in Fourier space
  // \param init_val [in] Initial value of the variables
  Variables(const Grid& grid, double init_val = 0.001) {
    auto rand_engine      = std::mt19937(0);
    auto rand_dist        = std::uniform_real_distribution<double>(0.0, 1.0);
    constexpr int nb_vars = 2;
    const int nkx2 = grid.m_ksq.extent(1), nkyh = grid.m_ksq.extent(0);
    m_fk = View3D<Kokkos::complex<double>>("fk", nb_vars, nkyh, nkx2);
    m_pk = View2D<Kokkos::complex<double>>("pk", nkyh, nkx2);

    const Kokkos::complex<double> z(0.0, 1.0);  // Imaginary unit
    auto h_fk = Kokkos::create_mirror_view(m_fk);
    auto h_ksq =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grid.m_ksq);
    for (int iky = 0; iky < nkyh; iky++) {
      for (int ikx = 0; ikx < nkx2; ikx++) {
        double random_number = rand_dist(rand_engine);
        h_fk(0, iky, ikx) =
            init_val / (1.0 + h_ksq(iky, ikx)) *
            Kokkos::exp(z * 2.0 * Kokkos::numbers::pi * random_number);
        h_fk(1, iky, ikx) = -h_fk(0, iky, ikx) * h_ksq(iky, ikx);
      }
    }
    Kokkos::deep_copy(m_fk, h_fk);
  }
};

// \brief A class to represent the 4th order Runge-Kutta method for solving ODE
// dy/dt = f(t, y) by
// y^{n+1} = y^{n} + (k1 + 2*k2 + 2*k3 + k4)/6
// t^{n+1} = t^{n} + h
// where h is a time step and
// k1 = f(t^{n}      , y^{n}     ) * h
// k2 = f(t^{n} + h/2, y^{n}+k1/2) * h
// k3 = f(t^{n} + h/2, y^{n}+k2/2) * h
// k4 = f(t^{n} + h  , y^{n}+k3  ) * h
//
// \tparam BufferType The type of the view
template <typename BufferType>
class RK4th {
  static_assert(BufferType::rank == 1, "RK4th: BufferType must have rank 1.");
  using value_type = typename BufferType::non_const_value_type;
  using float_type = KokkosFFT::Impl::base_floating_point_type<value_type>;

  //! Order of the Runge-Kutta method
  const int m_order = 4;

  //! Time step size
  const float_type m_h;

  //! Size of the input View after flattening
  std::size_t m_array_size;

  //! Buffer views for intermediate results
  BufferType m_y, m_k1, m_k2, m_k3;

 public:
  // \brief Constructor of a RK4th class
  // \param y [in] The variable to be solved
  // \param h [in] Time step
  RK4th(const BufferType& y, float_type h) : m_h(h) {
    m_array_size = y.size();
    m_y          = BufferType("y", m_array_size);
    m_k1         = BufferType("k1", m_array_size);
    m_k2         = BufferType("k2", m_array_size);
    m_k3         = BufferType("k3", m_array_size);
  }

  auto order() { return m_order; }

  // \brief Advances the solution by one step using the Runge-Kutta method.
  // \tparam ViewType The type of the view
  // \param dydt [in] The right-hand side of the ODE
  // \param y [inout] The current solution.
  // \param step [in] The current step (0, 1, 2, or 3)
  template <typename ViewType>
  void advance(const ViewType& dydt, const ViewType& y, int step) {
    static_assert(ViewType::rank == 1, "RK4th: ViewType must have rank 1.");
    auto h      = m_h;
    auto y_copy = m_y;
    if (step == 0) {
      auto k1 = m_k1;
      Kokkos::parallel_for(
          "rk_step0",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            y_copy(i) = y(i);
            k1(i)     = dydt(i) * h;
            y(i)      = y_copy(i) + k1(i) / 2.0;
          });
    } else if (step == 1) {
      auto k2 = m_k2;
      Kokkos::parallel_for(
          "rk_step1",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            k2(i) = dydt(i) * h;
            y(i)  = y_copy(i) + k2(i) / 2.0;
          });
    } else if (step == 2) {
      auto k3 = m_k3;
      Kokkos::parallel_for(
          "rk_step2",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            k3(i) = dydt(i) * h;
            y(i)  = y_copy(i) + k3(i);
          });
    } else if (step == 3) {
      auto k1 = m_k1;
      auto k2 = m_k2;
      auto k3 = m_k3;
      Kokkos::parallel_for(
          "rk_step3",
          Kokkos::RangePolicy<execution_space, Kokkos::IndexType<std::size_t>>(
              execution_space(), 0, m_array_size),
          KOKKOS_LAMBDA(const std::size_t& i) {
            auto tmp_dy =
                (k1(i) + 2.0 * k2(i) + 2.0 * k3(i) + dydt(i) * h) / 6.0;
            y(i) = y_copy(i) + tmp_dy;
          });
    } else {
      throw std::runtime_error("step should be 0, 1, 2, or 3");
    }
  }
};

// \brief Apply the reality condition in Fourier space
// Force A to satisfy the following conditions
// A[i] == conj(A[-i]) and A[0] == 0
//
// \tparam ViewType The type of the view
// \tparam MaskViewType The type of the mask view
// \param view [inout] View to be modified (2 * nkx + 1) or (nvars, 2 * nkx + 1)
// \param mask [in] The mask view [0, 1, 1, ..., 1] (nkx + 1)
template <typename ViewType, typename MaskViewType>
void realityCondition(const ViewType& view, const MaskViewType& mask) {
  static_assert(ViewType::rank() == 1 || ViewType::rank() == 2,
                "realityCondition: View rank should be 1 or 2");
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
  }
}

// \brief A class to simulate the Hasegawa-Wakatani plasma turbulence model.
// ddns/dt + {phi,dns} + dphi/dy = - ca * (dns-phi) - nu * \nabla^4 dns
//           domg/dt + {phi,omg} = - ca * (dns-phi) - nu * \nabla^4 omg
// omg = \nabal^2 phi
//
// periodic boundary conditions in x and y
class HasegawaWakatani {
  using OdeSolverType   = RK4th<View1D<Kokkos::complex<double>>>;
  using ForwardPlanType = KokkosFFT::Plan<execution_space, View3D<double>,
                                          View3D<Kokkos::complex<double>>, 2>;
  using BackwardPlanType =
      KokkosFFT::Plan<execution_space, View3D<Kokkos::complex<double>>,
                      View3D<double>, 2>;

  // MDRanges used in the kernels
  using range2D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  using tile2D_type  = typename range2D_type::tile_type;
  using point2D_type = typename range2D_type::point_type;

  // MDRanges used in the kernels
  using range3D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  using tile3D_type  = typename range3D_type::tile_type;
  using point3D_type = typename range3D_type::point_type;

  using pair_type = std::pair<int, int>;

  //! The grid used in the simulation
  std::unique_ptr<Grid> m_grid;

  //! The variables used in the simulation
  std::unique_ptr<Variables> m_variables;

  //! The ODE solver used in the simulation
  std::unique_ptr<OdeSolverType> m_ode;

  //! The forward fft plan used in the simulation
  std::unique_ptr<ForwardPlanType> m_forward_plan;

  //! The backward fft plan used in the simulation
  std::unique_ptr<BackwardPlanType> m_backward_plan;

  //! Buffer view to store the time derivative of fk
  View3D<Kokkos::complex<double>> m_dfkdt;

  //! Buffer view to store the nonlinear term
  View3D<Kokkos::complex<double>> m_nonlinear_k;

  //! Buffer view to store the forward fft result
  View3D<Kokkos::complex<double>> m_forward_buffer;

  //! Buffer view to store the backward fft result
  View3D<Kokkos::complex<double>> m_backward_buffer;

  //! Buffer view to store the derivative of fk and pk
  View3D<Kokkos::complex<double>> m_ik_fg_all;

  //! Buffer view to store the derivative of fk and pk in real space
  View3D<double> m_dfgdx_all;

  //! Buffer view to store the convolution result
  View3D<double> m_conv;

  //! View to store the mask for reality condition
  View1D<double> m_mask;

  //! View to store the adiabacity factor
  View1D<double> m_adiabacity_factor;

  //! View to store the Poisson operator
  View2D<double> m_poisson_operator;

  //! The total number of iterations.
  const int m_nbiter;

  //! The parameters used in the simulation.
  const double m_ca  = 3.0;
  const double m_nu  = 0.001;
  const double m_eta = 3.0;
  double m_norm_coef;
  double m_dt = 0.0, m_time = 0.0;
  int m_nkx2, m_nkyh, m_nx, m_ny;
  int m_diag_it = 0, m_diag_steps = 5000;

  //! The directory to output diagnostic data.
  std::string m_out_dir;

 public:
  // \brief Constructor of a HasegawaWakatani class
  // \param nx [in] The number of grid points in each direction.
  // \param lx [in] The length of the domain in each direction.
  // \param nbiter [in] The total number of iterations.
  // \param dt [in] The time step size.
  // \param out_dir [in] The directory to output diagnostic data.
  HasegawaWakatani(int nx, double lx, int nbiter, double dt,
                   const std::string& out_dir)
      : m_nbiter(nbiter), m_dt(dt), m_nx(nx), m_ny(nx), m_out_dir(out_dir) {
    m_grid      = std::make_unique<Grid>(nx, nx, lx, lx);
    m_variables = std::make_unique<Variables>(*m_grid);
    View1D<Kokkos::complex<double>> fk(m_variables->m_fk.data(),
                                       m_variables->m_fk.size());
    m_ode        = std::make_unique<OdeSolverType>(fk, dt);
    namespace fs = std::filesystem;
    IO::mkdir(m_out_dir, fs::perms::owner_all | fs::perms::group_read |
                             fs::perms::group_exec | fs::perms::others_read |
                             fs::perms::others_exec);

    m_nkx2                = m_grid->m_ksq.extent(1);
    m_nkyh                = m_grid->m_ksq.extent(0);
    m_norm_coef           = static_cast<double>(m_nx * m_ny);
    const int nkx         = (m_nkx2 - 1) / 2;
    constexpr int nb_vars = 2;
    m_dfkdt = View3D<Kokkos::complex<double>>("dfkdt", nb_vars, m_nkyh, m_nkx2);
    m_nonlinear_k =
        View3D<Kokkos::complex<double>>("nonlinear_k", nb_vars, m_nkyh, m_nkx2);
    m_forward_buffer = View3D<Kokkos::complex<double>>(
        "forward_buffer", nb_vars, m_ny, m_nx / 2 + 1);
    m_backward_buffer = View3D<Kokkos::complex<double>>("backward_buffer", 6,
                                                        m_ny, m_nx / 2 + 1);
    m_ik_fg_all =
        View3D<Kokkos::complex<double>>("ik_fg_all", 6, m_nkyh, m_nkx2);
    m_dfgdx_all         = View3D<double>("dfgdx_all", 6, m_ny, m_nx);
    m_conv              = View3D<double>("conv", nb_vars, m_ny, m_nx);
    m_mask              = View1D<double>("mask", nkx + 1);
    m_poisson_operator  = View2D<double>("poisson_operator", m_nkyh, m_nkx2);
    m_adiabacity_factor = View1D<double>("adiabacity_factor", m_nkyh);

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
    for (int ikx = 1; ikx < nkx + 1; ikx++) {
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

    auto vork = Kokkos::subview(m_variables->m_fk, 1, Kokkos::ALL, Kokkos::ALL);
    poisson(vork, m_variables->m_pk);
    // Reality condition on ky == 0 component
    auto sub_fk =
        Kokkos::subview(m_variables->m_fk, Kokkos::ALL, 0, Kokkos::ALL);
    auto sub_pk = Kokkos::subview(m_variables->m_pk, 0, Kokkos::ALL);
    realityCondition(sub_fk, m_mask);
    realityCondition(sub_pk, m_mask);
  }

  // \brief Runs the simulation for the specified number of iterations.
  void run() {
    m_time = 0.0;
    for (int iter = 0; iter < m_nbiter; iter++) {
      diag(iter);
      solve();
      m_time += m_dt;
    }
  }

  // \brief Performs diagnostics at a given simulation time.
  // \param iter [in] The current iteration number.
  void diag(const int iter) {
    if (iter % m_diag_steps == 0) {
      diag_fields(m_diag_it);
      m_diag_it += 1;
    }
  }

  // \brief Prepare Views to be saved to a binary file
  // \param iter [in] The current iteration number.
  void diag_fields(const int iter) {
    auto rhok = Kokkos::subview(m_variables->m_fk, 0, Kokkos::ALL, Kokkos::ALL);
    auto vork = Kokkos::subview(m_variables->m_fk, 1, Kokkos::ALL, Kokkos::ALL);
    to_binary_file("phi", m_variables->m_pk, iter);
    to_binary_file("density", rhok, iter);
    to_binary_file("vorticity", vork, iter);
  }

  // \brief Saves a View to a binary file
  //
  // \tparam ViewType The type of the field to be saved.
  // \param label [in] The label of the field.
  // \param value [in] The field to be saved.
  // \param iter [in] The current iteration number.
  template <typename ViewType>
  void to_binary_file(const std::string& label, const ViewType& value,
                      const int iter) {
    View3D<double> out(label, 2, m_nkyh, m_nkx2);
    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_nkyh, m_nkx2}},
                       tile2D_type{{TILE0, TILE1}});

    Kokkos::parallel_for(
        "Complex2DtoReal3D", range, KOKKOS_LAMBDA(int iky, int ikx) {
          out(0, iky, ikx) = value(iky, ikx).real();
          out(1, iky, ikx) = value(iky, ikx).imag();
        });
    std::string file_name =
        m_out_dir + "/" + label + "_" + IO::zfill(iter, 10) + ".dat";
    auto h_out = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
    IO::to_binary(file_name, h_out);
  }

  // \brief Advances the simulation by one time step.
  void solve() {
    for (int step = 0; step < m_ode->order(); step++) {
      rhs(m_variables->m_fk, m_variables->m_pk, m_dfkdt);

      // Flatten Views for time integral
      View1D<Kokkos::complex<double>> fk(m_variables->m_fk.data(),
                                         m_variables->m_fk.size()),
          dfkdt(m_dfkdt.data(), m_dfkdt.size());
      m_ode->advance(dfkdt, fk, step);

      auto vork =
          Kokkos::subview(m_variables->m_fk, 1, Kokkos::ALL, Kokkos::ALL);
      poisson(vork, m_variables->m_pk);

      // ky == 0 component
      auto sub_fk =
          Kokkos::subview(m_variables->m_fk, Kokkos::ALL, 0, Kokkos::ALL);
      auto sub_pk = Kokkos::subview(m_variables->m_pk, 0, Kokkos::ALL);
      realityCondition(sub_fk, m_mask);
      realityCondition(sub_pk, m_mask);
    }
  }

  // \brief Computes the RHS of vorticity equation
  //
  // \tparam FViewType The type of the density and vorticity field.
  // \tparam PViewType The type of the potential field.
  // \tparam dFViewType The type of the RHS of the vorticity equation.
  // \param fk [in] The density and vorticity field.
  // \param pk [in] The potential field.
  // \param dfkdt [out] The RHS of the vorticity equation.
  template <typename FViewType, typename PViewType, typename dFViewType>
  void rhs(const FViewType& fk, const PViewType& pk, const dFViewType& dfkdt) {
    poissonBracket(fk, pk, m_nonlinear_k);

    constexpr int nb_vars  = 2;
    auto nonlinear_k       = m_nonlinear_k;
    auto adiabacity_factor = m_adiabacity_factor;
    auto kyh               = m_grid->m_kyh;
    auto ksq               = m_grid->m_ksq;
    const Kokkos::complex<double> z(0.0, 1.0);  // Imaginary unit
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
            double is_dns = in == 0 ? 1.0 : 0.0;
            dfkdt(in, iky, ikx) =
                -nonlinear_k(in, iky, ikx) -
                z * eta * tmp_kyh * tmp_pk * is_dns -
                ca * tmp_adiabacity_factor * (fk(0, iky, ikx) - tmp_pk) -
                nu * fk(in, iky, ikx) * tmp_k4;
          }
        });
  }

  // \brief Computes the Poisson bracket of two fields
  // {f,g} = (df/dx)(dg/dy) - (df/dy)(dg/dx)
  //
  // \tparam FViewType The type of the first field.
  // \tparam GViewType The type of the second field.
  // \tparam PViewType The type of the Poisson bracket.
  // \param fk [in] The first field.
  // \param gk [in] The second field.
  // \param pk [out] The Poisson bracket of the two fields.
  template <typename FViewType, typename GViewType, typename PViewType>
  void poissonBracket(const FViewType& fk, const GViewType& gk, PViewType& pk) {
    derivative(fk, gk, m_ik_fg_all);
    backwardFFT(m_ik_fg_all, m_dfgdx_all);

    // Convolution in real space
    convolution(m_dfgdx_all, m_conv);

    // Forward FFT
    forwardFFT(m_conv, pk);

    // ky == 0 component
    auto sub_pk = Kokkos::subview(pk, Kokkos::ALL, 0, Kokkos::ALL);
    realityCondition(sub_pk, m_mask);
  }

  // \brief Computes the derivative in Fourier space
  //
  // \tparam FViewType The type of the first field.
  // \tparam GViewType The type of the second field.
  // \tparam FGViewType The type of the derivative.
  // \param fk [in] The first field.
  // \param gk [in] The second field.
  // \param ik_fg_all [out] The derivative in Fourier space.
  template <typename FViewType, typename GViewType, typename FGViewType>
  void derivative(const FViewType& fk, const GViewType& gk,
                  FGViewType& ik_fg_all) {
    auto ikx_f =
        Kokkos::subview(ik_fg_all, pair_type(0, 2), Kokkos::ALL, Kokkos::ALL);
    auto iky_f =
        Kokkos::subview(ik_fg_all, pair_type(2, 4), Kokkos::ALL, Kokkos::ALL);
    auto ikx_g = Kokkos::subview(ik_fg_all, 4, Kokkos::ALL, Kokkos::ALL);
    auto iky_g = Kokkos::subview(ik_fg_all, 5, Kokkos::ALL, Kokkos::ALL);
    auto kx    = m_grid->m_kx;
    auto kyh   = m_grid->m_kyh;

    const Kokkos::complex<double> z(0.0, 1.0);  // Imaginary unit
    constexpr int nb_vars = 2;
    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_nkyh, m_nkx2}},
                       tile2D_type{{TILE0, TILE1}});

    Kokkos::parallel_for(
        range, KOKKOS_LAMBDA(int iky, int ikx) {
          const auto tmp_ikx = z * kx(ikx);
          const auto tmp_iky = z * kyh(iky);
          for (int in = 0; in < nb_vars; in++) {
            const auto tmp_fk   = fk(in, iky, ikx);
            ikx_f(in, iky, ikx) = tmp_ikx * tmp_fk;
            iky_f(in, iky, ikx) = tmp_iky * tmp_fk;
          }
          const auto tmp_gk = gk(iky, ikx);
          ikx_g(iky, ikx)   = tmp_ikx * tmp_gk;
          iky_g(iky, ikx)   = tmp_iky * tmp_gk;
        });
  }

  // \brief Performs a forward FFT transforming a real space field into Fourier
  // space.
  //
  // \tparam InViewType The type of the input field.
  // \tparam OutViewType The type of the output field.
  // \param f [in] The input field.
  // \param fk [out] The output field.
  template <typename InViewType, typename OutViewType>
  void forwardFFT(const InViewType& f, OutViewType& fk) {
    KokkosFFT::execute(*m_forward_plan, f, m_forward_buffer);

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
          if (ikx == 0) ikx_neg = 0;
          if (iky == 0) {
            iky_neg     = ny - 1;
            iky_nonzero = 1;
          }
          fk(iv, iky_nonzero, ikx_neg) =
              Kokkos::conj(forward_buffer(iv, iky_neg, ikx)) * norm_coef;
        });
  }

  // \brief Performs a backward FFT transforming a Fourier space field into real
  // space.
  //
  // \tparam InViewType The type of the input field.
  // \tparam OutViewType The type of the output field.
  // \param fk [in] The input field.
  // \param f [out] The output field.
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
          if (ikx == 0) ikx_neg = 0;
          if (iky == 0) {
            iky_neg     = ny - 1;
            iky_nonzero = 1;
          }
          backward_buffer(iv, iky_neg, ikx) =
              Kokkos::conj(fk(iv, iky_nonzero, ikx_neg));
        });
    KokkosFFT::execute(*m_backward_plan, backward_buffer, f);
  }

  // \brief Computes the convolution of two fields
  //
  // \tparam InViewType The type of the input field.
  // \tparam OutViewType The type of the output field.
  // \param dfgdx_all [in] The input field.
  // \param conv [out] The output field.
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
          }
        });
  }

  // \brief Computes the Poisson equation
  //
  // \tparam InViewType The type of the input field.
  // \tparam OutViewType The type of the output field.
  // \param vork [in] The Fourier representation of the vorticity field.
  // \param phik [out] The solution of the Poisson equation in Fourier space.
  template <typename InViewType, typename OutViewType>
  void poisson(const InViewType& vork, OutViewType& phik) {
    range2D_type range(point2D_type{{0, 0}}, point2D_type{{m_nkyh, m_nkx2}},
                       tile2D_type{{TILE0, TILE1}});
    auto poisson_operator = m_poisson_operator;
    Kokkos::parallel_for(
        "poisson", range, KOKKOS_LAMBDA(int iky, int ikx) {
          phik(iky, ikx) = poisson_operator(iky, ikx) * vork(iky, ikx);
        });
  }
};

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  auto kwargs         = IO::parse_args(argc, argv);
  std::string out_dir = IO::get_arg(kwargs, "out_dir", "data_kokkos");
  int nx = 1024, nbiter = 1000000;
  double lx = 10.0, dt = 0.00025;
  HasegawaWakatani model(nx, lx, nbiter, dt, out_dir);
  Kokkos::Timer timer;
  model.run();
  Kokkos::fence();
  double seconds = timer.seconds();
  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;

  return 0;
}
