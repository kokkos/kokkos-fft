// SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
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

void compute_derivative(const int nx, const int ny, const int nz,
                        double& seconds);

// \brief Initialize the grid, wavenumbers, and the test function values
// u = sin(2 * x) + cos(3 * y)
// \tparam RealView1DType: Type for 1D grids in real space
// \tparam RealView3DType: Type for 3D values in real space
// \tparam ComplexView2DType: Type for 2D values in Fourier space
//
// \param x [out]: 1D grid in x direction
// \param y [out]: 1D grid in y direction
// \param ikx [out]: 2D grid in Fourier space for x direction
// \param iky [out]: 2D grid in Fourier space for y direction
// \param u [out]: 3D field in real space
template <typename RealView1DType, typename ComplexView2DType,
          typename RealView3DType>
void initialize(RealView1DType& x, RealView1DType& y, ComplexView2DType& ikx,
                ComplexView2DType& iky, RealView3DType& u) {
  using value_type    = typename RealView1DType::non_const_value_type;
  const auto pi       = Kokkos::numbers::pi_v<double>;
  const value_type Lx = 2.0 * pi, Ly = 2.0 * pi;
  const int nx = u.extent(2), ny = u.extent(1), nz = u.extent(0);
  const value_type dx = Lx / static_cast<value_type>(nx),
                   dy = Ly / static_cast<value_type>(ny);

  // Initialize grids
  auto h_x = Kokkos::create_mirror_view(x);
  auto h_y = Kokkos::create_mirror_view(y);
  for (int ix = 0; ix < nx; ++ix) h_x(ix) = static_cast<value_type>(ix) * dx;
  for (int iy = 0; iy < ny; ++iy) h_y(iy) = static_cast<value_type>(iy) * dy;

  // Initialize wave numbers
  const Kokkos::complex<value_type> I(0.0, 1.0);  // Imaginary unit
  auto h_ikx = Kokkos::create_mirror_view(ikx);
  auto h_iky = Kokkos::create_mirror_view(iky);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx / 2; ++ix) {
      h_ikx(iy, ix) = I * 2.0 * pi * static_cast<value_type>(ix) / Lx;
    }
  }

  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx / 2 + 1; ++ix) {
      auto tmp_iy   = iy < ny / 2 ? iy : iy - ny;
      h_iky(iy, ix) = I * 2.0 * pi * static_cast<value_type>(tmp_iy) / Ly;
    }
  }

  // Initialize field
  auto h_u = Kokkos::create_mirror_view(u);
  for (int jz = 0; jz < nz; jz++) {
    for (int jy = 0; jy < ny; jy++) {
      for (int jx = 0; jx < nx; jx++) {
        h_u(jz, jy, jx) = std::sin(2.0 * h_x(jx)) + std::cos(3.0 * h_y(jy));
      }
    }
  }

  Kokkos::deep_copy(x, h_x);
  Kokkos::deep_copy(y, h_y);
  Kokkos::deep_copy(ikx, h_ikx);
  Kokkos::deep_copy(iky, h_iky);
  Kokkos::deep_copy(u, h_u);
}

// \brief Compute analytical solution of the derivative
// du/dx + du/dy = 2 * cos(2 * x) - 3 * sin(3 * y)
// \tparam RealView1DType: Type for 1D grids in real space
// \tparam RealView3DType: Type for 3D values in real space
//
// \param x [in]: 1D grid in x direction
// \param y [in]: 1D grid in y direction
// \param dudxy [out]: 3D field of the analytical derivative value
template <typename RealView1DType, typename RealView3DType>
void analytical_solution(RealView1DType& x, RealView1DType& y,
                         RealView3DType& dudxy) {
  // Copy grids to host
  auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);

  // Compute the analytical solution on host
  const int nx = dudxy.extent(2), ny = dudxy.extent(1), nz = dudxy.extent(0);
  auto h_dudxy = Kokkos::create_mirror_view(dudxy);
  for (int iz = 0; iz < nz; iz++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int ix = 0; ix < nx; ix++) {
        h_dudxy(iz, iy, ix) =
            2.0 * std::cos(2.0 * h_x(ix)) - 3.0 * std::sin(3.0 * h_y(iy));
      }
    }
  }

  Kokkos::deep_copy(dudxy, h_dudxy);
}

// \brief Compute the derivative of a function using FFT-based methods and
// compare with the analytical solution
// \param nx [in]: Number of grid points in the x-direction
// \param ny [in]: Number of grid points in the y-direction
// \param nz [in]: Number of grid points in the z-direction
// \param seconds [out]: Time taken to compute the derivatives (in seconds)
void compute_derivative(const int nx, const int ny, const int nz,
                        double& seconds) {
  // View types
  using RealView1D    = View1D<double>;
  using RealView3D    = View3D<double>;
  using ComplexView2D = View2D<Kokkos::complex<double>>;
  using ComplexView3D = View3D<Kokkos::complex<double>>;

  // Declare grids
  RealView1D x("x", nx), y("y", ny);
  ComplexView2D ikx("ikx", ny, nx / 2 + 1), iky("iky", ny, nx / 2 + 1);

  // Variables to be transformed
  RealView3D u("u", nz, ny, nx), dudxy("dudxy", nz, ny, nx);
  ComplexView3D u_hat("u_hat", nz, ny, nx / 2 + 1);

  initialize(x, y, ikx, iky, u);
  analytical_solution(x, y, dudxy);

  // MDRanges used in the kernels
  using range2D_type = Kokkos::MDRangePolicy<
      execution_space,
      Kokkos::Rank<2, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>;
  using tile2D_type  = typename range2D_type::tile_type;
  using point2D_type = typename range2D_type::point_type;

  execution_space exec;

  range2D_type range2d(exec, point2D_type{{0, 0}},
                       point2D_type{{ny, nx / 2 + 1}},
                       tile2D_type{{TILE0, TILE1}});

  // kokkos-fft plans
  KokkosFFT::Plan r2c_plan(exec, u, u_hat, KokkosFFT::Direction::forward,
                           KokkosFFT::axis_type<2>({-2, -1}));
  KokkosFFT::Plan c2r_plan(exec, u_hat, u, KokkosFFT::Direction::backward,
                           KokkosFFT::axis_type<2>({-2, -1}));

  // Start computation
  Kokkos::Timer timer;

  // Forward transform u -> u_hat (=FFT (u))
  KokkosFFT::execute(r2c_plan, u, u_hat);

  // Compute derivatives by multiplications in Fourier space
  Kokkos::parallel_for(
      "ComputeDerivative", range2d, KOKKOS_LAMBDA(const int iy, const int ix) {
        auto ikx_tmp = ikx(iy, ix), iky_tmp = iky(iy, ix);
        for (int iz = 0; iz < nz; ++iz) {
          u_hat(iz, iy, ix) =
              (ikx_tmp * u_hat(iz, iy, ix) + iky_tmp * u_hat(iz, iy, ix));
        }
      });

  // Backward transform u_hat -> u (=IFFT (u_hat))
  KokkosFFT::execute(c2r_plan, u_hat, u);  // normalization is made here
  exec.fence();
  seconds = timer.seconds();

  // Check results
  auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), u);
  auto h_dudxy =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dudxy);

  const double epsilon = 1.e-8;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < nx; ++ix) {
        if (std::abs(h_dudxy(iz, iy, ix)) <= epsilon) continue;
        auto relative_error = std::abs(h_dudxy(iz, iy, ix) - h_u(iz, iy, ix)) /
                              std::abs(h_dudxy(iz, iy, ix));
        if (relative_error > epsilon) {
          std::cerr << "Error: " << h_dudxy(iz, iy, ix)
                    << " != " << h_u(iz, iy, ix) << std::endl;
          return;
        }
      }
    }
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int nx = 128, ny = 128, nz = 128;
    double seconds = 0.0;
    compute_derivative(nx, ny, nz, seconds);
    std::cout << "2D derivative with FFT took: " << seconds << " [s]"
              << std::endl;
  }
  Kokkos::finalize();

  return 0;
}
