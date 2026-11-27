// SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
//
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

/// \example PoissonEquation.cpp
/// \brief 3D Poisson equation solver using KokkosFFT with callbacks.
///
/// Solves: ∇²φ = f on [0,Lx]×[0,Ly]×[0,Lz] with per-axis periodic or
/// non-periodic boundary conditions.
///
/// Pipeline (mirrors a PSolver-style batched execution): for each axis,
/// a batched 1D FFT is applied. Non-periodic axes are zero-padded on the
/// fly by a LOAD callback (instead of explicitly zeroing the padded
/// buffer) and cropped on the way back by a STORE callback.
///
/// Verification: the callback run is compared against a reference run that
/// uses explicit zero-padding (no callbacks). With all-periodic BCs and a
/// pure Fourier mode RHS, the result is also compared to the analytical
/// solution φ = -f / k².
///
/// Usage: ./PoissonEquation <Nx> <Ny> <Nz> <Lx> <Ly> <Lz> [px] [py] [pz]
///                           [--dump <file.bin>]
///   px/py/pz: 1 = periodic, 0 = non-periodic (default: 0 0 0)
///   --dump: write the callback-computed solution phi to <file.bin> as
///           [int64 Nx, Ny, Nz][float64 Lx, Ly, Lz][Nx*Ny*Nz float64 phi],
///           k-fastest (C order), for plotting with postscript.py

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <KokkosFFT.hpp>

// =============================================================================
// CALLBACK PARAMETERS AND FUNCTIONS
// =============================================================================

namespace {
using execution_space = Kokkos::DefaultExecutionSpace;
using LayoutType      = Kokkos::LayoutRight;

struct Params {
  unsigned int original_size;  // valid length along the transformed axis
  unsigned int padded_size;    // full 1D FFT length along that axis
};

// Zero-padding load callback. For a batched 1D FFT the vendor library passes
// the global linear offset in the (transposed) buffer, where the transformed
// axis is fastest, so the index along that axis is offset % padded_size.
// Returns T{} (zero) for indices beyond original_size.
template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION T pad_load_callback(void* dataIn, size_t offset,
                                                void* callerInfo,
                                                void* /*sharedPointer*/) {
  auto* callback_params = static_cast<Params*>(callerInfo);
  const T* in_data      = static_cast<const T*>(dataIn);
  if (offset % callback_params->padded_size >= callback_params->original_size) {
    return T{};
  }
  return in_data[offset];
}

// One concrete __device__ global per precision and data kind
// (real for R2C load / C2R store, complex for C2C).
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadR d_load_callback_fp32 =
    pad_load_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadD d_load_callback_fp64 =
    pad_load_callback<KokkosFFT::fft_data_type<double>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadC d_load_callback_cfp32 =
    pad_load_callback<KokkosFFT::fft_data_type<Kokkos::complex<float>>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackLoadZ d_load_callback_cfp64 =
    pad_load_callback<KokkosFFT::fft_data_type<Kokkos::complex<double>>>;

// Cropping store callback: only writes for indices < original_size along the
// transformed axis (offset % padded_size). Normalization is handled by
// execute()'s norm argument.
template <typename T>
KOKKOS_IMPL_DEVICE_FUNCTION void crop_store_callback(void* dataOut,
                                                     size_t offset, T element,
                                                     void* callerInfo,
                                                     void* /*sharedPointer*/) {
  auto* callback_params = static_cast<Params*>(callerInfo);
  T* out_data           = static_cast<T*>(dataOut);
  if (offset % callback_params->padded_size < callback_params->original_size) {
    out_data[offset] = element;
  }
}

KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreR d_store_callback_fp32 =
    crop_store_callback<KokkosFFT::fft_data_type<float>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreD d_store_callback_fp64 =
    crop_store_callback<KokkosFFT::fft_data_type<double>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreC d_store_callback_cfp32 =
    crop_store_callback<KokkosFFT::fft_data_type<Kokkos::complex<float>>>;
KOKKOS_IMPL_DEVICE_FUNCTION kokkosfftCallbackStoreZ d_store_callback_cfp64 =
    crop_store_callback<KokkosFFT::fft_data_type<Kokkos::complex<double>>>;

// Attach the zero-padding load callback matching ValueType (real/complex)
// and precision.
template <typename ValueType, typename PlanType>
void set_pad_load_callback(PlanType& plan, const Params& params) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  if constexpr (KokkosFFT::Impl::is_real_v<ValueType>) {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_load_callback_fp32, params);
    } else {
      plan.set_callback(d_load_callback_fp64, params);
    }
  } else {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_load_callback_cfp32, params);
    } else {
      plan.set_callback(d_load_callback_cfp64, params);
    }
  }
}

// Attach the cropping store callback matching ValueType (real/complex)
// and precision.
template <typename ValueType, typename PlanType>
void set_crop_store_callback(PlanType& plan, const Params& params) {
  using float_type = KokkosFFT::Impl::base_floating_point_type<ValueType>;
  if constexpr (KokkosFFT::Impl::is_real_v<ValueType>) {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_store_callback_fp32, params);
    } else {
      plan.set_callback(d_store_callback_fp64, params);
    }
  } else {
    if constexpr (std::same_as<float_type, float>) {
      plan.set_callback(d_store_callback_cfp32, params);
    } else {
      plan.set_callback(d_store_callback_cfp64, params);
    }
  }
}

using View3D = Kokkos::View<double***, LayoutType, execution_space>;
using CView3D =
    Kokkos::View<Kokkos::complex<double>***, LayoutType, execution_space>;

// =============================================================================
// POISSON SOLVE (batched 1D FFTs per axis)
//   use_callbacks: zero-pad/crop via load/store callbacks, otherwise the
//   padded buffer is explicitly zero-initialized (reference).
// =============================================================================
View3D solve_poisson(const execution_space& exec, const View3D& input, int Nx,
                     int Ny, int Nz, double Lx, double Ly, double Lz, bool px,
                     bool py, bool pz, bool use_callbacks) {
  const int x_size = px ? Nx : 2 * Nx;
  const int y_size = py ? Ny : 2 * Ny;
  const int z_size = pz ? Nz : 2 * Nz;
  const int n0c    = x_size / 2 + 1;

  // =====================================================================
  // FORWARD PASS — batched 1D FFT per axis
  // =====================================================================

  // Step 1: FFT along X (R2C, axis 0)
  View3D padded_input("padded_input", x_size, Ny, Nz);
  if (!use_callbacks) Kokkos::deep_copy(padded_input, 0.0);
  Kokkos::deep_copy(Kokkos::subview(padded_input, std::make_pair(0, Nx),
                                    Kokkos::ALL, Kokkos::ALL),
                    input);
  exec.fence();

  CView3D fftX("fftX", n0c, Ny, Nz);
  KokkosFFT::Plan plan_X(exec, padded_input, fftX,
                         KokkosFFT::Direction::forward, 0);
  if (use_callbacks && !px) {
    set_pad_load_callback<double>(plan_X,
                                  Params{static_cast<unsigned int>(Nx),
                                         static_cast<unsigned int>(x_size)});
  }
  KokkosFFT::execute(plan_X, padded_input, fftX);
  exec.fence();

  // Step 2: FFT along Y (C2C, axis 1)
  CView3D padded_fftX("padded_fftX", n0c, y_size, Nz);
  if (!use_callbacks)
    Kokkos::deep_copy(padded_fftX, Kokkos::complex<double>(0.0, 0.0));
  Kokkos::deep_copy(Kokkos::subview(padded_fftX, Kokkos::ALL,
                                    std::make_pair(0, Ny), Kokkos::ALL),
                    fftX);
  exec.fence();

  KokkosFFT::Plan plan_Y(exec, padded_fftX, padded_fftX,
                         KokkosFFT::Direction::forward, 1);
  if (use_callbacks && !py) {
    set_pad_load_callback<Kokkos::complex<double>>(
        plan_Y, Params{static_cast<unsigned int>(Ny),
                       static_cast<unsigned int>(y_size)});
  }
  KokkosFFT::execute(plan_Y, padded_fftX, padded_fftX);
  exec.fence();

  // Step 3: FFT along Z (C2C, axis 2)
  CView3D fftY("fftY", n0c, y_size, z_size);
  if (!use_callbacks)
    Kokkos::deep_copy(fftY, Kokkos::complex<double>(0.0, 0.0));
  Kokkos::deep_copy(
      Kokkos::subview(fftY, Kokkos::ALL, Kokkos::ALL, std::make_pair(0, Nz)),
      padded_fftX);
  exec.fence();

  KokkosFFT::Plan plan_Z(exec, fftY, fftY, KokkosFFT::Direction::forward, 2);
  if (use_callbacks && !pz) {
    set_pad_load_callback<Kokkos::complex<double>>(
        plan_Z, Params{static_cast<unsigned int>(Nz),
                       static_cast<unsigned int>(z_size)});
  }
  KokkosFFT::execute(plan_Z, fftY, fftY);
  exec.fence();

  // =====================================================================
  // KERNEL MULTIPLICATION in Fourier space: φ̂ = f̂ / (-k²)
  // =====================================================================
  const double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;
  Kokkos::parallel_for(
      "poisson_kernel",
      Kokkos::MDRangePolicy<execution_space, Kokkos::Rank<3>>(
          {0, 0, 0}, {n0c, y_size, z_size}),
      KOKKOS_LAMBDA(int i, int j, int k) {
        // The (possibly padded) array spans x_size*dx, so k = 2πm/(x_size*dx)
        double kx = (i <= x_size / 2)
                        ? 2.0 * M_PI * i / (x_size * dx)
                        : 2.0 * M_PI * (i - x_size) / (x_size * dx);
        double ky = (j <= y_size / 2)
                        ? 2.0 * M_PI * j / (y_size * dy)
                        : 2.0 * M_PI * (j - y_size) / (y_size * dy);
        double kz = (k <= z_size / 2)
                        ? 2.0 * M_PI * k / (z_size * dz)
                        : 2.0 * M_PI * (k - z_size) / (z_size * dz);
        double k2 = kx * kx + ky * ky + kz * kz;
        if (k2 > 0.0) {
          fftY(i, j, k) = fftY(i, j, k) / (-k2);
        } else {
          // DC component: mean of f must be zero for a solution to exist
          fftY(i, j, k) = Kokkos::complex<double>(0.0, 0.0);
        }
      });
  exec.fence();

  // =====================================================================
  // INVERSE PASS — batched 1D IFFT per axis at the padded size
  // =====================================================================

  // Step 4: IFFT along Z (C2C, axis 2, in-place)
  KokkosFFT::Plan plan_iZ(exec, fftY, fftY, KokkosFFT::Direction::backward, 2);
  KokkosFFT::execute(plan_iZ, fftY, fftY);
  exec.fence();

  // Crop Z back to Nz
  CView3D unpad_Z("unpad_Z", n0c, y_size, Nz);
  Kokkos::deep_copy(unpad_Z, Kokkos::subview(fftY, Kokkos::ALL, Kokkos::ALL,
                                             std::make_pair(0, Nz)));
  exec.fence();

  // Step 5: IFFT along Y (C2C, axis 1, in-place)
  KokkosFFT::Plan plan_iY(exec, unpad_Z, unpad_Z,
                          KokkosFFT::Direction::backward, 1);
  KokkosFFT::execute(plan_iY, unpad_Z, unpad_Z);
  exec.fence();

  // Crop Y back to Ny
  CView3D unpad_Y("unpad_Y", n0c, Ny, Nz);
  Kokkos::deep_copy(unpad_Y,
                    Kokkos::subview(unpad_Z, Kokkos::ALL, std::make_pair(0, Ny),
                                    Kokkos::ALL));
  exec.fence();

  // Step 6: IFFT along X (C2R, axis 0). Output has the padded x_size along
  // axis 0; with callbacks the store callback crops it to Nx, otherwise the
  // padded output buffer is fully written and cropped by subview below.
  View3D output("output", x_size, Ny, Nz);
  KokkosFFT::Plan plan_iX(exec, unpad_Y, output, KokkosFFT::Direction::backward,
                          0);
  if (use_callbacks && !px) {
    set_crop_store_callback<double>(plan_iX,
                                    Params{static_cast<unsigned int>(Nx),
                                           static_cast<unsigned int>(x_size)});
  }
  KokkosFFT::execute(plan_iX, unpad_Y, output);
  exec.fence();

  View3D phi("phi", Nx, Ny, Nz);
  Kokkos::deep_copy(phi, Kokkos::subview(output, std::make_pair(0, Nx),
                                         Kokkos::ALL, Kokkos::ALL));
  exec.fence();
  return phi;
}

}  // namespace

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  int exit_code = 0;
  {
    if (argc < 7) {
      std::cerr << "Usage: " << argv[0]
                << " <Nx> <Ny> <Nz> <Lx> <Ly> <Lz> [px] [py] [pz]"
                   " [--dump <file.bin>]\n"
                << "  px/py/pz: 1=periodic, 0=non-periodic (default: 0 0 0)\n"
                << "  --dump: write the callback-computed solution to "
                   "<file.bin> for postscript.py\n";
      Kokkos::finalize();
      return 1;
    }
    const int Nx    = std::atoi(argv[1]);
    const int Ny    = std::atoi(argv[2]);
    const int Nz    = std::atoi(argv[3]);
    const double Lx = std::atof(argv[4]);
    const double Ly = std::atof(argv[5]);
    const double Lz = std::atof(argv[6]);
    const bool px   = (argc > 7) ? std::atoi(argv[7]) != 0 : false;
    const bool py   = (argc > 8) ? std::atoi(argv[8]) != 0 : false;
    const bool pz   = (argc > 9) ? std::atoi(argv[9]) != 0 : false;

    std::string dump_path;
    for (int i = 7; i < argc - 1; ++i) {
      if (std::string(argv[i]) == "--dump") dump_path = argv[i + 1];
    }

    execution_space exec;

    // RHS: a pure Fourier mode so that, for the all-periodic case, the
    // analytical solution φ = -f/k² is exact for the spectral solver.
    //   f(x,y,z) = sin(2πx/Lx)·sin(2πy/Ly)·sin(2πz/Lz)
    View3D input("input", Nx, Ny, Nz);
    auto h_input = Kokkos::create_mirror_view(input);
    for (int i = 0; i < Nx; ++i)
      for (int j = 0; j < Ny; ++j)
        for (int k = 0; k < Nz; ++k) {
          double x         = Lx * i / Nx;
          double y         = Ly * j / Ny;
          double z         = Lz * k / Nz;
          h_input(i, j, k) = std::sin(2.0 * M_PI * x / Lx) *
                             std::sin(2.0 * M_PI * y / Ly) *
                             std::sin(2.0 * M_PI * z / Lz);
        }
    Kokkos::deep_copy(input, h_input);
    exec.fence();

    // Run with callbacks and reference run with explicit zero-padding
    View3D phi_cb  = solve_poisson(exec, input, Nx, Ny, Nz, Lx, Ly, Lz, px, py,
                                   pz, /*use_callbacks=*/true);
    View3D phi_ref = solve_poisson(exec, input, Nx, Ny, Nz, Lx, Ly, Lz, px, py,
                                   pz, /*use_callbacks=*/false);

    auto h_cb =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi_cb);
    auto h_ref =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, phi_ref);

    // Optionally dump the callback-computed solution for postscript.py:
    // [int64 Nx, Ny, Nz][float64 Lx, Ly, Lz][Nx*Ny*Nz float64], k-fastest.
    if (!dump_path.empty()) {
      std::ofstream ofs(dump_path, std::ios::binary);
      if (!ofs) {
        std::cerr << "Cannot open " << dump_path << " for writing\n";
        Kokkos::finalize();
        return 1;
      }
      const std::int64_t dims[3] = {Nx, Ny, Nz};
      const double lens[3]       = {Lx, Ly, Lz};
      ofs.write(reinterpret_cast<const char*>(dims), sizeof(dims));
      ofs.write(reinterpret_cast<const char*>(lens), sizeof(lens));
      ofs.write(reinterpret_cast<const char*>(h_cb.data()),
                static_cast<std::streamsize>(sizeof(double) * h_cb.size()));
      std::cout << "Wrote callback solution to " << dump_path << "\n";
    }

    // Check 1: callbacks must reproduce the explicit-padding reference
    double max_diff = 0.0;
    for (int i = 0; i < Nx; ++i)
      for (int j = 0; j < Ny; ++j)
        for (int k = 0; k < Nz; ++k) {
          double d = std::abs(h_cb(i, j, k) - h_ref(i, j, k));
          if (d > max_diff) max_diff = d;
        }
    std::cout << "Max |callback - explicit-pad| = " << max_diff << "\n";

    // Check 2: all-periodic case against the analytical solution
    //   ∇²φ = f  =>  φ = -f / ((2π/Lx)² + (2π/Ly)² + (2π/Lz)²)
    double max_err = 0.0;
    if (px && py && pz) {
      const double lambda = 4.0 * M_PI * M_PI / (Lx * Lx) +
                            4.0 * M_PI * M_PI / (Ly * Ly) +
                            4.0 * M_PI * M_PI / (Lz * Lz);
      for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
          for (int k = 0; k < Nz; ++k) {
            double x     = Lx * i / Nx;
            double y     = Ly * j / Ny;
            double z     = Lz * k / Nz;
            double f_val = std::sin(2.0 * M_PI * x / Lx) *
                           std::sin(2.0 * M_PI * y / Ly) *
                           std::sin(2.0 * M_PI * z / Lz);
            double expected = -f_val / lambda;
            double err      = std::abs(h_cb(i, j, k) - expected);
            if (err > max_err) max_err = err;
          }
      std::cout << "Max error vs analytical solution (periodic) = " << max_err
                << "\n";
    }

    const bool ok =
        (max_diff < 1e-10) && (!px || !py || !pz || max_err < 1e-10);
    if (!ok) {
      std::cerr << "FAIL\n";
      exit_code = 1;
    } else {
      std::cout << "SUCCESS\n";
    }
  }
  Kokkos::finalize();
  return exit_code;
}
