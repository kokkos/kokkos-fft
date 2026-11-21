<!--
SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

<div style="line-height: 25%" align="center">
<h3> 3D Poisson equation solved with callbacks </h3>
<img src=../../docs/imgs/poisson_callback.gif>
</div>

# Solving the 3D Poisson equation with zero-padding callbacks

This example solves ∇²φ = f on [0,Lx]×[0,Ly]×[0,Lz] with per-axis periodic or
non-periodic boundary conditions. Non-periodic axes are zero-padded **on the
fly by a load callback** and cropped on the way back by a **store callback**,
instead of explicitly zero-initializing the padded buffer. The animation above
sweeps through the solution obtained this way along z.

The run is verified against a reference run that uses explicit zero-padding
(no callbacks); with all-periodic BCs the result is also compared to the
analytical solution φ = -f/k². See the [documentation
page](https://kokkosfft.readthedocs.io/en/latest/samples/11_Callback.html)
for a full description of the problem, the spectral method, and the callback
indexing scheme.

## How to run

Callbacks require relocatable device code, so the build needs
`-DKokkosFFT_ENABLE_CALLBACK=ON` together with
`-DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON` (HIP builds:
`-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON`):

```bash
# Compilation (adjust Kokkos_ARCH_* to your GPU)
cmake -B build \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_ADA89=ON \
      -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \
      -DKokkosFFT_ENABLE_INTERNAL_KOKKOS=ON \
      -DKokkosFFT_ENABLE_EXAMPLES=ON \
      -DKokkosFFT_ENABLE_CALLBACK=ON
cmake --build build --target PoissonEquation -j 8

# Run (64^3 grid, domain [0,1]^3, all non-periodic), dumping the
# callback-computed solution for plotting
./build/examples/11_Callback/PoissonEquation 64 64 64 1.0 1.0 1.0 0 0 0 \
    --dump phi.bin

# Render a sweep through the solution along z as a GIF
python examples/11_Callback/postscript.py phi.bin \
    --axis z --output ../../docs/imgs/poisson_callback.gif
```

Expected output ends with:

```text
Max |callback - explicit-pad| = 0
SUCCESS
```
