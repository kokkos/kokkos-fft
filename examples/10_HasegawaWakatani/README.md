<!--
SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

<div style style=”line-height: 25%” align="center">
<h3> 2D Hasegawa-Wakatani turbulence </h3>
<img src=../../docs/imgs/hw_anime.gif>
</div>

# Solving 2D Hasegawa-Wakatani turbulence with Fourier spectral method

For turbulence simulations, we sometimes consider periodic boundaries by assuming that a system is homogeneous and isotropic. Under the periodic boundary conditions, we can solve the system of equations with Fourier spectral method. Here, we consider a typical 2D turbulence plasma turbulence model, called Hasegawa-Wakatani equation `[Wakatani, 1984]`. With kokkos and kokkos-fft, we can easily implement the code just like python while getting a significant acceleration.

## Numerical description

In Fourier space, 2D Hasegawa-Wakatani model can be described as

 ![Vorticity eq](https://latex.codecogs.com/svg.latex?\frac{\partial\hat{u}_k}{\partial{t}}+\\{\tilde{\phi},\tilde{u}\\}_k=-C_{k}\left(\hat{\phi}_k-\hat{n}_k\right)-\nu{k}^4\hat{u}_k)

 ![Continuity eq](https://latex.codecogs.com/svg.latex?\frac{\partial\hat{n}_k}{\partial{t}}+\\{\tilde{\phi},\tilde{n}\\}_k=-i\eta{k_{y}}\hat{\phi}_k+C_{k}\left(\hat{\phi}_k-\hat{n}_k\right)-\nu{k}^4\hat{n}_k)

The upper one is the vorticity equation and the lower one is the continuity equation, where ![potential](https://latex.codecogs.com/svg.latex?\phi), ![density](https://latex.codecogs.com/svg.latex?n) and ![vorticity](https://latex.codecogs.com/svg.latex?u) are potential, density and vorticity respectively (shown in the animation). The vorticity ![vorticity](https://latex.codecogs.com/svg.latex?u) satisfies the following equation,

 ![Poisson eq](https://latex.codecogs.com/svg.latex?\tilde{u}=\nabla^2\tilde{\phi})

The nonlinear term can be described as follows, which can be efficiently computed with FFTs.

![Poisson bracket](https://latex.codecogs.com/svg.latex?\\{\tilde{\phi},\tilde{n}\\}_k=-\sum_{k'}\sum_{k''}\delta_{k+k'+k'',0}\left(k_{x}'k_{y}''-k_{y}'k_{x}''\right)\hat{\phi}^{\ast}_{k'}\hat{n}^{\ast}_{k''})

## Prerequisites

For kokkos version, we need the followings (for the latest requirements, see [README.md](../../README.md)):
   
* `CMake 3.22+`
* `Kokkos 4.4+`
* `gcc 8.3.0+` (CPUs)
* `IntelLLVM 2023.0.0+` (CPUs, Intel GPUs)
* `nvcc 11.0.0+` (NVIDIA GPUs)
* `rocm 5.3.0+` (AMD GPUs)

For python version, we need the followings:

* `python 3.8+`
* `numpy`
* `matplotlib`
* `xarray[io]`, `xarray[viz]`
* `joblib`

## python and kokkos implementation

The physical variables ![potential](https://latex.codecogs.com/svg.latex?\phi), ![density](https://latex.codecogs.com/svg.latex?n) and ![vorticity](https://latex.codecogs.com/svg.latex?u) are stored in two arrays. ![potential](https://latex.codecogs.com/svg.latex?\phi) is stored in a complex 2D array `pk`. ![density](https://latex.codecogs.com/svg.latex?n) and ![vorticity](https://latex.codecogs.com/svg.latex?u) are stacked and stored in a complex 3D array `fk`. Since these are originally real variables, they have Hermitian symmetry for their Fourier representation. Thus, we represent them in the half domain in y direction, whose shape is `[0:nky+1, -nkx:nkx]`. The numerical constants ![C_k](https://latex.codecogs.com/svg.latex?C_{k}), ![eta](https://latex.codecogs.com/svg.latex?\eta), and ![nu](https://latex.codecogs.com/svg.latex?\nu_{k}) in the vorticity and continuity equations are respectively stored as member variables, `ca`, `eta`, and `nu`.

Let us consider the most computationally intensive kernel of this code, the nonlinear term. In python, it is implemented by

```python
def _poissonBracket(self, f, g):
    ikx_f = 1j * self.grid.kx  * f
    iky_f = 1j * self.grid.kyh * f
    ikx_g = 1j * self.grid.kx  * g
    iky_g = 1j * self.grid.kyh * g

    # Inverse FFT complex [ny, nx/2+1] => real [ny, nx]
    dfdx = self._backwardFFT(ikx_f)
    dfdy = self._backwardFFT(iky_f)
    dgdx = self._backwardFFT(ikx_g)
    dgdy = self._backwardFFT(iky_g)

    # Convolution in real space
    conv = dfdx * dgdy - dfdy * dgdx

    # Forward FFT real [ny, nx] => [ny, nx/2+1]
    poisson_bracket = self._forwardFFT(conv)

    # Reality condition
    poisson_bracket = realityCondition(poisson_bracket)
    return poisson_bracket
```

We have 4 backward FFTs on `ikx_f`, `iky_f`, `ikx_g` and `iky_g`. Then, we perform convolution in real space followed by
forward FFT on the result to compute the poisson bracket. The equivalent kokkos code is given by

```C++
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
```

`derivative` and `convolution` are parallelized by [`parallel_for`](https://kokkos.org/kokkos-core-wiki/API/core/parallel-dispatch/parallel_for.html) with [`MDRangePolicy`](https://kokkos.org/kokkos-core-wiki/API/core/policies/MDRangePolicy.html). For forward and backward FFTs, we create plans at initialization which are reused with [`KokkosFFT::execute`](https://kokkosfft.readthedocs.io/en/latest/intro/using.html#reuse-fft-plan).

## How to run the simulation

You can compile and run the code in the following way.

```bash
# Compilation
cmake -B build \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ENABLE_CUDA=ON \
      -DKokkos_ARCH_AMPERE80=ON \
      -DKokkosFFT_ENABLE_EXAMPLES=ON
cmake --build build -j 8

# Run the kokkos code
./build/10_hasegawa_wakatani/10_hasegawa_wakatani 
```

Python version is also available which can be executed by

```bash
python examples/10_hasegawa_wakatani/hasegawa_wakatani.py
```

After the simulations, you will get the following standard output.

```bash
Elapsed time: 2371.14 [s]
```

You can visualize the simulation results with a python script `postscript.py` by

```bash
python examples/10_hasegawa_wakatani/postscript.py -data_dir <path/to/output>
```

The default `<path/to/output>` is `build/data_kokkos` for kokkos and `examples/10_hasegawa_wakatani/data_python` for python.

## Acknowledgement

The author Y.A thanks [Dr. S. Maeyama](https://github.com/smaeyama) who offered us the original code in Fortran.

## Reference

```bibtex
@article{Wakatani1984,
    author = {Wakatani, Masahiro and Hasegawa, Akira},
    title = {A collisional drift wave description of plasma edge turbulence},
    journal = {The Physics of Fluids},
    volume = {27},
    number = {3},
    pages = {611-618},
    year = {1984},
    month = {03},
    abstract = {Model mode‐coupling equations for the resistive drift wave instability are numerically solved for realistic parameters found in tokamak edge plasmas. The Bohm diffusion is found to result if the parallel wavenumber is chosen to maximize the growth rate for a given value of the perpendicular wavenumber. The saturated turbulence energy has a broad frequency spectrum with a large fluctuation level proportional to κ̄ (=ρs/Ln, the normalized inverse scale length of the density gradient) and a wavenumber spectrum of the two‐dimensional Kolmogorov–Kraichnan type, ∼k−3.},
    issn = {0031-9171},
    doi = {10.1063/1.864660},
    url = {https://doi.org/10.1063/1.864660},
    eprint = {https://pubs.aip.org/aip/pfl/article-pdf/27/3/611/12476138/611\_1\_online.pdf},
}
```
