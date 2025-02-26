<!--
SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
-->

# Solving 2D Hasegawa-Wakatani turbulence with Fourier spectral method

For turbulence simulations, we sometimes consider periodic boundaries by assuming that a system is homogeneous and isotropic. Under the periodic boundary conditions, we can solve the system of equations with Fourier spectral method. Here, we consider a typical 2D turbulence plasma turbulence model, called Hasegawa-Wakatani equation `[Wakatani, 1984](#Wakatani1984)`. With kokkos and kokkos-fft, we can easily implement the code just like python while getting a significant acceleration.

## Numerical description

In Fourier space, 2D Hasegawa-Wakatani model can be described as

 ![Continuity eq](https://latex.codecogs.com/svg.latex?\frac{\partial\hat{u}_k}{\partial{t}}+\\{\tilde{\phi},\tilde{u}\\}_k=-C_{k}\left(\hat{\phi}_k-\hat{n}_k\right)-\nu{k}^4\hat{u}_k)

 ![Vorticity eq](https://latex.codecogs.com/svg.latex?\frac{\partial\hat{n}_k}{\partial{t}}+\\{\tilde{\phi},\tilde{n}\\}_k=-i\kappa{k_{y}}\hat{\phi}_k+C_{k}\left(\hat{\phi}_k-\hat{n}_k\right)-\nu{k}^4\hat{n}_k)

The upper one is the vorticity equation and the lower one is the continuity equation, where ![vorticity](https://latex.codecogs.com/svg.latex?u), ![density](https://latex.codecogs.com/svg.latex?n) and ![potential](https://latex.codecogs.com/svg.latex?\phi) are vorticity, density and potential respectively. The vorticity ![vorticity](https://latex.codecogs.com/svg.latex?u) satisfies the following equation.

 ![Poisson eq](https://latex.codecogs.com/svg.latex?\tilde{u}=\nabla^2\tilde{\phi})

The nonlinear term can be described as follows, which can be efficiently computed with FFTs.

![Poisson bracket](https://latex.codecogs.com/svg.latex?\\{\tilde{\phi},\tilde{n}\\}_k=-\sum_{k'}\sum_{k''}\delta_{k+k'+k'',0}\left(k_{x}'k_{y}''-k_{y}'k_{x}''\right)\hat{\phi}^{\ast}_{k'}\hat{n}^{\ast}_{k''})

## python and kokkos implementation

As imagined, the nonlinear term is the core computational kernel of this code. In python, it is implemented by

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
Elapsed time: 62.834 [s]
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
