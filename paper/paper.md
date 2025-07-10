---
title: 'kokkos-fft: A shared-memory FFT for the Kokkos ecosystem'
tags:
  - C++
  - FFT
  - High performance computing
  - Performance portability
authors:
  - name: Yuuichi Asahi
    orcid: 0000-0002-9997-1274
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Thomas Padioleau
    orcid: 0000-0001-5496-0013
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Paul Zehner
    orcid: 0000-0002-4811-0079
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Julien Bigot
    orcid: 0000-0002-0015-4304
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Damien Lebrun-Grandie
    orcid: 0000-0003-1952-7219
    equal-contrib: true
    affiliation: "2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Université Paris-Saclay, UVSQ, CNRS, CEA, Maison de la Simulation, 91191, Gif-sur-Yvette, France
   index: 1
 - name: Oak Ridge National Laboratory, Oak Ridge, Tennessee, US
   index: 2

date: 6 June 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

kokkos-fft provides a unified, performance-portable interface for Fast Fourier Transforms (FFTs) within the Kokkos ecosystem [@Trott2021]. It seamlessly integrates with leading local FFT libraries including FFTW, cuFFT, rocFFT, and oneMKL. Designed for simplicity and efficiency, kokkos-fft offers a user experience akin to numpy.fft for in-place and out-of-place transforms, while leveraging the raw speed of vendor-optimized libraries. A demonstration solving 2D Hasegawa-Wakatani turbulence with the Fourier spectral method illustrates how kokkos-fft can deliver significant speedups over Python-based alternatives without drastically increasing code complexity, empowering researchers to perform high-performance FFTs simply and effectively.

# Statement of need

The fast Fourier transform (FFT) is a family of fundamental algorithms that is widely used in scientific computing and other areas [@Rockmore2000]. [kokkos-fft](https://github.com/kokkos/kokkos-fft) is designed to help [Kokkos](https://github.com/kokkos/kokkos) [@Trott2022] users who are:

* developing a Kokkos application which relies on FFT libraries. E.g., fluid simulation codes with periodic boundaries, plasma turbulence, etc.

* inclined to integrate in-situ signal and image processing with FFTs. E.g., spectral analyses, low pass filtering, etc.

* willing to use de facto standard FFT libraries just like [`numpy.fft`](https://numpy.org/doc/stable/reference/routines.fft.html) [@Harris2020].

kokkos-fft can benefit such users through the following features:

* A simple interface like [`numpy.fft`](https://numpy.org/doc/stable/reference/routines.fft.html) with in-place and out-of-place transforms:  
Only accepts [Kokkos Views](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html) which corresponds to the [numpy.array](https://numpy.org/doc/stable/reference/generated/numpy.array.html), to make APIs simple and safe.

* 1D, 2D, 3D standard and real FFT functions (similar to [`numpy.fft`](https://numpy.org/doc/stable/reference/routines.fft.html)) over 1D to 8D Kokkos Views:  
Batched plans are automatically used if View dimension is larger than FFT dimension.

* A reusable [FFT plan](https://kokkosfft.readthedocs.io/en/latest/api/plan/plan.html) which wraps the vendor libraries for each Kokkos backend:  
[FFTW](http://www.fftw.org), [cuFFT](https://developer.nvidia.com/cufft), [rocFFT](https://github.com/ROCm/rocFFT), and [oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) are automatically enabled based on the enabled Kokkos backend.

* Support for multiple CPU and GPU backends:  
FFT libraries for the enabled Kokkos backend are executed on the stream/queue used in that [`ExecutionSpace`](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html) where the parallel operations are performed.

* Compile time and/or runtime errors for invalid usage (e.g. `View` extents mismatch).

There already exists a couple of libraries to offer common APIs over performant vendor FFT libraries. Relying on data structures in Python, these APIs are offered by a dedicated FFT library like FluidFFT [@Mohanan2019] or a more general library offering GPU acceleration like Jax [@jax2018github]. In C++, offering this kind of APIs is non-trivial because of the lack of standard data structures with extents and/or data locations. Thanks to [Kokkos Views](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html) and [`ExecutionSpace`](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html), we can offer simple and safe APIs, which is the unique feature of this library.

# How to use kokkos-fft

For those who are familiar with [`numpy.fft`](https://numpy.org/doc/stable/reference/routines.fft.html), you may use kokkos-fft quite easily. In fact, all of the numpy.fft functions (`numpy.fft.<function_name>`) have an analogous counterpart in kokkos-fft (`KokkosFFT::<function_name>`), which can run on the Kokkos device. In addition, kokkos-fft supports [in-place transform](https://kokkosfft.readthedocs.io/en/latest/intro/using.html#inplace-transform) and [plan reuse](https://kokkosfft.readthedocs.io/en/latest/intro/using.html#reuse-fft-plan) capabilities.

Let's start with a simple example to perform the 1D real to complex transform using `rfft` in kokkos-fft.

```C++
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);
  const int n = 4;
  Kokkos::View<double*> x("x", n);
  Kokkos::View<Kokkos::complex<double>*> x_hat("x_hat", n/2+1);
  // initialize the input array with random values
  Kokkos::DefaultExecutionSpace exec;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::fill_random(exec, x, random_pool, /*range=*/1.0);
  KokkosFFT::rfft(exec, x, x_hat);
  // block the current thread until all work enqueued into exec is finished
  exec.fence();
}
```

This is equivalent to the following Python code.

```python
import numpy as np
x = np.random.rand(4)
x_hat = np.fft.rfft(x)
```

There are two additional arguments in the Kokkos version:

* `exec`: [*Kokkos execution space instance*](https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html) that encapsulates the underlying compute resources (e.g., CPU cores, GPU devices) where the task will be dispatched for execution.

* `x_hat`: [*Kokkos Views*](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html) where the complex-valued FFT output will be stored. By accepting this view as an argument, the function allows the user to pre-allocate memory and optimize data placement, avoiding unnecessary allocations and copies.

Also, kokkos-fft only accepts [Kokkos Views](https://kokkos.org/kokkos-core-wiki/API/core/view/view.html) as input data. The accessibility of a View from `ExecutionSpace` is statically checked and will result in a compilation error if not accessible. See [documentations](https://kokkosfft.readthedocs.io/en/latest/intro/quick_start.html) for basic usage.

# Benchmark: 2D Hasegawa-Wakatani turbulence with the Fourier spectral method

As a more scientific example, we solve a typical 2D plasma turbulence model, called the Hasegawa-Wakatani equation [@Wakatani1984] using the Fourier spectral method (see \autoref{fig:hw2D} for the vorticity structure).

![Vorticity.\label{fig:hw2D}](hw2D.png)

Using Kokkos and kokkos-fft, we can easily implement the code (see [example](https://github.com/kokkos/kokkos-fft/tree/main/examples/10_HasegawaWakatani/README.md)), just like Python, while getting a significant acceleration. The core computational kernel of the code is the nonlinear term which is computed with FFTs. We construct the forward and backward FFT plans once during initialization which are reused in the time evolution loops.

We have performed a benchmark of this application over multiple backends. We performed a simulation for 100 steps with a resolution of `1024 x 1024` while I/Os are disabled. The following table shows the achieved performance.

| Device | Icelake (python) | Icelake | A100 | H100 | MI250X | PVC |
| --- | --- | --- | --- | --- | --- | --- |
| Kokkos Backend | - | OpenMP | CUDA | CUDA | HIP | SYCL |
| LOC | 568 | 738 | 738 | 738 | 738 | 738 |
| Compiler/version | Python 3.12.3 | IntelLLVM 2023.0.0 | nvcc 12.2 | nvcc 12.3 | rocm 5.7 | IntelLLVM 2024.0.2 |
| GB/s (Theoretical peak) | 205 | 205 | 1555 | 3350 | 1600 | 3276.8 |
| Elapsed time [s] | 463 | 9.28 | 0.25 | 0.14 | 0.41 | 0.30 |

Here, the testbed includes Intel Xeon Platinum 8360Y (referred to as Icelake), NVIDIA A100 and H100 GPUs, AMD MI250X GPU (1 GCD) and Intel Data Center GPU Max 1550 (referred to as PVC). On Icelake, we use 36 cores with OpenMP parallelization. As expected, the Python version is the simplest in terms of lines of code (LOC). With Kokkos and kokkos-fft, the same logic can be implemented without significantly increasing the source code size (roughly 1.5 times longer). However, the benefit is enormous, allowing a single and simple code runs on multiple architectures efficiently.

# Acknowledgements

This work has received support by the CExA Moonshot project of the CEA [cexa-project](https://cexa-project.org). This work was carried out using FUJITSU PRIMERGY GX2570 (Wisteria/BDEC-01) at The University of Tokyo. This work was partly supported by JHPCN project jh220036. This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725. This work was also granted access to the HPC resources of CINES under the allocation 2023-cin4492 made by GENCI.

# References
