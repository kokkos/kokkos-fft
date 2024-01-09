# kokkos-fft

[![CI](https://github.com/CExA-project/kokkos-fft/actions/workflows/cmake.yml/badge.svg)](https://github.com/CExA-project/kokkos-fft/actions)

UNOFFICIAL FFT interfaces for Kokkos C++ Performance Portability Programming EcoSystem

KokkosFFT implements local interfaces Kokkos and de facto standard FFT libraries, including fftw, cufft and hipfft. 
"Local" means not using MPI, or running within a
single MPI process without knowing about MPI. We are inclined to implement the numpy FFT interfaces based on Kokkos.
Here is the example for 1D case in python and KokkosFFT.
```python3
import numpy as np
x = np.random.rand(4)
x_hat = np.fft.rfft(x)
```

```C++
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T> using View1D = Kokkos::View<T*, execution_space>;
constexpr int n = 4;

View1D<double> x("x", n);
View1D<Kokkos::complex<double> > x_hat("x_hat", n/2+1);

Kokkos::Random_XorShift64_Pool<> random_pool(12345);
Kokkos::fill_random(x, random_pool, 1);
Kokkos::fence();

KokkosFFT::rfft(execution_space(), x, x_hat);
```

Depending on the View dimension, it automatically uses the batched plans as follows
```python3
import numpy as np
x = np.random.rand(4, 8)
x_hat = np.fft.rfft(x, axis=-1)
```

```C++
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
using execution_space = Kokkos::DefaultExecutionSpace;
template <typename T> using View2D = Kokkos::View<T**, execution_space>;
constexpr int n0 = 4, n1 = 8;

View2D<double> x("x", n0, n1);
View2D<Kokkos::complex<double> > x_hat("x_hat", n0, n1/2+1);

Kokkos::Random_XorShift64_Pool<> random_pool(12345);
Kokkos::fill_random(x, random_pool, 1);
Kokkos::fence();

int axis = -1;
KokkosFFT::rfft(execution_space(), x, x_hat, KokkosFFT::FFT_Normalization::BACKWARD, axis); // FFT along -1 axis and batched along 0th axis
```

## Building KokkosFFT

### CMake
Up to now, we just rely on the CMake options for Kokkos.
