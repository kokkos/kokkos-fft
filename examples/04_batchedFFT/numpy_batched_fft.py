# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" Example of batched FFTs with numpy.fft
"""

import numpy as np

if __name__ == '__main__':
    n0, n1, n2 = 128, 128, 16

    # 1D batched C2C FFT (Forward and Backward)
    xc2c = np.random.rand(n0, n1, n2) + 1j * np.random.rand(n0, n1, n2)
    xc2c_hat = np.fft.fft(xc2c, axis=-1)
    xc2c_inv = np.fft.ifft(xc2c_hat, axis=-1)

    # 1D batched R2C FFT
    xr2c = np.random.rand(n0, n1, n2)
    xr2c_hat = np.fft.rfft(xr2c, axis=-1)

    # 1D batched C2R FFT
    xc2r = np.random.rand(n0, n1, n2//2+1)
    xc2r_hat = np.fft.irfft(xc2r, axis=-1)
