# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" Example of 2D FFTs with numpy.fft
"""

import numpy as np

if __name__ == '__main__':
    n0, n1 = 128, 128

    # 2D C2C FFT (Forward and Backward)
    xc2c = np.random.rand(n0, n1) + 1j * np.random.rand(n0, n1)
    xc2c_hat = np.fft.fft2(xc2c)
    xc2c_inv = np.fft.ifft2(xc2c_hat)

    # 2D R2C FFT
    xr2c = np.random.rand(n0, n1)
    xr2c_hat = np.fft.rfft2(xr2c)

    # 2D C2R FFT
    xc2r = np.random.rand(n0, n1//2+1)
    xc2r_hat = np.fft.irfft2(xc2r)
