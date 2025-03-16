# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" Example of 1D FFTs with numpy.fft
"""

import numpy as np

if __name__ == '__main__':
    n0 = 128

    # 1D C2C FFT (Forward and Backward)
    xc2c = np.random.rand(n0) + 1j * np.random.rand(n0)
    xc2c_hat = np.fft.fft(xc2c)
    xc2c_inv = np.fft.ifft(xc2c_hat)

    # 1D R2C FFT
    xr2c = np.random.rand(n0)
    xr2c_hat = np.fft.rfft(xr2c)

    # 1D C2R FFT
    xc2r = np.random.rand(n0//2+1)
    xc2r_hat = np.fft.irfft(xc2r)
