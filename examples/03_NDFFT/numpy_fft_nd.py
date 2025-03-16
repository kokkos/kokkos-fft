# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" Example of ND FFTs with numpy.fft
"""

import numpy as np

if __name__ == '__main__':
    n0, n1, n2 = 128, 128, 16

    # 3D C2C FFT (Forward and Backward)
    xc2c = np.random.rand(n0, n1, n2) + 1j * np.random.rand(n0, n1, n2)
    xc2c_hat = np.fft.fftn(xc2c)
    xc2c_inv = np.fft.ifftn(xc2c_hat)

    # 3D R2C FFT
    xr2c = np.random.rand(n0, n1, n2)
    xr2c_hat = np.fft.rfftn(xr2c)

    # 3D C2R FFT
    xc2r = np.random.rand(n0, n1, n2//2+1)
    xc2r_hat = np.fft.irfftn(xc2r)
