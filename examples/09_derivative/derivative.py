# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" Example of derivative with numpy.fft
"""

import time
from typing import Tuple
import numpy as np

def initialize(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """
    Initialize the grid, wavenumbers, and test function values.
    The test function is defined as:
        u = sin(2 * x) + cos(3 * y)

    Parameters
    ----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    nz : int
        Number of grid points in the z-direction (batch dimension).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        X : np.ndarray
            2D array of x-coordinates on the grid.
        Y : np.ndarray
            2D array of y-coordinates on the grid.
        ikx : np.ndarray
            2D array of Fourier wavenumbers in the x-direction.
        iky : np.ndarray
            2D array of Fourier wavenumbers in the y-direction.
        u : np.ndarray
            3D array of test function values.
    """
    lx, ly = 2*np.pi, 2*np.pi

    # Define and initialize the grid
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, ny, endpoint=False)

    X, Y = np.meshgrid(x, y)

    # Initialize the wavenumbers
    ikx = np.zeros((ny, nx//2+1), dtype=np.complex128)
    iky = np.zeros((ny, nx//2+1), dtype=np.complex128)

    for iy in range(ny):
        for ix in range(nx//2):
            ikx[iy, ix] = 1j * 2 * np.pi / lx * ix

    for iy in range(ny):
        for ix in range(nx//2+1):
            _iy = iy if iy < ny//2 else iy - ny
            iky[iy, ix] = 1j * 2 * np.pi / ly * _iy

    # Initialize the data
    u = np.sin(2 * X) + np.cos(3 * Y)
    u_batched = np.repeat(u[np.newaxis, :, :], nz, axis=0)

    return X, Y, ikx, iky, u_batched

def analytical_solution(X: np.ndarray, Y: np.ndarray, nz: int) -> np.ndarray:
    """
    Compute the analytical solution for the derivative of the test function:

    The analytical derivative is given by:
        dudxy = du/dx + du/dy = 2 * cos(2 * x) - 3 * sin(3 * y)

    Parameters
    ----------
    X : np.ndarray
        2D array of x-coordinates on the grid.
    Y : np.ndarray
        2D array of y-coordinates on the grid.
    nz : int
        Number of grid points in the z-direction (batch dimension).

    Returns
    -------
    np.ndarray
        3D array of the analytical derivative values.
    """
    dudxy = 2 * np.cos(2 * X) - 3 * np.sin(3 * Y)
    return np.repeat(dudxy[np.newaxis, :, :], nz, axis=0)

def compute_derivative(nx: int, ny: int, nz: int) -> np.float64:
    """
    Compute the derivative of a function using FFT-based methods and
    compare with the analytical solution.

    The process includes:
        - Forward FFT to obtain u_hat.
        - Multiplication by wavenumbers to compute derivatives in Fourier space.
        - Inverse FFT to obtain the spatial derivative.
    The function execution time is returned.

    Parameters
    ----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    nz : int
        Number of grid points in the z-direction (batch dimension).

    Returns
    -------
    np.float64
        Time taken to compute the derivatives (in seconds).
    """
    X, Y, ikx, iky, u = initialize(nx, ny, nz)
    dudxy = analytical_solution(X, Y, nz)

    # Compute the derivative
    start = time.time()

    # Forward transform u -> u_hat (=FFT (u))
    u_hat = np.fft.rfft2(u)

    # Compute derivatives by multiplications in Fourier space
    u_hat = ikx * u_hat + iky * u_hat

    # Backward transform u_hat -> u (=IFFT (u_hat))
    u = np.fft.irfft2(u_hat)
    seconds = time.time() - start

    np.testing.assert_allclose(u, dudxy, atol=1e-10)

    return seconds

if __name__ == '__main__':
    s = compute_derivative(nx=128, ny=128, nz=128)
    print(f"2D derivative with FFT took {s} [s]")
