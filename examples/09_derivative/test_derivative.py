# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest
from derivative import (
    initialize,
    analytical_solution,
    compute_derivative,
)

@pytest.mark.parametrize("n", [16, 32])
def test_initialize(n: int) -> None:
    X, Y, ikx, iky, u = initialize(n, n, n)
    
    # Check shapes
    assert X.shape == (n, n)
    assert Y.shape == (n, n)
    assert ikx.shape == (n, n//2+1)
    assert iky.shape == (n, n//2+1)
    assert u.shape == (n, n, n)
    
    # Check dx, dy
    lx, ly = 2*np.pi, 2*np.pi
    dx, dy = lx / n, ly / n
    np.testing.assert_allclose(X[0, 1] - X[0, 0], dx)
    np.testing.assert_allclose(Y[1, 0] - Y[0, 0], dx)
    
    # Check dikx, diky
    dikx, diky = 1j * 2 * np.pi / lx, 1j * 2 * np.pi / ly
    np.testing.assert_allclose(ikx[0, 1], dikx)
    np.testing.assert_allclose(iky[1, 0], diky)
    
    # Check the Hermitian symmetry of iky
    # iky is Hermitian, so iky[i, j] = -iky[-i, j]
    for i in range(1, n//2):
        np.testing.assert_allclose(iky[i], -iky[-i])
    
    # Check it is a batch
    u0 = u[0]
    for i in range(1, n):
        np.testing.assert_allclose(u[i], u0)
    
@pytest.mark.parametrize("n", [16, 32])
def test_analytical_solution(n: int) -> None:
    X, Y, ikx, iky, u = initialize(n, n, n)
    dudxy = analytical_solution(X, Y, n)
    
    # Check shapes
    assert dudxy.shape == (n, n, n)
    
    # Check it is a batch
    dudxy0 = dudxy[0]
    for i in range(1, n):
        np.testing.assert_allclose(dudxy[i], dudxy0)

@pytest.mark.parametrize("n", [16, 32])
def test_derivative(n: int) -> None:
    # The following function fails if it is not correct
    _ = compute_derivative(n, n, n)
    