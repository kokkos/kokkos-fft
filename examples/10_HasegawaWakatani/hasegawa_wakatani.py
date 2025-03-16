# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

"""
This script implements the Hasegawa-Wakatani plasma turbulence model simulation
with Fourier spectral method.

The simulation uses periodic boundary conditions in both x and y directions.
"""

import argparse
import pathlib
import time
from typing import Callable
from functools import partial
from dataclasses import dataclass
import numpy as np
import xarray as xr

@dataclass
class Grid:
    """
    A class to represent the grid used in the Hasegawa-Wakatani model.

    Attributes
    ----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    lx : np.float64
        Length of the domain in the x-direction.
    ly : np.float64
        Length of the domain in the y-direction.
    """
    nx: int
    ny: int
    lx: np.float64
    ly: np.float64

    def __init__(self, nx: int, ny: int, lx: np.float64, ly: np.float64) -> None:
        """
        Initializes the Grid class with the given dimensions and computes the wave numbers.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        lx : np.float64
            Length of the domain in the x-direction.
        ly : np.float64
            Length of the domain in the y-direction.
        """
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx * np.pi, ly * np.pi

        nkx, nky = (self.nx-2)//3, (self.ny-2)//3
        self.nky, self.nkx2 = nky, nkx * 2 + 1

        # Half plane with ky >= 0.
        self.kx  = np.fft.fftfreq(nkx*2+1, lx / (nkx*2+1))
        self.kyh = np.fft.rfftfreq(nky*2+1, ly / (nky*2+1))

        self.kx  = np.expand_dims(self.kx, axis=0)
        self.kyh = np.expand_dims(self.kyh, axis=1)

        KX, KY = np.meshgrid(self.kx, self.kyh)
        self.ksq = KX**2 + KY**2

        # Defined in [0:Nky, -Nkx:Nkx]
        self.inv_ksq = 1. / (1. + self.ksq)

@dataclass
class Variables:
    """
    A class to represent the variables used in the Hasegawa-Wakatani model.

    Attributes
    ----------
    fk : np.ndarray
        The Fourier representation of the density and vorticity variables.
    pk : np.ndarray
        The Fourier representation of the potential variable.
    """
    fk: np.ndarray
    pk: np.ndarray

    def __init__(self, grid: Grid, init_val: np.float64=0.001) -> None:
        """
        Initializes the Grid class with the given dimensions and computes the wave numbers.

        Parameters
        ----------
        grid : Grid
            The computational grid.
        init_val : np.float64, optional
            The initial value of the variables. Defaults to 0.001.
        """
        random_number = np.random.rand(*grid.inv_ksq.shape)
        fk0 = init_val * grid.inv_ksq * np.exp(1j * 2. * np.pi * random_number)
        fk1 = - fk0 * grid.ksq

        self.fk = np.stack([fk0, fk1])
        self.pk = np.zeros_like(fk0)

class RungeKutta4th:
    """
    A class to represent the 4th order Runge-Kutta method for solving ODE
    dy/dt = f(t, y) by
    y^{n+1} = y^{n} + (k1 + 2*k2 + 2*k3 + k4)/6
    t^{n+1} = t^{n} + h
    where h is a time step and
    k1 = f(t^{n}      , y^{n}     ) * h
    k2 = f(t^{n} + h/2, y^{n}+k1/2) * h
    k3 = f(t^{n} + h/2, y^{n}+k2/2) * h
    k4 = f(t^{n} + h  , y^{n}+k3  ) * h

    Attributes
    ----------
    order : int
        The order of the Runge-Kutta method.
    h : np.float64
        The time step size.

    Methods
    -------
    advance(f: Callable[[np.ndarray], np.ndarray], y: np.ndarray, step: int)
        Advances the solution by one step using the Runge-Kutta method.
    """
    order: int = 4
    h: np.float64
    def __init__(self, h: np.float64) -> None:
        """
        Initializes the RungeKutta4th class with the given time step size.

        Parameters
        ----------
        h : np.float64
            The time step size.
        """
        self.y  = None
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.k4 = None
        self.h  = h

    def advance(self, f: Callable[[np.ndarray], np.ndarray],
                y: np.ndarray, step: int) -> np.ndarray:
        """
        Advances the solution by one step using the Runge-Kutta method.

        Parameters
        ----------
        f : Callable[[np.ndarray], np.ndarray]
            The right-hand side function of the ODE.
        y : np.ndarray
            The current solution.
        step : int
            The current Runge-Kutta sub-step (0, 1, 2, or 3).

        Returns
        -------
        np.ndarray
            The updated solution after the sub-step.
        """
        y = np.asarray(y)
        if step==0:
            self.y = np.copy(y)
            self.k1 = f(y) * self.h
            y = self.y + self.k1/2
        elif step==1:
            self.k2 = f(y) * self.h
            y = self.y + self.k2/2
        elif step==2:
            self.k3 = f(y) * self.h
            y = self.y + self.k3
        elif step==3:
            self.k4 = f(y) * self.h
            y = self.y + (self.k1 + 2*self.k2 + 2*self.k3 + self.k4) / 6
        else:
            raise ValueError('step should be 0, 1, 2, or 3')

        return y

def realityCondition(A: np.ndarray) -> np.ndarray:
    """
    Enforces the reality condition on a 2D or 3D complex array.

    Parameters
    ----------
    A : np.ndarray
        A 2D or 3D array of complex numbers.

    Returns
    -------
    np.ndarray
        The array with the reality condition enforced.
    """
    def realityCondition2D(A_col):
        _, nkx2 = A_col.shape
        nkx = (nkx2-1)//2

        conj = np.conj(A_col[0,1:nkx+1]).copy()[::-1]
        A_col[0,nkx+1:] = conj
        A_col[0,0] = 0. + 0.j

        return A_col

    if A.ndim == 2:
        A = np.expand_dims(A, axis=0)

    tmp_A = []
    for A_col in A:
        tmp_A.append( realityCondition2D(A_col) )

    tmp_A = np.asarray(tmp_A)

    return np.squeeze(tmp_A)

def Complex2DtoReal3D(A: np.ndarray) -> np.ndarray:
    """
    Convert a 2D complex array to a 3D real array.

    Parameters
    ----------
    A : np.ndarray
        A 2D array of complex numbers.

    Returns
    -------
    np.ndarray
        A 3D array where the first dimension represents the real and imaginary parts.
    """

    real, img = np.real(A), np.imag(A)
    return np.stack([real, img])

class HasegawaWakatani:
    """
    A class to simulate the Hasegawa-Wakatani plasma turbulence model.
    ddns/dt + {phi,dns} + dphi/dy = - ca * (dns-phi) - nu * \nabla^4 dns
    domg/dt + {phi,omg} = - ca * (dns-phi) - nu * \nabla^4 omg
    omg = \nabal^2 phi

    periodic boundary conditions in x and y

    Attributes
    ----------
    ca : np.float64
        The adiabaticity parameter.
    nu : np.float64
        The viscosity coefficient.
    eta : np.float64
        The diffusion coefficient.
    it : int
        The iteration counter.
    dt : np.float64
        The time step size.
    diag_it : int
        The diagnostic iteration counter.
    diag_steps : int
        The number of steps between diagnostics.
    out_dir : str
        The directory to output diagnostic data.
    grid : Grid
        The computational grid.
    variables : Variables
        The simulation variables.
    ode : RungeKutta4th
        The ODE solver instance.
    nbiter : int
        The total number of iterations.
    poisson_operator : np.ndarray
        The spectral operator used to solve Poisson's equation.
    adiabacity_factor : np.ndarray
        Factor used in the adiabaticity-related term.

    Methods
    -------
    run()
        Runs the simulation for the specified number of iterations.
    """
    ca: np.float64 = 3.
    nu: np.float64 = 0.01
    eta: np.float64 = 3.
    it: int = 0
    dt: np.float64 = 0.005
    diag_it: int = 0
    diag_steps: int = 100
    out_dir: str = 'data_python'

    def __init__(self, nx: int, lx: int, nbiter: int, dt: np.float64,
                 out_dir: str) -> None:
        """
        Initializes the HasegawaWakatani simulation with the specified grid dimensions,
        number of iterations, time step, and output directory.

        Parameters
        ----------
        nx : int
            The number of grid points in each direction.
        lx : int
            The length of the domain in each direction.
        nbiter : int
            The total number of iterations.
        dt : np.float64
            The time step size.
        out_dir : str
            The directory to output diagnostic data.
        """
        self.dt = dt
        self.grid = Grid(nx=nx, ny=nx, lx=lx, ly=lx)
        self.variables = Variables(grid=self.grid)
        self.ode = RungeKutta4th(h = self.dt)
        self.nbiter = nbiter
        self.out_dir = out_dir
        if not pathlib.Path(self.out_dir).exists():
            pathlib.Path(self.out_dir).mkdir(parents=True)

        self.poisson_operator = np.where(self.grid.ksq == 0, 0., -1. / self.grid.ksq)
        self.adiabacity_factor = self.grid.kyh**2

        self.variables.pk = self._poisson(self.variables.fk[1])
        self.variables.fk = realityCondition(self.variables.fk)
        self.variables.pk = realityCondition(self.variables.pk)

    def run(self) -> None:
        """
        Runs the simulation for the specified number of iterations.
        """
        t = 0.
        for it in range(self.nbiter):
            self._diag(it, t)
            self._solve()
            t += self.dt

    def _diag(self, it: int, t: np.float64) -> None:
        """
        Performs diagnostics at a given simulation time.

        Parameters
        ----------
        it : int
            The current iteration number.
        t : np.float64
            The current simulation time.
        """
        if it % self.diag_steps == 0:
            self._diag_fields(self.diag_it, t)
            self.diag_it += 1

    def _diag_fields(self, it: int, t: np.float64) -> None:
        """
        Saves field diagnostics to a NetCDF file.

        Parameters
        ----------
        it : int
            The diagnostic iteration index.
        t : np.float64
            The current simulation time.
        """
        data_vars = {
            'time': t,
            'phi': (
                ['complex_axis', 'kyg', 'kxg'],
                Complex2DtoReal3D(self.variables.pk)
            ),
            'density': (
                ['complex_axis', 'kyg', 'kxg'],
                Complex2DtoReal3D(self.variables.fk[0])
            ),
            'vorticity': (
                ['complex_axis', 'kyg', 'kxg'],
                Complex2DtoReal3D(self.variables.fk[1])
            ),
        }
        coords = {
            'kxg': np.squeeze(self.grid.kx),
            'kyg': np.squeeze(self.grid.kyh),
            'complex_axis': np.arange(2),  # axis to save complex number
        }
        filename = pathlib.Path(self.out_dir) / f'fields_{it:04d}.nc'
        ds = xr.Dataset(data_vars = data_vars, coords = coords)
        ds.to_netcdf(filename)

    def _solve(self):
        """
        Advances the simulation by one time step.
        """
        for step in range(self.ode.order):
            dfkdt = partial(self._rhs, pk=self.variables.pk)
            self.variables.fk = self.ode.advance(
                f=dfkdt, y=self.variables.fk, step=step
            )
            self.variables.pk = self._poisson(fk=self.variables.fk[1])

            self.variables.fk = realityCondition(self.variables.fk)
            self.variables.pk = realityCondition(self.variables.pk)

    def _rhs(self, fk, pk):
        """
        Computes the RHS of vorticity equation

        Parameters
        ----------
        fk : np.ndarray
            The density and vorticity field.
        pk : np.ndarray
            The potential field.

        Returns
        -------
        np.ndarray
            The RHS of the vorticity equation.
        """
        pbk = np.zeros_like(fk, dtype=np.complex128)
        dfkdt = np.zeros_like(fk, dtype=np.complex128)

        phiky = 1j * self.eta * self.grid.kyh * pk
        for i in range(2):
            pbk[i] = self._poissonBracket(f=fk[i], g=pk)
            is_dns = i == 0
            dfkdt[i] = (
                -pbk[i]
                - phiky * np.float64(is_dns)
                - self.ca * self.adiabacity_factor * (fk[0] - pk)
                - self.nu * fk[i] * self.grid.ksq**2
            )

        return dfkdt

    def _poissonBracket(self, f, g):
        """
        Computes the Poisson bracket of two fields.
        {f,g} = (df/dx)(dg/dy) - (df/dy)(dg/dx)

        Parameters
        ----------
        f : np.ndarray
            The first field.
        g : np.ndarray
            The second field.

        Returns
        -------
        np.ndarray
            The Poisson bracket of the two fields.
        """
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

    def _forwardFFT(self, f):
        """
        Performs a forward FFT transforming a real space field into Fourier space.

        Parameters
        ----------
        f : np.ndarray
            The real space field.

        Returns
        -------
        np.ndarray
            The Fourier representation of the real space field.
        """
        nky, nkx2 = self.grid.nky, self.grid.nkx2
        nkx = (nkx2-1)//2
        ny, nx = self.grid.ny, self.grid.nx
        normalization_coefficient = nx*ny

        fk = np.fft.rfft2(f)
        fk_buffer = np.zeros((nky+1, nkx2), dtype=np.complex128)

        fk_buffer[:,0:nkx+1] = fk[:nky+1,0:nkx+1]
        for iy in range(1, nky+1):
            for ix in range(0, -(nkx+1), -1):
                fk_buffer[iy,ix] = np.conj(fk[ny-iy, -ix])

        return fk_buffer * normalization_coefficient

    def _backwardFFT(self, fk):
        """
        Performs a backward FFT transforming a Fourier space field into real space.

        Parameters
        ----------
        fk : np.ndarray
            The Fourier space field.

        Returns
        -------
        np.ndarray
            The real representation of the Fourier space field.
        """
        nky, nkx2 = self.grid.nky, self.grid.nkx2
        nkx = (nkx2-1)//2
        ny, nx = self.grid.ny, self.grid.nx

        fk_buffer = np.zeros((ny,nx//2+1), dtype=np.complex128)
        fk_buffer[:nky+1,0:nkx+1] = fk[:,0:nkx+1]

        for iy in range(1, nky+1):
            for ix in range(0, -(nkx+1), -1):
                fk_buffer[ny-iy, -ix] = np.conj(fk[iy,ix])

        f = np.fft.irfft2(fk_buffer)
        return f

    def _poisson(self, fk):
        """
        Solves the Poisson equation in Fourier space.

        Parameters
        ----------
        fk : np.ndarray
            The Fourier representation of the vorticity field.

        Returns
        -------
        np.ndarray
            The solution of the Poisson equation in Fourier space.
        """
        return self.poisson_operator * fk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-nx', nargs='?', type=int, default=128,
                        help="Number of grid points in each spatial dimension")
    parser.add_argument('-lx', nargs='?', type=float, default=10.0,
                        help="Physical domain length in one spatial direction")
    parser.add_argument('-nbiter', nargs='?', type=int, default=10000,
                        help="Total number of simulation iterations")
    parser.add_argument('-dt', nargs='?', type=float, default=0.005,
                        help="Time step size for the integration scheme")
    parser.add_argument('-out_dir', nargs='?', type=str, default='data_python',
                        help="Directory where diagnostic output files will be saved")
    args = parser.parse_args()

    model = HasegawaWakatani(**vars(args))
    start = time.time()
    model.run()
    seconds = time.time() - start

    print(f'Elapsed time: {seconds} [s]')
