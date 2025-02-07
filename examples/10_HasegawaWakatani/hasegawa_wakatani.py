import argparse
import pathlib
import time
import numpy as np
import xarray as xr
from functools import partial
from dataclasses import dataclass

@dataclass
class Grid:
    nx: int
    ny: int
    lx: np.float64
    ly: np.float64

    def __init__(self, nx, ny, lx, ly):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx * np.pi, ly * np.pi

        nkx, nky = (self.nx-2)//3, (self.ny-2)//3
        self.nky, self.nkx2 = nky, nkx * 2 + 1

        self.kx = np.linspace(-nkx, nkx+1, nkx*2+1, endpoint=False) / lx
        self.ky = np.linspace(-nky, nky+1, nky*2+1, endpoint=False) / ly

        # Set zero-frequency component to the zero-th component
        self.kx = np.fft.ifftshift(self.kx)
        self.ky = np.fft.ifftshift(self.ky)

        # Half plane with ky >= 0.
        self.kyh = self.ky[:nky+1]

        self.kx  = np.expand_dims(self.kx, axis=0)
        self.kyh = np.expand_dims(self.kyh, axis=1)

        KX, KY = np.meshgrid(self.kx, self.kyh)
        self.ksq = KX**2 + KY**2

        # Defined in [0:Nky, -Nkx:Nkx]
        self.inv_ksq = 1. / (1. + self.ksq)

@dataclass
class Variables:
    fk: np.ndarray
    pk: np.ndarray

    def __init__(self, grid, init_val=0.001):
        #random_number = np.random.rand(*grid.inv_ksq.shape)
        random_number = 1.
        fk0 = init_val * grid.inv_ksq * np.exp(1j * 2. * np.pi * random_number)
        fk1 = - fk0 * grid.ksq

        self.fk = np.stack([fk0, fk1])
        self.pk = np.zeros_like(fk0)

class RKG4th:
    order: int = 4
    h: np.float64
    def __init__(self, h):
        self.y  = None
        self.k1 = None
        self.k2 = None
        self.k3 = None
        self.k4 = None
        self.h  = h

    def advance(self, f, y, step):
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

def realityCondition(A):
    def realityCondition2D(A_col):
        nky, nkx2 = A_col.shape
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

class HasegawaWakatani:
    ca: np.float64 = 3.
    nu: np.float64 = 0.01
    eta: np.float64 = 5.
    it: int = 0

    def __init__(self, nx, lx, nbiter, dt):
        self.grid = Grid(nx=nx, ny=nx, lx=lx, ly=lx)
        self.variables = Variables(grid=self.grid)
        self.ode = RKG4th(h = dt)
        self.nbiter = nbiter

        self.poisson_operator = -1. / self.grid.ksq
        self.poisson_operator[0,0] = 0.

        self.adiabacity_factor = self.grid.kyh**2

        self.variables.pk = self._poisson(self.variables.fk[1])
        self.variables.fk = realityCondition(self.variables.fk)
        self.variables.pk = realityCondition(self.variables.pk)

        

    def run(self):
        for _ in range(self.nbiter):
            self._diag()
            self._solve()

    def _diag(self):
        pass
    def _solve(self):
        for step in range(self.ode.order):
            vorticity = partial(self._vorticity, pk=self.variables.pk) 
            self.variables.fk = self.ode.advance(f=vorticity,
                                                 y=self.variables.fk,
                                                 step=step)
            self.variables.pk = self._poisson(fk=self.variables.fk[1])

            self.variables.fk = realityCondition(self.variables.fk)
            self.variables.pk = realityCondition(self.variables.pk)

    def _vorticity(self, fk, pk):
        phik = np.zeros_like(fk, dtype=np.complex128)
        dfkdt = np.zeros_like(fk, dtype=np.complex128)

        for i in range(2):
            phik[i] = self._poissonBracket(f=fk[i], g=pk)
            dfkdt[i] = - phik[i] - 1j * self.eta * self.grid.kyh * pk \
                       - self.ca * self.adiabacity_factor * (fk[i] - pk) \
                       - self.nu * fk[i] * self.grid.ksq**2

        return dfkdt

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

    def _forwardFFT(self, f):
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
        return self.poisson_operator * fk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('nx', nargs='?', type=int, default=1024)
    parser.add_argument('lx', nargs='?', type=float, default=10.0)
    parser.add_argument('nbiter', nargs='?', type=int, default=100)
    parser.add_argument('dt', nargs='?', type=float, default=0.005)
    args = parser.parse_args()

    model = HasegawaWakatani(**vars(args))
    start = time.time()
    model.run()
    seconds = time.time() - start
    
    print(f'Elapsed time: {seconds} [s]')
    