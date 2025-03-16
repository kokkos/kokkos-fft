# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

""" 
A module for loading simulation data files and providing access to the data
for different variables and iterations.

"""

import argparse
import pathlib
from typing import Tuple, List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

class DataLoader:
    """
    A class to load simulation data files and providing access to the data
    for different variables and iterations. It supports both binary (.dat) and netCDF (.nc) formats.

    Attributes
    ----------
    data_dir : str
        Directory containing the data files.
    var_names : List[str]
        Names of the variables to be loaded.
    suffix : str
        File format suffix used to identify file type.
    shape : tuple
        The expected shape (ny, nx) of the simulation grid.
    nb_iters : int
        Number of iterations available in the data directory.

    Methods
    -------
    load(var_name: str, iter: int)
        Load and process data for a specific variable and iteration.
    mesh()
        Generate a meshgrid from the specified grid shape.
    """
    data_dir: str
    var_names: List[str]
    suffix: str
    shape: tuple
    nb_iters: int = 0
    def __init__(self, data_dir: str, var_names: List[str], suffix: str, shape: tuple) -> None:
        """
        Initializes the DataLoader.

        Parameters
        ----------
        data_dir : str
            Directory containing the data files.
        var_names : List[str]
            List of variable names to load.
        suffix : str
            File format suffix, must be either 'nc' or 'dat'.
        shape : tuple
            Expected grid shape (ny, nx).

        Raises
        ------
        ValueError
            If the provided suffix is not 'nc' or 'dat', or if the number of files available
            for each variable does not match (for binary data).
        FileNotFoundError
            If the expected data files are not found in the provided directory.
        """
        self.data_dir = data_dir
        self.var_names = var_names
        self.shape = shape

        # python data: .nc and kokkos data: .dat
        if suffix not in ['nc', 'dat']:
            raise ValueError(
                f'suffix {suffix} is not allowed. '
                f'suffix must be either nc or dat.'
            )
        self.suffix = suffix

        # How many iterations are saved in the output directory
        # For binary data
        if suffix == 'dat':
            nb_files = {}
            for var_name in var_names:
                file_paths = list(pathlib.Path(data_dir).glob(f'{var_name}*.{suffix}'))
                if not file_paths:
                    raise FileNotFoundError(
                        f'{var_name}_###.{suffix} files are not found under {data_dir}.'
                    )
                nb_files[var_name] = len(file_paths)
            nb_file_elems = list(nb_files.values())
            is_nb_file_identical = len(set(nb_file_elems)) == 1
            if is_nb_file_identical:
                self.nb_iters = nb_file_elems[0]
            else:
                raise ValueError(
                    f'Number of output files are not identical for each field: {nb_files}'
                )
        elif suffix == 'nc':
            file_paths = list(pathlib.Path(data_dir).glob(f'fields*.{suffix}'))
            if not file_paths:
                raise FileNotFoundError(
                    f'fields_###.{suffix} files are not found under {data_dir}.'
                )
            self.nb_iters = len(file_paths)

    def load(self, var_name: str, it: int) -> np.ndarray:
        """
        Load and process data for a specific variable and iteration.

        Parameters
        ----------
        var_name : str
            Name of the variable to load.
        it : int
            Iteration index to load.

        Returns
        -------
        np.ndarray
            Processed data array for the specified variable and iteration.

        Raises
        ------
        ValueError
            If the binary file does not have the expected shape.
        FileNotFoundError
            If the file for the specified iteration cannot be found.
        """
        def to_real_data(var):
            return backwardFFT(Real3DtoComplex2D(var), shape=self.shape)

        if self.suffix == 'dat':
            file_paths = sorted(pathlib.Path(self.data_dir).glob(f'{var_name}*.dat'))
            file_path = file_paths[it]
            ny, nx = self.shape
            nky, nkx = (ny - 2) // 3, (nx - 2) // 3
            nkyh, nkx2 = nky + 1, 2 * nkx + 1
            with open(file_path, 'rb') as f:
                var = np.fromfile(f, dtype=np.float64)
                if var.size != 2 * nkyh * nkx2:
                    raise ValueError(
                        f'The binary file does not have the expected shape '
                        f'(2, {nkyh}, {nkx2}). Current nx = {nx} does not fit '
                        f'in the data from the file {file_path}'
                    )
                var = var.reshape(2, nkyh, nkx2)
            return to_real_data(var)

        if self.suffix == 'nc':
            file_paths = sorted(pathlib.Path(self.data_dir).glob('fields*.nc'))
            file_path = file_paths[it]
            var = xr.open_dataset(file_path)[var_name].values
            return to_real_data(var)

        # Should never reach here because suffix is validated in __init__
        raise ValueError(f"Unsupported file format {self.suffix} in DataLoader.load")

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a meshgrid from the specified grid shape.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing two arrays representing the x and y coordinates.
        """
        ny, nx = self.shape
        x, y = np.arange(nx), np.arange(ny)
        return x, y

def Real3DtoComplex2D(A: np.ndarray) -> np.ndarray:
    """
    Convert a 3D real array to 2D complex array.

    Parameters
    ----------
    A : np.ndarray
        A 3D array where the first dimension contains the real and imaginary parts.

    Returns
    -------
    np.ndarray
        A 2D array of complex numbers.
    """

    real_part, imaginary_part = A[0], A[1]
    return real_part + 1j * imaginary_part

def backwardFFT(fk: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Compute the inverse 2D FFT to transform a complex Fourier representation 
    to a real spatial domain field.

    Parameters
    ----------
    fk : np.ndarray
        A 2D complex array of shape (nky+1, 2*nkx+1) representing the Fourier space.
    shape : tuple
        A tuple (ny, nx) representing the desired shape in real space.

    Returns
    -------
    np.ndarray
        A 2D real array of shape (ny, nx) corresponding to the inverse FFT of fk.
    """

    ny, nx = shape
    nkyh, nkx2 = fk.shape
    nky, nkx = nkyh - 1, (nkx2-1)//2

    fk_buffer = np.zeros((ny,nx//2+1), dtype=np.complex128)
    fk_buffer[:nky+1,0:nkx+1] = fk[:,0:nkx+1]

    for iy in range(1, nky+1):
        for ix in range(0, -(nkx+1), -1):
            fk_buffer[ny-iy, -ix] = np.conj(fk[iy,ix])

    f = np.fft.irfft2(fk_buffer)
    return f

def plot_fields(nx: int, data_dir: str, fig_dir: str, suffix: str,
                n_jobs: int, var_dict: dict) -> None:
    """
    Generate and save field plots for specified variables over multiple iterations.

    Parameters
    ----------
    nx : int
        Number of grid points in each spatial dimension (grid is nx by nx).
    data_dir : str
        Directory containing the simulation data files.
    fig_dir : str
        Directory where the figure files (PDFs) will be saved.
    suffix : str
        File format suffix, must be either 'nc' or 'dat'.
    n_jobs : int
        Number of parallel jobs to utilize for processing.
    var_dict : dict
        Dictionary with variable names as keys and maximum values (vmax) for the
        colormap as values.

    Raises
    ------
    FileNotFoundError
        If the specified data directory does not exist.
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    fig_path = pathlib.Path(fig_dir)
    if not fig_path.exists():
        fig_path.mkdir(parents=True)

    # Plot settings
    fontname = 'Times New Roman'
    fontsize = 28
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font',  family=fontname)

    axis_font = {'fontname':fontname, 'size':fontsize}
    title_font = {
        'fontname': fontname,
        'size': fontsize,
        'color': 'black',
        'verticalalignment': 'bottom'
    }

    var_names = list(var_dict.keys())
    data_loader = DataLoader(
        data_dir=data_dir, var_names=var_names, suffix=suffix,
        shape=(nx, nx)
    )

    nb_iters = data_loader.nb_iters
    x, y = data_loader.mesh()
    for var_name, vmax in var_dict.items():
        def save_fig(it: int, var_name=var_name, vmax_val=vmax) -> None:
            if it >= nb_iters:
                return
            var = data_loader.load(var_name=var_name, it=it)
            fig, ax = plt.subplots(figsize=(12,10), subplot_kw={'xticks':[], 'yticks':[]})
            pmesh = ax.pcolormesh(x, y, var, cmap='twilight',
                                  vmin=-vmax_val, vmax=vmax_val)
            ax.set_title(f'Time $t = {it:03d}$', **title_font)
            ax.set_xlabel(r'$x$', **axis_font)
            ax.set_ylabel(r'$y$', **axis_font)
            ax.set_aspect('equal')
            ax.axis('tight')
            fig.colorbar(pmesh, ax=ax)

            filename = fig_path / f'{var_name}_t{it:06d}.pdf'
            fig.savefig(filename, format='pdf', bbox_inches="tight")
            plt.close('all')

        Parallel(n_jobs=n_jobs)(delayed(save_fig)(it) for it in range(nb_iters))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-nx', nargs='?', type=int, default=1024,
                        help="Grid size in one dimension (total grid will be nx by nx)")
    parser.add_argument('-data_dir', nargs='?', type=str, default='data_kokkos',
                        help="Directory containing the simulation data files")
    parser.add_argument('-fig_dir', nargs='?', type=str, default='img_kokkos',
                        help="Directory where the generated plot PDFs will be saved")
    parser.add_argument('-suffix', nargs='?', type=str, default='dat',
                        help="File extension of the simulation data files ('nc' or 'dat')")
    parser.add_argument('-n_jobs', nargs='?', type=int, default=8,
                        help="Number of parallel jobs for generating plots")
    args = parser.parse_args()

    plot_fields(**vars(args), var_dict={'phi': 2e-05, 'density': 2e-05, 'vorticity': 2e-05})
