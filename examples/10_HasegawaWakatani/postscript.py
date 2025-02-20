# SPDX-FileCopyrightText: (C) The kokkos-fft development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import List
from joblib import Parallel, delayed

class DataLoader:
    """
    A class to load simulation data files and providing access to the data
    for different variables and iterations. It supports both binary (.dat) and netCDF (.nc) formats.

    Attributes
    ----------
    data_dir (str): Directory containing the data files.
    var_names (List[str]): List of variable names to be loaded.
    suffix (str): File format suffix, must be either 'nc' or 'dat'.
    shape (tuple): Expected shape (ny, nx) of the simulation grid.
    nb_iters (int): Number of iterations available in the data directory.

    Raises:
    ValueError: If an unsupported suffix is provided or if the number of files for each field
                does not match for binary data.
    FileNotFoundError: If the expected data files are not found in the provided directory.
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
        data_dir (str): Directory containing the data files.
        var_names (List[str]): List of variable names to load.
        suffix (str): File format suffix, must be either 'nc' or 'dat'.
        shape (tuple): Expected grid shape (ny, nx).

        Methods
        -------
        load(var_name: str, iter: int) -> np.ndarray:
            Load and process data for a specific variable and iteration.
        mesh() -> List[np.ndarray]:
            Generate a meshgrid from the specified grid shape.

        Raises
        ------
        ValueError: If the provided suffix is not 'nc' or 'dat'.
        FileNotFoundError: If no files corresponding to the provided variable names or fields are found.
        """
        self.data_dir = data_dir
        self.var_names = var_names
        self.shape = shape
        
        # python data: .nc and kokkos data: .dat
        if not suffix in ['nc', 'dat']:
            raise ValueError(f'suffix {suffix} is not allowed. suffix must be either nc or dat.')
        self.suffix = suffix
        
        # How many iterations are saved in the output directory
        # For binary data
        if suffix == 'dat':
            nb_files = {}
            for var_name in var_names:
                file_paths = list(pathlib.Path(data_dir).glob(f'{var_name}*.{suffix}'))
                if not file_paths:
                    raise FileNotFoundError(f'{var_name}_###.{suffix} files are not found under {data_dir}.')
                nb_files[var_name] = len(file_paths)
            nb_file_elems = list(nb_files.values())
            is_nb_file_identical = len(set(nb_file_elems)) == 1
            if is_nb_file_identical:
                self.nb_iters = nb_file_elems[0]
            else:
                raise ValueError(f'Number of output files are not identical for each field: {nb_files}')
        elif suffix == 'nc':
            file_paths = list(pathlib.Path(data_dir).glob(f'fields*.{suffix}'))
            if not file_paths:
                raise FileNotFoundError(f'fields_###.{suffix} files are not found under {data_dir}.')
            self.nb_iters = len(file_paths)
        
    def load(self, var_name: str, iter: int) -> np.ndarray:
        """
        Load and process data for a specific variable and iteration.

        Parameters
        ----------
        var_name (str): The name of the variable to load.
        iter (int): The iteration index to load.

        Returns
        -------
        np.ndarray: Processed data array for the specified variable and iteration.

        Raises
        ------
        ValueError: If the binary file does not have the expected shape.
        FileNotFoundError: If the file for the given iteration is not found.
        """
        to_real_data = lambda var: backwardFFT( Real3DtoComplex2D(A = var), shape=self.shape )
        
        if self.suffix == 'dat':
            file_paths = sorted(pathlib.Path(self.data_dir).glob(f'{var_name}*.dat'))
            file_path = file_paths[iter]
            ny, nx = self.shape
            nky, nkx = (ny - 2) // 3, (nx - 2) // 3
            nkyh, nkx2 = nky + 1, 2 * nkx + 1
            with open(file_path) as f:
                var = np.fromfile(f, dtype=np.float64)
                if var.size != 2 * nkyh * nkx2:
                    raise ValueError(f"""The binary file does not have the expected shape (2, {nkyh}, {nkx2}).
                                     Current nx = {nx} does not fit in the data from the file {file_path}""")
                
                var = var.reshape(2, nkyh, nkx2)
            return to_real_data(var)
        elif self.suffix == 'nc':
            file_paths = sorted(pathlib.Path(self.data_dir).glob('fields*.nc'))
            file_path = file_paths[iter]
            var = xr.open_dataset(file_path)[var_name].values
            return to_real_data(var)
            
    def mesh(self) -> List[np.ndarray]:
        """
        Generate a meshgrid from the specified grid shape.

        Returns
        -------
        List[np.ndarray]: Two arrays representing the x and y coordinates of the grid.
        """
        ny, nx = self.shape
        x, y = np.arange(nx), np.arange(ny)
        return x, y

def Real3DtoComplex2D(A: np.ndarray) -> np.ndarray:
    """
    Convert a 3D real array to 2D complex array.

    Parameters
    ----------
    A (xr.DataArray): A 3D array where the first dimension represents the real and imaginary parts.

    Returns
    -------
    np.ndarray: A 2D array of complex numbers.
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
        A 2D complex array of shape (nky+1, 2*nkx+1) representing the Fourier 
        space representation of a field.
    shape : tuple
        A tuple (ny, nx) representing the desired shape in the real space.

    Returns
    -------
    np.ndarray
        A 2D real array of shape (ny, nx) corresponding to real space representation of fk.
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

    Parameters:
    nx (int): The number of grid points in each dimension.
    data_dir (str): Directory containing the data files.
    fig_dir (str): Directory where the figure files will be saved.
    suffix (str): File format suffix, must be either 'nc' or 'dat'.
    n_jobs (int): Number of parallel jobs for processing.
    var_dict (dict): Dictionary with variable names as keys and corresponding maximum values for the colormap (vmax) as values.

    Raises:
    FileNotFoundError: If the specified data directory does not exist.
    """
    data_path = pathlib.Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")
    
    var_names = list(var_dict.keys())
    data_loader = DataLoader(data_dir=data_dir, var_names=var_names, suffix=suffix,
                             shape=(nx, nx))
    
    nb_iters = data_loader.nb_iters
    x, y = data_loader.mesh()
    for var_name, vmax in var_dict.items():
        fig_path = pathlib.Path(fig_dir)
        if not fig_path.exists():
            fig_path.mkdir(parents=True)

        def save_fig(iter: int) -> None:
            if iter >= nb_iters:
                return
            var = data_loader.load(var_name=var_name, iter=iter)
            fig, ax = plt.subplots(figsize=(12,10))
            pmesh = ax.pcolormesh(x, y, var, cmap='twilight', vmin=-vmax, vmax=vmax)
            ax.set_title(r'Time $t = {:03d}$'.format(iter), **title_font)
            ax.set_xlabel(r'$x$', **axis_font)
            ax.set_ylabel(r'$y$', **axis_font)
            ax.set_aspect('equal')
            ax.axis('tight')
            fig.colorbar(pmesh, ax=ax)
            
            filename = fig_path / f'{var_name}_t{iter:06d}.pdf'
            fig.savefig(filename, format='pdf', bbox_inches="tight")
            plt.close('all')
        
        Parallel(n_jobs=n_jobs)(delayed(save_fig)(iter) for iter in range(nb_iters))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-nx', nargs='?', type=int, default=1024)
    parser.add_argument('-data_dir', nargs='?', type=str, default='data_kokkos')
    parser.add_argument('-fig_dir', nargs='?', type=str, default='img_kokkos')
    parser.add_argument('-suffix', nargs='?', type=str, default='dat')
    parser.add_argument('-n_jobs', nargs='?', type=int, default=8)
    args = parser.parse_args()

    var_dict = dict(phi= 0.0005, density=0.0005, vorticity=0.0005)
    
    # Plot settings
    fontname = 'Times New Roman'
    fontsize = 28
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    plt.rc('font',  family=fontname)
     
    axis_font = {'fontname':fontname, 'size':fontsize}
    title_font = {'fontname':fontname, 'size':fontsize, 'color':'black',
                  'verticalalignment':'bottom'}
    plot_fields(**vars(args), var_dict=var_dict)
