# SPDX-FileCopyrightText: (C) The Kokkos-FFT development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception

"""
Animate a sweep through the Poisson solution computed by PoissonEquation.cpp.

The C++ example writes the callback-computed solution with --dump as a
binary file laid out as

    [int64 Nx, Ny, Nz][float64 Lx, Ly, Lz][Nx*Ny*Nz float64 phi]  (k-fastest)

This script reads that file and renders an animated GIF sweeping through the
volume along one axis (default: z), one 2D slice per frame.

Usage:
    python postscript.py <file.bin> [--axis z] [--fps 12] [--output out.gif]
"""

import argparse
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_dump(path):
    """Read a --dump file, return (shape, lengths, phi[Nx, Ny, Nz])."""
    raw = np.fromfile(path, dtype=np.int64, count=3)
    shape = tuple(int(v) for v in raw)
    lens = np.fromfile(path, dtype=np.float64, count=3, offset=3 * 8)
    n = shape[0] * shape[1] * shape[2]
    phi = np.fromfile(path, dtype=np.float64, count=n, offset=6 * 8)
    return shape, tuple(float(v) for v in lens), phi.reshape(shape)


def make_gif(shape, lens, phi, axis, output, fps):
    """Animate a sweep along one axis, one 2D slice per frame."""
    axis = {"x": 0, "y": 1, "z": 2}[axis]
    n = shape[axis]
    coords = [np.linspace(0.0, lens[a], shape[a]) for a in range(3)]
    fixed = "xyz"[axis]
    vmin, vmax = float(phi.min()), float(phi.max())

    frames = []
    for i in range(n):
        fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        data = np.take(phi, i, axis=axis)
        others = [a for a in range(3) if a != axis]
        pcm = ax.pcolormesh(coords[others[0]], coords[others[1]], data.T,
                            shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=r"$\phi$", format="%.1e")
        ax.set_xlabel("xyz"[others[0]])
        ax.set_ylabel("xyz"[others[1]])
        ax.set_title(rf"Poisson solution $\phi$ at ${fixed} = {coords[axis][i]:.3g}$"
                     "\ncomputed with KokkosFFT zero-padding callbacks")
        ax.set_aspect("equal")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))

    duration = max(1, int(round(1000.0 / fps)))
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    print(f"Wrote {output} ({n} frames along {fixed}, {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="binary dump written by --dump")
    parser.add_argument(
        "--axis", choices=["x", "y", "z"], default="z",
        help="axis to sweep along (default: z)",
    )
    parser.add_argument(
        "--output", default="poisson_callback.gif",
        help="output GIF path (default: %(default)s)",
    )
    parser.add_argument(
        "--fps", type=int, default=12, help="GIF frames per second (default: 12)"
    )
    args = parser.parse_args()

    shape, lens, phi = load_dump(args.input)
    make_gif(shape, lens, phi, args.axis, args.output, args.fps)


if __name__ == "__main__":
    main()
