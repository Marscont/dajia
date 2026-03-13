from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def save_2d_nc(path: Path, var_name: str, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ny, nx = data.shape
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        v = ds.createVariable(var_name, "f4", ("south_north", "west_east"))
        v[:] = data.astype(np.float32)


def save_3d_nc(path: Path, var_name: str, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        nz, ny, nx = data.shape
        ds.createDimension("bottom_top", nz)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        v = ds.createVariable(var_name, "f4", ("bottom_top", "south_north", "west_east"))
        v[:] = data.astype(np.float32)
