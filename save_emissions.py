from __future__ import annotations

from pathlib import Path

import numpy as np
from netCDF4 import Dataset


class ResultSaver:
    @staticmethod
    def save_posterior_emissions(Xa: np.ndarray, shape_2d, output_path: Path):
        data = np.mean(Xa, axis=1).reshape(shape_2d)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(output_path, "w") as ds:
            ny, nx = shape_2d
            ds.createDimension("south_north", ny)
            ds.createDimension("west_east", nx)
            v = ds.createVariable("E_POST", "f4", ("south_north", "west_east"))
            v[:] = data.astype(np.float32)
