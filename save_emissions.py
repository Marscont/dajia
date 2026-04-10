from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


class ResultSaver:
    @staticmethod
    def _guess_wrfchemi_var(ds: Dataset) -> str:
        # 常见排放变量名，按优先级查找
        candidates = [
            "E_CO2",
            "E_CO2_ANT",
            "CO2_ANT",
            "CO2_BIO",
            "CO2_BCK",
        ]
        for name in candidates:
            if name in ds.variables:
                return name
        # fallback: 选择第一个具备 south_north/west_east 维度的变量
        for name, var in ds.variables.items():
            dims = set(var.dimensions)
            if "south_north" in dims and "west_east" in dims:
                return name
        raise KeyError("No suitable emission variable found in wrfchemi template")

    @staticmethod
    def save_posterior_emissions_like_wrfchemi(
        Xa: np.ndarray,
        shape_2d,
        output_path: Path,
        wrfchemi_template_path: Path,
        emission_var_name: str | None = None,
        time_index: int = -1,
    ):
        """按 wrfchemi 模板写后验排放：复制模板，覆写指定排放变量。"""
        data2d = np.mean(Xa, axis=1).reshape(shape_2d).astype(np.float32)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wrfchemi_template_path, output_path)

        with Dataset(output_path, "r+") as ds:
            vname = emission_var_name or ResultSaver._guess_wrfchemi_var(ds)
            var = ds.variables[vname]

            # 常见维度: (Time, emissions_zdim, south_north, west_east) 或 (Time, south_north, west_east)
            dims = var.dimensions
            if len(dims) == 4:
                tidx = time_index if time_index >= 0 else (var.shape[0] - 1)
                zdim = var.shape[1]
                for k in range(zdim):
                    var[tidx, k, :, :] = data2d
            elif len(dims) == 3:
                tidx = time_index if time_index >= 0 else (var.shape[0] - 1)
                var[tidx, :, :] = data2d
            elif len(dims) == 2:
                var[:, :] = data2d
            else:
                raise ValueError(f"Unsupported emission variable dims: {dims}")

    @staticmethod
    def save_posterior_emissions(Xa: np.ndarray, shape_2d, output_path: Path):
        """兼容旧接口：简单2D netcdf。"""
        data = np.mean(Xa, axis=1).reshape(shape_2d)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(output_path, "w") as ds:
            ny, nx = shape_2d
            ds.createDimension("south_north", ny)
            ds.createDimension("west_east", nx)
            v = ds.createVariable("E_POST", "f4", ("south_north", "west_east"))
            v[:] = data.astype(np.float32)
