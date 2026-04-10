from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def _copy_template(src_template: Path, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_template, out_path)


def save_like_wrfout(
    wrfout_template: Path,
    output_path: Path,
    posterior_3d: np.ndarray,
    posterior_surface_2d: np.ndarray,
    var3d_name: str = "CO2_POST",
    var2d_name: str = "CO2_SURF_POST",
    time_index: int = -1,
):
    """
    按 wrfout 模板格式输出：复制模板后写入同维度变量，保持 wrfout 的维度/坐标/属性结构。
    - posterior_3d: shape (bottom_top, south_north, west_east)
    - posterior_surface_2d: shape (south_north, west_east)
    """
    _copy_template(Path(wrfout_template), Path(output_path))

    with Dataset(output_path, "r+") as ds:
        time_dim = "Time" if "Time" in ds.dimensions else None

        # 写 3D 变量（若不存在则创建）
        if var3d_name not in ds.variables:
            if time_dim is not None:
                v3d = ds.createVariable(var3d_name, "f4", ("Time", "bottom_top", "south_north", "west_east"))
            else:
                v3d = ds.createVariable(var3d_name, "f4", ("bottom_top", "south_north", "west_east"))
            v3d.description = "Posterior CO2 3D field written from EnKF"
            v3d.units = "ppm"
        else:
            v3d = ds.variables[var3d_name]

        if "Time" in v3d.dimensions:
            tidx = time_index if time_index >= 0 else (v3d.shape[0] - 1)
            v3d[tidx, :, :, :] = posterior_3d.astype(np.float32)
        else:
            v3d[:, :, :] = posterior_3d.astype(np.float32)

        # 写 2D 变量（若不存在则创建）
        if var2d_name not in ds.variables:
            if time_dim is not None:
                v2d = ds.createVariable(var2d_name, "f4", ("Time", "south_north", "west_east"))
            else:
                v2d = ds.createVariable(var2d_name, "f4", ("south_north", "west_east"))
            v2d.description = "Posterior surface CO2 field written from EnKF"
            v2d.units = "ppm"
        else:
            v2d = ds.variables[var2d_name]

        if "Time" in v2d.dimensions:
            tidx = time_index if time_index >= 0 else (v2d.shape[0] - 1)
            v2d[tidx, :, :] = posterior_surface_2d.astype(np.float32)
        else:
            v2d[:, :] = posterior_surface_2d.astype(np.float32)


def save_2d_nc(path: Path, var_name: str, data: np.ndarray):
    """兼容旧接口：简单二维输出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ny, nx = data.shape
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        v = ds.createVariable(var_name, "f4", ("south_north", "west_east"))
        v[:] = data.astype(np.float32)


def save_3d_nc(path: Path, var_name: str, data: np.ndarray):
    """兼容旧接口：简单三维输出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        nz, ny, nx = data.shape
        ds.createDimension("bottom_top", nz)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        v = ds.createVariable(var_name, "f4", ("bottom_top", "south_north", "west_east"))
        v[:] = data.astype(np.float32)
