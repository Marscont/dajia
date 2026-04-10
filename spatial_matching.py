from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from wrf import ll_to_xy


class SpatialMatcher:
    @staticmethod
    def match_obs_to_grid(df: pd.DataFrame, src_cfg: Dict, wrf_ds, wrf_grid_info: Dict) -> pd.DataFrame:
        if df.empty:
            return df
        lat_col = src_cfg["columns"]["lat"]
        lon_col = src_cfg["columns"]["lon"]

        coords = ll_to_xy(wrf_ds, df[lat_col].values, df[lon_col].values)
        out = df.copy()
        out["grid_y"] = np.asarray(coords[0]).astype(int)
        out["grid_x"] = np.asarray(coords[1]).astype(int)

        ny, nx = wrf_grid_info["nlat"], wrf_grid_info["nlon"]
        m = (out["grid_y"] >= 0) & (out["grid_y"] < ny) & (out["grid_x"] >= 0) & (out["grid_x"] < nx)
        return out[m].reset_index(drop=True)
