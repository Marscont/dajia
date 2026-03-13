from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from config_init import Config
from observation_operators import build_total_co2


class DataLoader:
    @staticmethod
    def load_wrf_dataset(config: Config) -> Dataset:
        path = config.wrfout_files.get(config.assim_domain)
        if not path:
            raise FileNotFoundError(f"Missing wrfout path for domain={config.assim_domain}")
        return Dataset(path)

    @staticmethod
    def load_wrf_grid_info(config: Config, ds: Dataset) -> Dict:
        t = -1 if config.time_index is None else config.time_index
        xlat = ds.variables["XLAT"][t]
        xlon = ds.variables["XLONG"][t]
        return {
            "lat2d": np.array(xlat),
            "lon2d": np.array(xlon),
            "nlat": xlat.shape[0],
            "nlon": xlat.shape[1],
            "resolution_km": config.domain_resolution_km[config.assim_domain],
        }

    @staticmethod
    def load_wrf_fields(config: Config, ds: Dataset) -> Dict[str, np.ndarray]:
        t = -1 if config.time_index is None else config.time_index
        co2_ant = np.array(ds.variables["CO2_ANT"][t])
        co2_bck = np.array(ds.variables["CO2_BCK"][t])
        co2_bio = np.array(ds.variables["CO2_BIO"][t])
        p = np.array(ds.variables["P"][t])
        pb = np.array(ds.variables["PB"][t])
        ph = np.array(ds.variables["PH"][t])
        phb = np.array(ds.variables["PHB"][t])

        total = build_total_co2(co2_ant, co2_bck, co2_bio, config.co2_system_bias)
        pressure_hpa = (p + pb) / 100.0

        g = 9.81
        z_w = (ph + phb) / g
        z_mass = 0.5 * (z_w[:-1] + z_w[1:])

        return {
            "total_co2": total,
            "pressure_hpa": pressure_hpa,
            "z_mass": z_mass,
        }

    @staticmethod
    def load_observation_data(config: Config) -> Dict[str, pd.DataFrame]:
        out = {}
        for src in config.observation_sources:
            file_path = Path(config.obs_root_dir) / src["csv_file"]
            if not file_path.exists():
                continue
            df = pd.read_csv(file_path)
            if "time" in df.columns and config.targettime:
                dt = f"{config.targettime[:4]}-{config.targettime[4:6]}-{config.targettime[6:8]}-{config.targettime[8:10] if len(config.targettime)>=10 else '00'}"
                df = df[df["time"].astype(str) == dt].copy()
            vcol = src["columns"]["val"]
            vmin, vmax = src.get("val_range", [-np.inf, np.inf])
            df = df[(df[vcol] >= vmin) & (df[vcol] <= vmax)].copy()
            if not df.empty:
                mean = df[vcol].mean()
                std = df[vcol].std(ddof=0)
                if std > 0:
                    m = (df[vcol] >= mean - config.sigma_threshold * std) & (df[vcol] <= mean + config.sigma_threshold * std)
                    df = df[m].copy()
            out[src["name"]] = df
        return out

    @staticmethod
    def build_initial_ensemble(state_mean: np.ndarray, num_members: int, std: float, floor: float = 1e-6) -> np.ndarray:
        rng = np.random.default_rng(42)
        pert = rng.normal(0.0, std, size=(state_mean.size, num_members))
        ens = state_mean[:, None] + pert
        ens[ens < floor] = floor
        return ens
