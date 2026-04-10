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
    def discover_member_wrfouts(config: Config) -> List[Path]:
        paths: List[Path] = []
        for i in range(1, config.num_members + 1):
            mdir = config.wrf_member_dir(i)
            if not mdir.exists():
                continue
            candidates = sorted(mdir.glob(f"wrfout_{config.assim_domain}_*"))
            if candidates:
                paths.append(candidates[-1])
        return paths

    @staticmethod
    def load_wrf_dataset(config: Config) -> Dataset:
        path = config.wrfout_files.get(config.assim_domain)
        if path:
            return Dataset(path)

        member_wrfouts = DataLoader.discover_member_wrfouts(config)
        if not member_wrfouts:
            raise FileNotFoundError(
                f"Missing wrfout path for domain={config.assim_domain}, and no file found in {config.concentration_base_dir}/{config.concentration_member_prefix}xx"
            )
        return Dataset(str(member_wrfouts[0]))

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
    def _filter_by_target_hour(df: pd.DataFrame, targettime: str) -> pd.DataFrame:
        if "time" not in df.columns:
            return df
        target_dt = pd.to_datetime(targettime, format="%Y%m%d%H", errors="coerce")
        if pd.isna(target_dt):
            return df
        # 兼容 YYYY-MM-DD-HH / YYYY-MM-DD HH:MM:SS / 其他可解析格式
        obs_dt = pd.to_datetime(df["time"].astype(str).str.replace("-", "-", regex=False), errors="coerce")
        # 对于 YYYY-MM-DD-HH，pandas 有时不稳定，手动兜底
        fallback = pd.to_datetime(df["time"].astype(str), format="%Y-%m-%d-%H", errors="coerce")
        obs_dt = obs_dt.fillna(fallback)
        m = (obs_dt.dt.year == target_dt.year) & (obs_dt.dt.month == target_dt.month) & (obs_dt.dt.day == target_dt.day) & (obs_dt.dt.hour == target_dt.hour)
        return df[m].copy()

    @staticmethod
    def load_observation_data(config: Config) -> Dict[str, pd.DataFrame]:
        out = {}
        sat_day_dir = config.sat_day_dir()
        for src in config.observation_sources:
            file_path = sat_day_dir / src["csv_file"]
            if not file_path.exists():
                file_path = Path(config.obs_root_dir) / src["csv_file"]
            if not file_path.exists():
                out[src["name"]] = pd.DataFrame()
                continue

            df = pd.read_csv(file_path)
            df = DataLoader._filter_by_target_hour(df, config.targettime)

            vcol = src["columns"]["val"]
            if vcol not in df.columns:
                out[src["name"]] = pd.DataFrame()
                continue

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
