from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Config:
    time_index: Optional[int]
    targettime: str

    assim_domain: str
    domain_resolution_km: Dict[str, float]

    wrfout_files: Dict[str, str]
    output_base_dir: str
    obs_root_dir: str
    emissions_base_dir: str
    concentration_base_dir: str
    observation_sources: List[Dict[str, Any]]

    num_members: int
    sigma_threshold: float
    co2_system_bias: float

    matching_method: str
    surface_interp_height_m: float
    xco2_levels: int

    inflation_mode: str
    inflation_factor: float
    adaptive_inflation_min: float
    adaptive_inflation_max: float
    localization_radius_grid_base: float

    initial_ensemble_std: float

    min_emission: float = 1e-4
    emissions_member_prefix: str = "e"
    concentration_member_prefix: str = "t"

    @property
    def day(self) -> str:
        return self.targettime[:8]

    @property
    def output_dir(self) -> Path:
        return Path(self.output_base_dir) / self.day / self.assim_domain

    def sat_day_dir(self) -> Path:
        return Path(self.obs_root_dir) / self.day

    def emiss_member_dir(self, idx_1based: int) -> Path:
        return Path(self.emissions_base_dir) / f"{self.emissions_member_prefix}{idx_1based:02d}"

    def wrf_member_dir(self, idx_1based: int) -> Path:
        return Path(self.concentration_base_dir) / f"{self.concentration_member_prefix}{idx_1based:02d}"


class Initializer:
    @staticmethod
    def load_config(config_path: str) -> Config:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return Config(
            time_index=raw.get("time_index"),
            targettime=str(raw.get("targettime", "")),
            assim_domain=raw.get("assim_domain", "d02"),
            domain_resolution_km=raw.get("domain_resolution_km", {"d01": 27, "d02": 9, "d03": 3}),
            wrfout_files=raw.get("wrfout_files", {}),
            output_base_dir=raw["output_base_dir"],
            obs_root_dir=raw["obs_root_dir"],
            emissions_base_dir=raw.get("emissions_base_dir", "./EMISS"),
            concentration_base_dir=raw.get("concentration_base_dir", "./run_wrf"),
            observation_sources=raw.get("observation_sources", []),
            num_members=int(raw.get("num_members", 32)),
            sigma_threshold=float(raw.get("sigma_threshold", 3.0)),
            co2_system_bias=float(raw.get("co2_system_bias", 410.0)),
            matching_method=raw.get("matching_method", "nearest"),
            surface_interp_height_m=float(raw.get("surface_interp_height_m", 50.0)),
            xco2_levels=int(raw.get("xco2_levels", 20)),
            inflation_mode=raw.get("inflation_mode", "adaptive"),
            inflation_factor=float(raw.get("inflation_factor", 1.1)),
            adaptive_inflation_min=float(raw.get("adaptive_inflation_min", 1.0)),
            adaptive_inflation_max=float(raw.get("adaptive_inflation_max", 1.3)),
            localization_radius_grid_base=float(raw.get("localization_radius_grid_base", 8.0)),
            initial_ensemble_std=float(raw.get("initial_ensemble_std", 0.8)),
            min_emission=float(raw.get("min_emission", 1e-4)),
            emissions_member_prefix=str(raw.get("emissions_member_prefix", "e")),
            concentration_member_prefix=str(raw.get("concentration_member_prefix", "t")),
        )

    @staticmethod
    def init_output_dir(config: Config) -> Path:
        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
