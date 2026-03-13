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

    @property
    def output_dir(self) -> Path:
        return Path(self.output_base_dir) / self.targettime / self.assim_domain


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
        )

    @staticmethod
    def init_output_dir(config: Config) -> Path:
        out = config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        return out
