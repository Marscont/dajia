from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config_init import Initializer
from data_loading import DataLoader
from enkf_assimilation import EnKF
from observation_operators import interp_surface_ppm_at_height, xco2_with_averaging_kernel
from save_concentrations import save_2d_nc, save_3d_nc
from save_emissions import ResultSaver as EmissionSaver
from spatial_matching import SpatialMatcher


def _build_surface_obs_and_hx(df, src_cfg, wrf_ds, wrf_grid_info, wrf_fields, config):
    matched = SpatialMatcher.match_obs_to_grid(df, src_cfg, wrf_ds, wrf_grid_info)
    if matched.empty:
        return matched, np.array([]), np.array([]), np.array([])

    vals, errs, hx = [], [], []
    vcol = src_cfg["columns"]["val"]
    ecol = src_cfg["columns"].get("err")
    for _, row in matched.iterrows():
        y, x = int(row["grid_y"]), int(row["grid_x"])
        prof = wrf_fields["total_co2"][:, y, x]
        z = wrf_fields["z_mass"][:, y, x]
        hx.append(interp_surface_ppm_at_height(prof, z, config.surface_interp_height_m))
        vals.append(float(row[vcol]))
        errs.append(float(row[ecol]) if ecol and ecol in row and not pd.isna(row[ecol]) else float(src_cfg["default_err"]))

    return matched, np.array(vals), np.array(errs) ** 2, np.array(hx)


def _build_xco2_obs_and_hx(df, src_cfg, wrf_ds, wrf_grid_info, wrf_fields, config):
    matched = SpatialMatcher.match_obs_to_grid(df, src_cfg, wrf_ds, wrf_grid_info)
    if matched.empty:
        return matched, np.array([]), np.array([]), np.array([])

    vals, errs, hx = [], [], []
    vcol = src_cfg["columns"]["val"]
    ecol = src_cfg["columns"].get("err")
    for _, row in matched.iterrows():
        y, x = int(row["grid_y"]), int(row["grid_x"])
        prof = wrf_fields["total_co2"][:, y, x]
        p = wrf_fields["pressure_hpa"][:, y, x]
        hx.append(xco2_with_averaging_kernel(prof, p, row, levels=config.xco2_levels))
        vals.append(float(row[vcol]))
        errs.append(float(row[ecol]) if ecol and ecol in row and not pd.isna(row[ecol]) else float(src_cfg["default_err"]))

    return matched, np.array(vals), np.array(errs) ** 2, np.array(hx)


def run_single_date(config_path: str):
    config = Initializer.load_config(config_path)
    out_dir = Initializer.init_output_dir(config)

    wrf_ds = DataLoader.load_wrf_dataset(config)
    wrf_grid_info = DataLoader.load_wrf_grid_info(config, wrf_ds)
    wrf_fields = DataLoader.load_wrf_fields(config, wrf_ds)

    obs = DataLoader.load_observation_data(config)

    state_mean = wrf_fields["total_co2"][0].reshape(-1)
    Xf = DataLoader.build_initial_ensemble(state_mean, config.num_members, config.initial_ensemble_std, config.min_emission)

    diagnostics = {"steps": []}

    # Step 1: surface assimilation
    surface_sources = [s for s in config.observation_sources if s["type"] == "surface"]
    for src in surface_sources:
        df = obs.get(src["name"], pd.DataFrame())
        mdf, y, r, hx = _build_surface_obs_and_hx(df, src, wrf_ds, wrf_grid_info, wrf_fields, config)
        if y.size == 0:
            continue
        Yf = np.tile(hx[:, None], (1, config.num_members))
        infl = EnKF.choose_inflation(config, Yf, y, r)
        Xf = EnKF.update(Xf, Yf, y, r, infl)

        innov = y - hx
        diagnostics["steps"].append({
            "source": src["name"],
            "type": "surface",
            "nobs": int(y.size),
            "inflation": float(infl),
            "bias": float(np.mean(innov)),
            "rmse": float(np.sqrt(np.mean(innov**2))),
        })
        mdf.assign(hx_surface=hx, innovation=innov).to_csv(out_dir / f"matched_{src['name']}.csv", index=False)

    # Step 2: satellite xco2 assimilation
    xco2_sources = [s for s in config.observation_sources if s["type"] == "xco2"]
    for src in xco2_sources:
        df = obs.get(src["name"], pd.DataFrame())
        mdf, y, r, hx = _build_xco2_obs_and_hx(df, src, wrf_ds, wrf_grid_info, wrf_fields, config)
        if y.size == 0:
            continue
        Yf = np.tile(hx[:, None], (1, config.num_members))
        infl = EnKF.choose_inflation(config, Yf, y, r)
        Xf = EnKF.update(Xf, Yf, y, r, infl)

        innov = y - hx
        diagnostics["steps"].append({
            "source": src["name"],
            "type": "xco2",
            "nobs": int(y.size),
            "inflation": float(infl),
            "bias": float(np.mean(innov)),
            "rmse": float(np.sqrt(np.mean(innov**2))),
        })
        mdf.assign(hx_xco2=hx, innovation=innov).to_csv(out_dir / f"matched_{src['name']}.csv", index=False)

    xa_mean_2d = np.mean(Xf, axis=1).reshape(wrf_grid_info["nlat"], wrf_grid_info["nlon"])
    prior3d = wrf_fields["total_co2"]
    post3d = prior3d.copy()
    post3d[0] = xa_mean_2d

    EmissionSaver.save_posterior_emissions(Xf, (wrf_grid_info["nlat"], wrf_grid_info["nlon"]), out_dir / "posterior_emission.nc")
    save_2d_nc(out_dir / "posterior_surface_co2.nc", "CO2_SURF_POST", xa_mean_2d)
    save_3d_nc(out_dir / "posterior_3d_co2.nc", "CO2_POST", post3d)

    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    wrf_ds.close()


if __name__ == "__main__":
    run_single_date("configure.yaml")
