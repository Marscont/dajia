from __future__ import annotations
from pathlib import Path

import copy
import json

import numpy as np
import pandas as pd

from config_init import Initializer
from data_loading import DataLoader
from enkf_assimilation import EnKF
from observation_operators import interp_surface_ppm_at_height, xco2_with_averaging_kernel
from save_concentrations import save_like_wrfout
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


def _apply_conc_system_error(config, y, hx):
    if y.size == 0:
        return hx, 0.0
    if isinstance(config.conc_system_error, str) and config.conc_system_error.lower() == "auto":
        bias = float(np.mean(y) - np.mean(hx))
        return hx + bias, bias
    try:
        bias = float(config.conc_system_error)
    except Exception:
        bias = 0.0
    return hx + bias, bias


def run_single_date(config):
    out_dir = Initializer.init_output_dir(config)

    wrf_ds = DataLoader.load_wrf_dataset(config)
    wrf_grid_info = DataLoader.load_wrf_grid_info(config, wrf_ds)
    wrf_fields = DataLoader.load_wrf_fields(config, wrf_ds)

    obs = DataLoader.load_observation_data(config)

    state_mean = wrf_fields["total_co2"][config.conc_level].reshape(-1)
    Xf = DataLoader.build_initial_ensemble(state_mean, config.num_members, config.initial_ensemble_std, config.min_emission)

    diagnostics = {"targettime": config.targettime, "steps": []}

    # Step 1: surface assimilation
    surface_sources = [s for s in config.observation_sources if s["type"] == "surface"]
    for src in surface_sources:
        df = obs.get(src["name"], pd.DataFrame())
        mdf, y, r, hx = _build_surface_obs_and_hx(df, src, wrf_ds, wrf_grid_info, wrf_fields, config)
        if y.size == 0:
            continue

        hx_corr, sys_bias = _apply_conc_system_error(config, y, hx)
        Yf = np.tile(hx_corr[:, None], (1, config.num_members))
        infl = EnKF.choose_inflation(config, Yf, y, r)
        Xf = EnKF.update(Xf, Yf, y, r, infl)

        innov = y - hx_corr
        diagnostics["steps"].append(
            {
                "source": src["name"],
                "type": "surface",
                "nobs": int(y.size),
                "inflation": float(infl),
                "conc_system_error": float(sys_bias),
                "bias": float(np.mean(innov)),
                "rmse": float(np.sqrt(np.mean(innov**2))),
            }
        )
        mdf.assign(hx_surface=hx_corr, innovation=innov).to_csv(out_dir / f"matched_{src['name']}.csv", index=False)

    # Step 2: satellite xco2 assimilation
    xco2_sources = [s for s in config.observation_sources if s["type"] == "xco2"]
    for src in xco2_sources:
        df = obs.get(src["name"], pd.DataFrame())
        mdf, y, r, hx = _build_xco2_obs_and_hx(df, src, wrf_ds, wrf_grid_info, wrf_fields, config)
        if y.size == 0:
            continue

        hx_corr, sys_bias = _apply_conc_system_error(config, y, hx)
        Yf = np.tile(hx_corr[:, None], (1, config.num_members))
        infl = EnKF.choose_inflation(config, Yf, y, r)
        Xf = EnKF.update(Xf, Yf, y, r, infl)

        innov = y - hx_corr
        diagnostics["steps"].append(
            {
                "source": src["name"],
                "type": "xco2",
                "nobs": int(y.size),
                "inflation": float(infl),
                "conc_system_error": float(sys_bias),
                "bias": float(np.mean(innov)),
                "rmse": float(np.sqrt(np.mean(innov**2))),
            }
        )
        mdf.assign(hx_xco2=hx_corr, innovation=innov).to_csv(out_dir / f"matched_{src['name']}.csv", index=False)

    xa_mean_2d = np.mean(Xf, axis=1).reshape(wrf_grid_info["nlat"], wrf_grid_info["nlon"])
    prior3d = wrf_fields["total_co2"]
    post3d = prior3d.copy()
    post3d[config.conc_level] = xa_mean_2d

    wrfout_template = Path(wrf_ds.filepath())

    # wrfout格式浓度输出（复制模板后写新变量）
    save_like_wrfout(
        wrfout_template=wrfout_template,
        output_path=out_dir / "posterior_wrfout_like.nc",
        posterior_3d=post3d,
        posterior_surface_2d=xa_mean_2d,
        var3d_name="CO2_POST",
        var2d_name="CO2_SURF_POST",
        time_index=-1,
    )

    # wrfchemi格式排放输出（复制EMISS模板后覆写排放变量）
    wrfchemi_tpl = None
    emdir = config.emiss_member_dir(1)
    if emdir.exists():
        cands = sorted(emdir.glob("wrfchemi*"))
        if cands:
            wrfchemi_tpl = cands[-1]

    if wrfchemi_tpl is not None:
        EmissionSaver.save_posterior_emissions_like_wrfchemi(
            Xa=Xf,
            shape_2d=(wrf_grid_info["nlat"], wrf_grid_info["nlon"]),
            output_path=out_dir / "posterior_wrfchemi_like.nc",
            wrfchemi_template_path=wrfchemi_tpl,
            emission_var_name=None,
            time_index=-1,
        )
    else:
        # fallback: 若模板不存在，保留简化输出避免流程中断
        EmissionSaver.save_posterior_emissions(Xf, (wrf_grid_info["nlat"], wrf_grid_info["nlon"]), out_dir / "posterior_emission_fallback.nc")

    with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    wrf_ds.close()


def main():
    base_config = Initializer.load_config("configure.yaml")
    for target in Initializer.iter_target_days(base_config):
        cfg = copy.deepcopy(base_config)
        cfg.targettime = target
        print(f"===== 开始同化 {target} / {cfg.assim_domain} =====")
        run_single_date(cfg)
        print(f"===== 完成 {target} =====")


if __name__ == "__main__":
    main()
