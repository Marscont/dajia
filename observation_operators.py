from __future__ import annotations

import numpy as np


def pressure_weights_from_boundaries(p_boundaries, p_surf=None, eps=1e-12):
    p = np.asarray(p_boundaries, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_boundaries must be a 1D array-like")
    n = p.size
    if n < 2:
        raise ValueError("need at least two boundaries")
    if p_surf is None:
        p_surf = p[-1]
    if p_surf <= 0:
        raise ValueError("p_surf must be positive")

    h = np.zeros_like(p)

    def safe_term_diff(pa, pb):
        if pa <= 0 or pb <= 0:
            return np.nan
        if abs(pb - pa) < 1e-6 * max(abs(pa), abs(pb), 1.0):
            return 0.5 * (pa + pb)
        denom = np.log(pb / pa)
        if abs(denom) < eps:
            return 0.5 * (pa + pb)
        return (pb - pa) / denom

    for i in range(n):
        left_term = 0.0
        right_term = 0.0
        if i + 1 < n:
            pa = p[i]
            pb = p[i + 1]
            right_term = -pa + safe_term_diff(pa, pb)
        if i - 1 >= 0:
            pa = p[i - 1]
            pb = p[i]
            left_term = pb - safe_term_diff(pa, pb)
        h[i] = abs(right_term + left_term) / float(p_surf)
    return h


def calculate_weights_for_profile(p_profile):
    p = np.asarray(p_profile, dtype=float)
    p_sorted = p[::-1] if p[0] > p[-1] else p
    p_surf = p_sorted[-1]
    return pressure_weights_from_boundaries(p_sorted, p_surf=p_surf)


def build_total_co2(co2_ant, co2_bck, co2_bio, system_bias):
    return co2_ant + co2_bck + co2_bio - float(system_bias)


def xco2_with_averaging_kernel(co2_profile_wrf, pressure_profile_wrf, row, levels=20):
    target_p = np.array([row[f"pressure_levels_{i}"] for i in range(levels)], dtype=float)
    apr = np.array([row[f"co2_profile_apriori_{i}"] for i in range(levels)], dtype=float)
    ak = np.array([row[f"xco2_avg_kernel_{i}"] for i in range(levels)], dtype=float)

    p = np.asarray(pressure_profile_wrf, dtype=float)
    c = np.asarray(co2_profile_wrf, dtype=float)
    if p[0] > p[-1]:
        p = p[::-1]
        c = c[::-1]

    interp = np.interp(target_p, p, c)
    w = pressure_weights_from_boundaries(target_p, p_surf=target_p[-1])
    return float(row["xco2_apriori"] + np.sum(w * ak * (interp - apr)))


def interp_surface_ppm_at_height(co2_profile_wrf, z_profile_m, target_height_m):
    z = np.asarray(z_profile_m, dtype=float)
    c = np.asarray(co2_profile_wrf, dtype=float)
    if z[0] > z[-1]:
        z = z[::-1]
        c = c[::-1]
    return float(np.interp(target_height_m, z, c))
