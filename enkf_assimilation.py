from __future__ import annotations

import numpy as np


def _adaptive_inflation_factor(yf: np.ndarray, y: np.ndarray, r: np.ndarray, low: float, high: float) -> float:
    var_yf = float(np.nanvar(yf, ddof=1)) if yf.size > 1 else 0.0
    innov = y - np.nanmean(yf, axis=1)
    innov_var = float(np.nanmean(innov**2)) if innov.size else 0.0
    r_mean = float(np.nanmean(r)) if r.size else 0.0
    denom = max(var_yf, 1e-6)
    lam = np.sqrt(max((innov_var - r_mean) / denom, 1.0))
    return float(np.clip(lam, low, high))


class EnKF:
    @staticmethod
    def update(Xf: np.ndarray, Yf: np.ndarray, y: np.ndarray, r: np.ndarray, inflation: float) -> np.ndarray:
        nstate, nens = Xf.shape
        nobs = y.size

        x_mean = np.mean(Xf, axis=1, keepdims=True)
        y_mean = np.mean(Yf, axis=1, keepdims=True)
        Xa_pert = (Xf - x_mean) * inflation
        Ya_pert = (Yf - y_mean) * inflation

        pxy = Xa_pert @ Ya_pert.T / (nens - 1)
        pyy = Ya_pert @ Ya_pert.T / (nens - 1)
        rmat = np.diag(r)
        k = pxy @ np.linalg.inv(pyy + rmat)

        rng = np.random.default_rng(2024)
        eps = rng.normal(0.0, np.sqrt(r)[:, None], size=(nobs, nens))
        y_pert = y[:, None] + eps

        Xa = Xf + k @ (y_pert - Yf)
        return Xa

    @staticmethod
    def choose_inflation(config, Yf: np.ndarray, y: np.ndarray, r: np.ndarray) -> float:
        if config.inflation_mode == "adaptive":
            return _adaptive_inflation_factor(
                Yf=Yf,
                y=y,
                r=r,
                low=config.adaptive_inflation_min,
                high=config.adaptive_inflation_max,
            )
        return config.inflation_factor
