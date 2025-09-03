# -*- coding: utf-8 -*-
"""
Feature engineering and optimal feature selection for time-series windows.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-9

# --------------------------------------------------------------------------- #
# Utilities and feature extraction
# --------------------------------------------------------------------------- #

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default on invalid inputs."""
    if denominator == 0 or pd.isna(numerator) or pd.isna(denominator):
        return default
    return numerator / denominator


def _compute_series_stats(series: pd.Series, prefix: str) -> Dict[str, float]:
    """Common stats with prefixed keys."""
    if series is None or series.empty:
        return {
            f"{prefix}_mean": 0.0, f"{prefix}_sum": 0.0, f"{prefix}_std": 0.0,
            f"{prefix}_var": 0.0, f"{prefix}_max": 0.0, f"{prefix}_min": 0.0,
        }
    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_sum": float(series.sum()),
        f"{prefix}_std": float(series.std()),
        f"{prefix}_var": float(series.var()),
        f"{prefix}_max": float(series.max()),
        f"{prefix}_min": float(series.min()),
    }


def extract_all_features(window_df: pd.DataFrame, start_row: pd.Series) -> Dict[str, Any]:
    """Extract a comprehensive set of window features."""
    if window_df is None or window_df.empty:
        return {}

    features: Dict[str, Any] = {}
    cols = window_df.columns

    # Discharging subset if Current available (project convention may differ)
    discharging_df = window_df[window_df["Current"] > 0] if "Current" in cols else pd.DataFrame()

    # Time features
    if "timestamp" in cols and pd.notna(start_row.get("timestamp")):
        t_end = window_df["timestamp"].iloc[-1]
        t_start = start_row["timestamp"]
        if pd.notna(t_end) and pd.notna(t_start):
            features["delta_time"] = (t_end - t_start).total_seconds()

    # Current and power
    if "Current" in cols:
        features.update(_compute_series_stats(discharging_df["Current"] if not discharging_df.empty else window_df["Current"], "current"))
        features["current_abs_mean"] = float(window_df["Current"].abs().mean())
        if "PackV" in cols and not discharging_df.empty:
            power = discharging_df["PackV"] * discharging_df["Current"]
            features["power_max"] = float(power.max())
            features["power_mean"] = float(power.mean())

    # Pack voltage
    if "PackV" in cols:
        pv = discharging_df["PackV"] if not discharging_df.empty else window_df["PackV"]
        features.update(_compute_series_stats(pv, "voltage"))
        features["start_voltage"] = float(start_row.get("PackV", 0.0) or 0.0)
        features["end_voltage"] = float(window_df["PackV"].iloc[-1])

    # Cell-level aggregates
    for col_prefix in ["Vmax", "Vmin", "Tmax", "Tmin"]:
        if col_prefix in cols:
            features.update(_compute_series_stats(window_df[col_prefix], col_prefix.lower()))

    if "Vmax" in cols and "Vmin" in cols:
        features["volt_diff_mean"] = float((window_df["Vmax"] - window_df["Vmin"]).mean())

    if "Tmax" in cols and "Tmin" in cols:
        td = window_df["Tmax"] - window_df["Tmin"]
        features["temp_diff_mean"] = float(td.mean())
        features["temp_mean_total"] = float(((window_df["Tmax"] + window_df["Tmin"]) / 2).mean())

    # SOC features
    if "SOC" in cols:
        start_soc = float(start_row.get("SOC", 0.0) or 0.0)
        end_soc = float(window_df["SOC"].iloc[-1])
        features["start_soc"] = start_soc
        features["end_soc"] = end_soc
        features["delta_soc"] = start_soc - end_soc
        features["soc_below_50"] = 1 if start_soc < 50 else 0
        features["soc_70_to_80"] = 1 if 70 <= start_soc < 80 else 0
        features["soc_over_90"] = 1 if start_soc > 90 else 0

    # Clean NaNs
    return {k: (0.0 if pd.isna(v) or not np.isfinite(v) else float(v)) for k, v in features.items()}


def extract_event_window_features_selected(
    df_window: pd.DataFrame,
    original_row: pd.Series,
    selected_feature_names_list: List[str],
) -> Dict[str, Any]:
    """Extract selected features for an event window."""
    all_features = extract_all_features(df_window, original_row)
    out = {
        "event_time": original_row.get("timestamp"),
        "SOC_event": original_row.get("SOC"),
    }
    for name in selected_feature_names_list:
        out[name] = all_features.get(name, 0.0)
    return out

# --------------------------------------------------------------------------- #
# Feature dataset generation (placeholder)
# --------------------------------------------------------------------------- #

def generate_feature_dataset(
    data_df: pd.DataFrame,
    thresholds: Dict[str, Any],
    config: object,
    file_name: str,
    event_types: List[str] = ("acc", "brk"),
) -> pd.DataFrame:
    """
    Detect events and assemble per-event features.
    Implement event scanning with your project's detector, then call
    `extract_event_window_features_selected` per event window.
    """
    all_event_features: List[Dict[str, Any]] = []

    if len(data_df) > getattr(config, "EVENT_TIME_WINDOW", 0):
        logging.info("Scanning for events to generate feature dataset...")

    if not all_event_features:
        logging.warning(f"No events of types '{', '.join(event_types)}' were detected.")
        return pd.DataFrame()

    return pd.DataFrame(all_event_features)

# --------------------------------------------------------------------------- #
# Optimal feature selection
# --------------------------------------------------------------------------- #

def select_optimal_features(
    features_df: pd.DataFrame,
    target_column: str = "dV_target",
    config: Optional[object] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Select features using an ensemble of five scoring methods."""
    if features_df.empty or target_column not in features_df.columns:
        logging.warning("Empty features or missing target.")
        return [], pd.DataFrame()

    df = features_df.copy()
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    exclude = {target_column, "event_type", "group_id", "event_start_time"}
    feature_cols = [c for c in df.columns if c not in exclude]
    if not feature_cols:
        logging.warning("No candidate feature columns.")
        return [], pd.DataFrame()

    # Coerce to numeric and impute per-column median
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    df = df.dropna()
    if len(df) < 20:
        logging.warning(f"Insufficient rows ({len(df)}) for robust selection. Returning first 5.")
        return feature_cols[:5], pd.DataFrame()

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_column].to_numpy(dtype=float)

    # Remove zero-variance features
    var_mask = np.var(X, axis=0) > 1e-9
    X = X[:, var_mask]
    feature_cols = [c for c, m in zip(feature_cols, var_mask) if m]

    # Scoring
    scores = pd.DataFrame(index=feature_cols)
    seed = int(getattr(config, "SEED_VALUE", 42))

    # A. Spearman (handle constant columns -> NaN)
    sp = []
    for i in range(X.shape[1]):
        r = spearmanr(X[:, i], y, nan_policy="omit").correlation
        sp.append(0.0 if pd.isna(r) else float(abs(r)))
    scores["spearman"] = sp

    # B. Mutual information
    scores["mi"] = mutual_info_regression(X, y, random_state=seed)

    # C. F-test
    f_vals, _ = f_regression(X, y)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    scores["f_test"] = f_vals

    # D. Random Forest importance
    rf = RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
    rf.fit(X, y)
    scores["rf"] = rf.feature_importances_

    # E. Lasso coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=seed, n_jobs=-1)
    lasso.fit(X_scaled, y)
    scores["lasso"] = np.abs(lasso.coef_)

    # Normalize each column to [0,1]; avoid scalar return on constant columns
    def _norm(s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0.0

    normalized = scores.apply(_norm, axis=0)
    ensemble = normalized.mean(axis=1).sort_values(ascending=False)

    # Top-N and multicollinearity pruning
    max_k = int(getattr(config, "MAX_SELECTED_FEATURES", 10))
    top_features = list(ensemble.head(max_k).index)

    corr_thr = float(getattr(config, "SCA_INTERFEATURE_THRESHOLD", 0.9))
    corr = df[top_features].corr().abs()
    drop: set[str] = set()
    for i in range(len(corr.columns)):
        for j in range(i):
            if corr.iloc[i, j] > corr_thr:
                fi, fj = corr.columns[i], corr.columns[j]
                drop.add(fi if ensemble[fi] < ensemble[fj] else fj)

    final_features = [f for f in top_features if f not in drop]
    final_corr = df[final_features + [target_column]].corr()

    return final_features, final_corr
