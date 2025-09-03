"""
Event detection and battery health labeling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from feature_engineering import extract_event_window_features_selected

# A small constant for robust floating-point comparisons.
EPSILON = 1e-9


# =========================================================================== #
# Event Detection Algorithm
# =========================================================================== #

def identify_events_by_group(
    data_df: pd.DataFrame,
    thresholds: Dict[str, Any],
    feature_list: List[str],
    config: object,
    file_name: str,
    event_type: str = "acc",
) -> pd.DataFrame:
    """
    Detects events ('acc' or 'brk') within data groups using a rule-based sliding window.

    This function implements the core logic for identifying valid events by checking for
    monotonic trends, comparing changes against dynamic thresholds, and ensuring signal
    consistency throughout the event window.
    """
    window_size = int(getattr(config, "EVENT_TIME_WINDOW", 0))
    if window_size < 2:
        logging.error(f"EVENT_TIME_WINDOW must be an integer >= 2, but got {window_size}.")
        return pd.DataFrame()

    # --- Data validation and preparation ---
    required_cols = {"Group", "timestamp", "I_filt", "Current", "filt_dI_dt", "Vmin", "SOC", "deltaV", "Tmin"}
    if not required_cols.issubset(data_df.columns):
        raise ValueError(f"Input DataFrame is missing required columns: {required_cols - set(data_df.columns)}")

    df = data_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=list(required_cols))

    all_events: List[Dict[str, Any]] = []

    # Check for repair dates to apply different thresholds if necessary
    repair_dates = getattr(config, "REPAIR_DATES", {})
    is_repaired = file_name in repair_dates
    repair_ts = pd.to_datetime(repair_dates.get(file_name)) if is_repaired else None

    # --- Process data group by group ---
    for group_id, group_data in df.groupby("Group"):
        if len(group_data) < window_size:
            continue

        last_event_time = pd.NaT

        # --- Sliding Window Logic ---
        for i in range(len(group_data) - window_size + 1):
            window = group_data.iloc[i : i + window_size]
            start_row = window.iloc[0]

            # 1. Determine which set of thresholds to use (e.g., before/after repair)
            if is_repaired and repair_ts:
                period = "before" if start_row["timestamp"] < repair_ts else "after"
                active_thresholds = thresholds.get(period, {})
            else:
                active_thresholds = thresholds.get("default", thresholds)
            if not active_thresholds:
                continue

            # 2. Check for monotonic behavior of current and voltage
            seg_current_filt = window['I_filt'].values
            seg_vmin = window['Vmin'].values
            
            is_monotonic = False
            if event_type == "acc" and np.all(np.diff(seg_current_filt) > EPSILON) and np.all(np.diff(seg_vmin) <= 0):
                is_monotonic = True
            elif event_type == "brk" and np.all(np.diff(seg_current_filt) < -EPSILON) and np.all(np.diff(seg_vmin) >= 0):
                is_monotonic = True
            
            if not is_monotonic:
                continue

            # 3. Apply kinetic and magnitude thresholds
            prefix = "ACC" if event_type == "acc" else "BRK"
            current_thresh = float(active_thresholds.get(f"REC_CURRENT_THRESHOLD_{prefix}", 0.0))
            diff_thresh = float(active_thresholds.get(f"{prefix}_DIFF", 0.0))
            rate_thresh = float(active_thresholds.get(f"{prefix}_RATE", 0.0))
            
            starts_from_idle = abs(start_row['Current']) < current_thresh
            passes_diff_check = abs(seg_current_filt[-1] - seg_current_filt[0]) > diff_thresh
            passes_rate_check = abs(seg_current_filt[1] - seg_current_filt[0]) > rate_thresh
            
            if not (starts_from_idle and passes_diff_check and passes_rate_check):
                continue
            
            # 4. Final check for signal consistency across the entire window
            if not np.all(np.abs(window['filt_dI_dt'].values) > rate_thresh):
                continue

            # 5. If all checks pass, register the event and extract features
            current_event_time = start_row["timestamp"]
            if pd.notna(last_event_time) and (current_event_time - last_event_time).total_seconds() < window_size:
                continue
            last_event_time = current_event_time

            # Estimate internal resistance as a key physical feature
            dV = float(window['Vmin'].max() - window['Vmin'].min())
            dI = float(window['Current'].iloc[-1] - window['Current'].iloc[0])
            resistance = abs(dV / dI) if abs(dI) > EPSILON else np.nan

            event_data = {
                "event_time": current_event_time,
                "SOC_event": start_row["SOC"],
                "maxDV_event": start_row["deltaV"],
                "Temp_event": start_row["Tmin"],
                "R_event": resistance,
                "event_type": event_type,
                "group_id": group_id,
            }
            
            # Extract additional project-specific features using the external helper
            custom_features = extract_event_window_features_selected(window, start_row, feature_list)
            event_data.update(custom_features)
            all_events.append(event_data)

    if not all_events:
        return pd.DataFrame()

    # --- Post-processing: Filter out invalid events and outliers ---
    events_df = pd.DataFrame(all_events).dropna(subset=["R_event"])
    if events_df.empty:
        return pd.DataFrame()

    Q1 = events_df["R_event"].quantile(0.25)
    Q3 = events_df["R_event"].quantile(0.75)
    IQR = Q3 - Q1
    if IQR > EPSILON:
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        events_df = events_df[events_df["R_event"].between(lower_bound, upper_bound)]

    return events_df.reset_index(drop=True)


# =========================================================================== #
# Ground-Truth Labeler Training
# =========================================================================== #

def train_rf_labeler(config: object) -> Optional[RandomForestClassifier]:
    """
    Trains a RandomForest classifier to act as a ground-truth labeler.

    This function encapsulates the process of loading and merging vehicle data,
    engineering the core features (SOC and cell delta-V), assigning labels based
    on a summary file, and training a final model.
    """
    logging.info("--- Training Ground Truth Labeler (Random Forest) ---")

    # --- 1. Load and Merge Data ---
    try:
        vehicle_df = pd.read_parquet(config.VEHICLE_DATA_PATH)
        summary_df = pd.read_csv(config.SUMMARY_FILE_PATH, encoding=config.CSV_ENCODING)
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        return None

    summary_df = summary_df.rename(columns={"순번": "vid"})
    merged_df = pd.merge(vehicle_df, summary_df[["vid", "비고"]], on="vid", how="left")

    # --- 2. Feature Engineering ---
    soc_col = next((c for c in merged_df.columns if "soc_r" in c.lower()), None)
    cell_cols = [c for c in merged_df.columns if c.startswith("cell_")]
    if not soc_col or not cell_cols:
        logging.error("Required SOC or cell voltage columns not found in vehicle data.")
        return None

    V = merged_df[cell_cols].to_numpy(dtype=float) / 1000.0  # Convert mV to V
    soc = merged_df[soc_col].to_numpy(dtype=float) / 10.0      # Convert to %
    delta_v = (np.nanmax(V, axis=1) - np.nanmin(V, axis=1)) * 1000.0  # Calculate delta in mV

    # --- 3. Label Assignment ---
    merged_df["label"] = merged_df["비고"].map(config.LMAP_SUMMARY)
    
    # Allow for manual overrides for specific vehicles known to be degraded
    if hasattr(config, "FORCE_POTENTIAL_DEGRADATION_VIDS"):
        degraded_vids = set(config.FORCE_POTENTIAL_DEGRADATION_VIDS)
        pd_label = config.INV_LABELS_KOR.get("Potential Degradation")
        if pd_label is not None:
            merged_df.loc[merged_df["vid"].isin(degraded_vids), "label"] = pd_label

    # --- 4. Prepare Final Training Set and Train Model ---
    valid_mask = ~np.isnan(soc) & ~np.isnan(delta_v) & merged_df["label"].notna()
    X_train = np.column_stack([soc[valid_mask], delta_v[valid_mask]])
    y_train = merged_df.loc[valid_mask, "label"].values.astype(int)

    if len(X_train) == 0:
        logging.error("No valid data available for training the labeler.")
        return None

    logging.info(f"Training labeler on {len(X_train)} data points.")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=config.SEED_VALUE,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    logging.info("Ground Truth Labeler training complete.")
    return model