# -*- coding: utf-8 -*-
"""
Battery health diagnosis from dV–SOC profiles: reference templates, FITS model,
envelope extraction, per-group fine-tuning, and full-file diagnosis.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler


# =========================================================================== #
# Reference Profile Generation
# =========================================================================== #

def build_profile_templates(config: object) -> Dict[int, interp1d]:
    """
    Builds or loads cached reference dV-vs-SOC profiles for each health class.

    This function processes a fleet of historical data, aggregates the dV-SOC
    relationship for each health state (e.g., Normal, Degraded, Fault), and
    creates smooth, representative interpolation functions.
    """
    cache_path = Path(getattr(config, "PROFILE_CACHE_PATH", "profile_cache.pkl"))
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                logging.info(f"Loading cached reference profiles from {cache_path}.")
                return pickle.load(f)
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_path}: {e}. Rebuilding.")

    logging.info("Building new reference dV profiles from source data...")

    summary_df = pd.read_csv(getattr(config, "SUMMARY_FILE_PATH"))
    label_map = getattr(config, "LABEL_MAP")
    
    profiles_data = {lbl: {"soc": [], "dv": []} for lbl in label_map.values()}
    
    # This loop conceptually represents iterating through a fleet's data
    for vehicle_id, vehicle_data in load_vehicle_data_generator(summary_df, config):
        label = get_vehicle_label(vehicle_id, vehicle_data, config)
        if label is None:
            continue
        
        soc, dv = extract_soc_dv(vehicle_data, config)
        if soc is not None and dv is not None:
            profiles_data[label]["soc"].append(soc)
            profiles_data[label]["dv"].append(dv)

    ref_funcs: Dict[int, interp1d] = {}
    for label, data in profiles_data.items():
        if not data["soc"]:
            continue
        all_soc = np.concatenate(data["soc"])
        all_dv = np.concatenate(data["dv"])
        
        # Aggregate data to create a smooth profile
        agg_df = pd.DataFrame({"SOC": all_soc, "dV": all_dv}) \
            .groupby(pd.cut(all_soc, bins=101))['dV'].mean().reset_index()

        if len(agg_df) > 2:
            ref_funcs[label] = interp1d(
                agg_df['SOC'], agg_df['dV'],
                kind="linear", bounds_error=False,
                fill_value="extrapolate"
            )

    with open(cache_path, "wb") as f:
        pickle.dump(ref_funcs, f)
    logging.info(f"Saved new reference profiles to {cache_path}.")
    return ref_funcs


# =========================================================================== #
# Time-Series Forecasting Model
# =========================================================================== #

class FITSModel(nn.Module):
    """
    FITS model with an SOC-conditional adjustment network for dV profile forecasting.
    """
    def __init__(self, seq_len: int, pred_len: int, cut_freq: int, config: object):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cut_freq = cut_freq
        hidden_neurons = int(getattr(config, "FITS_HIDDEN_NEURONS", 512))

        # Frequency domain layers for upsampling (extrapolation)
        self.length_ratio = (seq_len + pred_len) / seq_len
        self.upsample_real = nn.Linear(cut_freq, int(cut_freq * self.length_ratio))
        self.upsample_imag = nn.Linear(cut_freq, int(cut_freq * self.length_ratio))

        # Time domain MLP to learn SOC-dependent corrections
        self.adjust_net = nn.Sequential(
            nn.Linear(1 + 1, hidden_neurons), # Input: [SOC, dV_from_FITS]
            nn.GELU(), nn.LayerNorm(hidden_neurons), nn.Dropout(0.3),
            nn.Linear(hidden_neurons, hidden_neurons), nn.GELU(), nn.LayerNorm(hidden_neurons), nn.Dropout(0.3),
            nn.Linear(hidden_neurons, 1) # Output: dV_adjustment
        )

    def forward(self, x: torch.Tensor, soc: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        # 1. Instance Normalization
        mean = x.mean(1, keepdim=True)
        variance = x.var(1, keepdim=True) + 1e-5
        x_norm = (x - mean) / variance.sqrt()

        # 2. Frequency Domain Extrapolation
        spec = torch.fft.rfft(x_norm, dim=1)[:, :self.cut_freq, :]
        upsampled_spec = torch.complex(
            self.upsample_real(spec.real.transpose(1, 2)).transpose(1, 2),
            self.upsample_imag(spec.imag.transpose(1, 2)).transpose(1, 2)
        )
        extrapolated = torch.fft.irfft(upsampled_spec, n=self.seq_len + self.pred_len, dim=1)
        
        # 3. Denormalization
        fits_out = extrapolated * self.length_ratio * variance.sqrt() + mean
        if is_training:
            fits_out = fits_out[:, :self.seq_len, :]

        # 4. SOC-Conditional Adjustment
        adjust_input = torch.cat([soc.view(-1, 1), fits_out.view(-1, 1)], dim=1)
        adjustments = self.adjust_net(adjust_input).view(fits_out.shape)
        
        return fits_out + adjustments


# =========================================================================== #
# Fine-Tuning and Prediction Algorithm
# =========================================================================== #

def extract_upper_envelope(soc: np.ndarray, dv: np.ndarray, config: object) -> interp1d:
    """
    Extracts a smoothed upper envelope from sparse dV-SOC data points.
    This involves robust downsampling (e.g., LTTB) and smoothing.
    """
    
    # Fallback for conceptual representation:
    valid_mask = ~np.isnan(soc) & ~np.isnan(dv)
    if valid_mask.sum() < 2:
        return interp1d([0, 100], [0, 0], fill_value="extrapolate")
    
    df = pd.DataFrame({'SOC': soc[valid_mask], 'dV': dv[valid_mask]})
    agg = df.groupby(pd.cut(df['SOC'], bins=101))['dV'].max().ffill().bfill()
    return interp1d(agg.index.mid, agg.values, bounds_error=False, fill_value="extrapolate")


def fine_tune_and_predict(
    group_df: pd.DataFrame,
    ref_profiles: Dict[int, interp1d],
    config: object,
    device: torch.device
) -> Optional[pd.DataFrame]:
    """
    Fine-tunes the FITS model on a single data group and predicts the full dV profile.

    This is the core algorithm. It trains a temporary, specialized FITS model for each
    data group, using its observed data for reconstruction and a reference profile
    for guidance. An ensemble of models with Monte Carlo dropout is used to estimate
    a robust prediction with confidence intervals.
    """
    # --- 1. Data Preparation for the specific group ---
    soc_raw = group_df[getattr(config, "SOC_COL")].values
    dv_raw = (group_df[getattr(config, "VMAX_COL")] - group_df[getattr(config, "VMIN_COL")]).values

    if np.isnan(soc_raw).all() or len(np.unique(soc_raw[~np.isnan(soc_raw)])) < 2:
        return None

    envelope_interp = extract_upper_envelope(soc_raw, dv_raw, config)
    observed_soc = np.unique(np.round(soc_raw[~np.isnan(soc_raw)]).astype(int))
    observed_dv = envelope_interp(observed_soc)
    
    min_soc, max_soc = int(observed_soc.min()), int(observed_soc.max())
    train_soc_grid = np.arange(max_soc, min_soc - 1, -1) # Uniform grid for FFT
    train_dv_grid = interp1d(observed_soc, observed_dv, fill_value="extrapolate")(train_soc_grid)
    
    # Scalers and Tensors
    dv_scaler = MinMaxScaler().fit(train_dv_grid.reshape(-1, 1))
    soc_scaler = MinMaxScaler().fit(train_soc_grid.reshape(-1, 1))
    inp_tensor = torch.tensor(dv_scaler.transform(train_dv_grid.reshape(-1, 1)), dtype=torch.float32, device=device).unsqueeze(0)
    soc_tensor_train = torch.tensor(soc_scaler.transform(train_soc_grid.reshape(-1, 1)), dtype=torch.float32, device=device).unsqueeze(0)
    
    # Determine the best reference profile to use as a learning guide
    initial_label = 0 # Placeholder for DTW-based label determination
    ref_dv_scaled = torch.tensor(dv_scaler.transform(ref_profiles[initial_label](train_soc_grid).reshape(-1,1)), dtype=torch.float32, device=device).flatten()

    # --- 2. Ensemble Fine-Tuning Loop ---
    ensemble_predictions = []
    for seed in range(int(getattr(config, "N_ENSEMBLES", 3))):
        torch.manual_seed(seed)
        model = FITSModel(len(train_soc_grid), int(min_soc), getattr(config, "FITS_CUT_FREQ"), config).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=float(getattr(config, "FITS_LR")))
        
        for _ in range(int(getattr(config, "FITS_EPOCHS", 100))):
            model.train()
            optimizer.zero_grad()
            output = model(inp_tensor, soc_tensor_train, is_training=True)
            
            # Composite loss: reconstruct observed data + follow reference guide
            loss_recon = F.mse_loss(output, inp_tensor)
            loss_guide = F.mse_loss(output.squeeze(), ref_dv_scaled)
            total_loss = float(getattr(config, "W_RECON")) * loss_recon + float(getattr(config, "W_GUIDE")) * loss_guide
            
            total_loss.backward()
            optimizer.step()

        # --- 3. Prediction with Uncertainty (Monte Carlo Dropout) ---
        full_soc_grid = np.arange(max_soc, -1, -1)
        full_soc_tensor = torch.tensor(soc_scaler.transform(full_soc_grid.reshape(-1, 1)), dtype=torch.float32, device=device).unsqueeze(0)

        model.train() # Keep dropout active for MC sampling
        with torch.no_grad():
            for _ in range(int(getattr(config, "N_MC_SAMPLES", 10))):
                full_output = model(inp_tensor, full_soc_tensor, is_training=False)
                full_dv = dv_scaler.inverse_transform(full_output.squeeze().cpu().numpy().reshape(-1, 1)).flatten()
                ensemble_predictions.append(np.maximum(0, full_dv))

    # --- 4. Final Profile with Confidence Intervals ---
    preds_array = np.array(ensemble_predictions)
    dv_mean = np.mean(preds_array, axis=0)
    dv_std = np.std(preds_array, axis=0)
    
    # Widen CI in extrapolated regions
    ci_scale = np.ones_like(full_soc_grid, dtype=float)
    extrap_mask = full_soc_grid < min_soc
    if extrap_mask.any() and min_soc > 0:
        ci_scale[extrap_mask] = 1.0 + (min_soc - full_soc_grid[extrap_mask]) / min_soc * 2.0
        
    return pd.DataFrame({
        "pred": dv_mean,
        "CI_upper": dv_mean + 1.96 * dv_std * ci_scale,
        "CI_lower": np.maximum(0, dv_mean - 1.96 * dv_std * ci_scale)
    }, index=full_soc_grid)
