# DiagX_BMS

# Introduction
Electric vehicle batteries degrade under coupled electrochemical, thermal, and mechanical stresses, yet early fault detection remains challenging due to limited sensitivity and interpretability of conventional diagnostics. We present a two-stage framework that converts sparse, noisy field data into physically grounded metrics. Stage 1 detects high-stress acceleration and braking events and applies a neuro-symbolic model for rapid, interpretable screening. For trips with detected events, stage 2 reconstructs voltage-deviation trajectories into low state-of-charge regions using a physics-guided FITS extrapolator, enabling re-diagnosis that reveals latent defects. This repository contains the Python scripts to recreate the results in the paper: “*Interpretable Early Diagnosis of EV Battery Faults from Field Data*”.

<img width="9387" height="3024" alt="Image" src="https://github.com/user-attachments/assets/25e36f35-0f99-4407-9974-6432e28ae47a" />

# Description
The Python code includes the following components:
- `event_detection.py` : 
Implement a rule-based sliding window algorithm to detect significant acceleration and braking events from raw time-series data. These events serve as the fundamental units for analysis.

- `feature_engineering.py` : 
Extract a comprehensive set of statistical and physical features from each event window and uses an ensemble method to select the most optimal and impactful features for the diagnostic models.

- `neuro_symbolic.py` : 
Contain the primary neuro-symbolic diagnostic model. Its unique architecture combines class-specific rule banks with a Transformer, enabling it to generate clear, logical explanations for its predictions.

- `extrapolator.py` : 
Provide dV-SOC profile forecasting using the FITS model. It fine-tunes on a per-trip basis to predict the full battery behavior profile from sparse data, enabling a highly precise secondary diagnosis of ambiguous cases.

# Contact
For questions or support, please contact [kkh1897@yonsei.ac.kr]
