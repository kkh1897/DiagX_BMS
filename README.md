# DiagX_BMS
Interpretable Early Diagnosis of EV Battery Faults from Field Data

# Description
The Python code includes the following components:
- event_detection.py : 
Implements a rule-based sliding window algorithm to detect significant acceleration and braking events from raw time-series data. These events serve as the fundamental units for analysis.

- feature_engineering.py : 
Extracts a comprehensive set of statistical and physical features from each event window and uses an ensemble method to select the most optimal and impactful features for the diagnostic models.

- neuro_symbolic.py : 
Contains the primary neuro-symbolic diagnostic model. Its unique architecture combines class-specific rule banks with a Transformer, enabling it to generate clear, logical explanations for its predictions.

- extrapolator.py : 
Provides dV-SOC profile forecasting using the FITS model. It fine-tunes on a per-trip basis to predict the full battery behavior profile from sparse data, enabling a highly precise secondary diagnosis of ambiguous cases.

# Contact
For questions or support, please contact [kkh1897@yonsei.ac.kr]
