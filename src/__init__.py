"""Reusable library for the Sentinel-1 SAR bushfire detection pipeline.

Sub-packages:
    src.data      - ingestion (CDSE/FIRMS + synthetic), calibration, PyTorch dataset
    src.features  - multi-channel feature engineering and feature selection
    src.models    - loss functions, U-Net, metrics, training loop, inference
    src.utils     - geospatial helpers, spatial cross-validation, logging
    src.config    - typed configuration loader
"""

__version__ = "0.1.0"
