# Lens Search: Machine Learning and Spectral Fitting Pipeline (QSL-ML: Quasar as Strong Lenses using Machine Learning)

This repository provides a full pipeline to identify quasars (QSOs) that act as strong lenses using deep learning and Gaussian modeling on spectral data.

## Overview

The core of this pipeline is the class, `QSOFluxProcessor`, which combines:

- A Convolutional Neural Network (CNN) classifier to detect QSOs acting as strong lenses.
- A CNN-based redshift predictor.
- A dual-Gaussian fitting method to detect and model the [OII] doublet emission lines.
- Comprehensive visualization tools to assess fits and spectral features.

This tool is built for batch processing of QSO spectra, producing classification scores, redshift estimates, and diagnostics like signal-to-noise ratio (SNR) and reduced chi-squared.

## Features

- **Classification**: Predict whether a spectrum corresponds to a lensed QSO.
- **Redshift Estimation**: CNN-based regression model to estimate QSO redshifts.
- **Spectral Line Fitting**: Fit the [OII] 3726, 3729 Å doublet with physical constraints.
- **Plotting**: Visualization of spectral fits, QSO emission lines, and prominent emission features like Hβ and [OIII].
- **Batch Processing**: Designed to process large datasets and return a pandas DataFrame summarizing key parameters.

## File Requirements

- `wave.npy`: A NumPy array of DESI wavelength values.
- `SDSS2.txt`: Text file containing QSO emission line wavelengths and equivalent widths (used for plotting).

## Key Methods

### `QSOFluxProcessor.process(fluxes, Phase, names=None, noise=None, z_qsos=None)`
Main pipeline for classification, redshift estimation, and spectral fitting.

Returns:
- A pandas DataFrame with object name, score, classification, redshift values, fit quality, and SNR.

### `QSOFluxProcessor.plot_gaussian_fit_with_subplots(...)`
Generates a multi-panel figure showing the full spectrum, OII fit, and Hβ/OIII region, optionally overlaying known QSO lines.

## Dependencies

- Python 3.7+
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `keras` with TensorFlow backend

## Example Usage

```python
from Lens_Search import QSOFluxProcessor
import numpy as np

# Load your flux data (shape: [n_samples, 7781])
fluxes = np.load("my_flux_data.npy")
noise = np.load("my_noise_data.npy")
names = ["Target1", "Target2", "Target3"]
z_qsos = [1.23, 0.98, 1.05]  # Optional true QSO redshifts

results_df = QSOFluxProcessor.process(fluxes, Phase=1, names=names, noise=noise, z_qsos=z_qsos)
print(results_df)

 
