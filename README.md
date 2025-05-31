

# Machine Learning in Astrophysics: Case Studies, Examples, and Resources

This repository represents the use cases of machine learning in astrophysics, with code examples and links to real astronomical datasets. 
Designed to work with most some interesting problems in astrophysics.

For more recent papers in the field please checkout : [Machine Learning for Astrophysics](https://ml4astro.github.io/icml2022/)


---

## Table of Contents

- [Overview](#overview)
- [Case Studies](#caseStudies)
- [Getting Started](#gettingStarted)
- [Example: Galaxy Classification with CNNs](#exampleGalaxyClassificationWithCnns)
- [Other Examples](#otherExamples)
- [Datasets](#datasets)
- [Additional Resources](#additionalResources)
- [Best Practices](#bestPractices)


---

## Overview

ML is essential in astrophysics for processing and analyzing vast, complex datasets. ML algorithms are used for:

- **Classifying astronomical objects** (galaxies, stars, exoplanets)
- **Detecting rare events** (gravitational waves, exoplanet transits)
- **Predicting and simulating cosmic phenomena**

---

## Case Studies

- **Galaxy Morphology Classification:** Automating galaxy shape classification using convolutional neural networks (CNNs).
- **Exoplanet Detection:** Identifying exoplanets from light curve data with deep learning.
- **Gravitational Wave Signal Classification:** Distinguishing real gravitational wave events from noise.
- **Solar Activity Forecasting:** Predicting solar flares using time series and image data.

---

## Getting Started

### Requirements

- Python 3.8+
- TensorFlow or PyTorch
- NumPy, scikit-learn, matplotlib

Install dependencies with:

```bash
pip install tensorflow numpy scikit-learn matplotlib
```


---

## Example: Galaxy Classification with CNNs

This example demonstrates how to build and train a simple CNN to classify galaxies using image data. For demonstration, synthetic data is used, but you can replace it with real astronomical datasets (see [Datasets](#datasets)).

**File:** [`galaxy_classification_cnn.py`](galaxy_classification_cnn.py)

---

## Other Examples

### 1. Exoplanet Detection from Light Curves

Detect exoplanet transits in stellar light curves using an LSTM neural network.

**File:** [`exoplanet_detection_rnn.py`](exoplanet_detection_rnn.py)

---

### 2. Gravitational Wave Signal Classification

Classify gravitational wave signals versus noise using a 1D CNN.

**File:** [`gravitational_wave_classification.py`](gravitational_wave_classification.py)

---

### 3. Solar Activity Forecasting

Forecast future solar activity values from time-series data using an LSTM.

**File:** [`solar_activity_forecasting_lstm.py`](solar_activity_forecasting_lstm.py)

---

## Datasets

- [GalaxiesML](https://arxiv.org/pdf/2410.00271.pdf): Multi-band galaxy images, photometry, redshifts, and structural parameters.
- [NASA Kepler \& TESS](https://exoplanetarchive.ipac.caltech.edu/): Light curve data for exoplanet detection.
- [LIGO Open Science Center](https://losc.ligo.org/): Gravitational wave signal data.
- [Solar Dynamics Observatory](https://sdo.gsfc.nasa.gov/data/): Solar image and time series data.

---

## Additional Resources

- [astroML](https://www.astroml.org/): Python tools and example datasets for ML in astronomy.
- [CAMELS Multifield Dataset](https://camels.readthedocs.io/en/latest/): Cosmological simulations for ML.
- [Multimodal Universe](https://github.com/MultimodalUniverse): Large scale multimodal ML dataset for astrophysics.



