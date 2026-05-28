# Analysis and Optimization of Interferometric Evaluation Using Machine Learning

## Overview

This repository supports a master-level research project exploring the analysis and optimization of interferometric evaluation in optical metrology through machine learning techniques. The work combines synthetic data generation, image preprocessing, segmentation, and regression-based Zernike coefficient prediction to improve the evaluation of interferometric measurements.

## Project Scope

The project contains three primary pillars:

1. Synthetic interferogram generation from Zernike coefficient sets.
2. Preprocessing and mask extraction of interferometric images.
3. Machine learning models for predicting Zernike-induced wavefront aberrations.

The implementation emphasizes reproducible experiments, configurable data generation, and practical inference on both synthetic and experimental data.

## Repository Structure

- `data/`
  - `preprocessing/` — datasets and partitions used for segmentation and bounding-box preprocessing.
  - `zernike_prediction/` — stored experimental and synthetic prediction results.
- `results/`
  - `preprocessing/` — bounding box and segmentation outputs.
  - `zernike_prediction/` — trained model outputs, test results, and evaluation artifacts.
- `src/`
  - `data_generation/` — synthetic interferogram and Zernike data generation utilities.
  - `data_meopta_processing/` — domain-specific utilities for reading and converting Meopta `.dat` data and masks.
  - `preprocessing/` — image preprocessing pipelines for bounding box detection and segmentation.
  - `zernike_prediction/` — model training, inference, tilt extraction, and plotting utilities.

## Core Components

### Synthetic Data Generation (`src/data_generation`)

This module simulates interferometric intensity images from Zernike coefficient sets.

- `polynomials.py` defines a Zernike polynomial basis for a circular aperture.
- `coefficients.py` generates randomized Zernike coefficient vectors with physically motivated ranges.
- `noise.py` implements realistic sensor and interferometric noise, including read noise, Poisson-like noise, dark current, speckle disturbances, and dust artifacts.
- `interferogram.py` and the various `interferogram_zernike` scripts synthesize intensity interferograms by combining Zernike modes and adding noise.
- `config_0.json` stores generation parameters such as image resolution, wavelength, noise levels, and output filenames.
- `run.sh` is an HPC job wrapper that prepares the scratch environment and executes interferogram generation in a PBS cluster context.

### Preprocessing and Segmentation (`src/preprocessing`)

This branch prepares input images for downstream learning and analysis.

- `bounding_box/` contains tools for detecting the region of interest in interferometric images.
- `segmentation/` implements a U-Net segmentation pipeline for mask extraction.
- `segmentation/inference.py` demonstrates a full mask prediction flow using a U-Net model, including resizing, model inference, mask interpolation back to original resolution, IoU computation, and visualization.

### Zernike Prediction and Model Training (`src/zernike_prediction`)

This section develops and evaluates machine learning models for predicting wavefront aberrations.

- `models.py` defines a modified ResNet-18 regressor that accepts interferogram images plus explicit tilt inputs and predicts Zernike coefficients.
- `tilt_extractor.py` extracts tilt coefficients (Z2, Z3) from an interferogram using FFT-based spectral peak detection.
- `training_zernike.py` implements the training pipeline, data loading, dataset splitting, custom loss scheduling, and batch preparation.
- `inference_zenrikes.py` loads a saved checkpoint together with test data, performs prediction, and visualizes model performance.
- `inference_experimental.py` is a dedicated script for experimental inference using actual measurement images and Zemax-style coefficient files.
- `train_config.json` centralizes hyperparameters and resource settings for training experiments, including noise handling, normalization, optimizer settings, and dataset paths.
- `run_train.sh` is an HPC launch wrapper for submitting the training script to a PBS cluster.

## Usage Guide

### Environment

The codebase assumes a Python scientific environment with packages such as:

- `numpy`
- `torch` / `torchvision`
- `scikit-learn`
- `matplotlib`
- `opencv-python`
- `Pillow`
- `segmentation_models_pytorch`
- `pydantic`

### Generating Synthetic Data

The central generation script is `src/data_generation/interferogram_zernike.py`, which loads parameters from `src/data_generation/config_0.json`. The simplest usage pattern is:

```bash
cd src/data_generation
python interferogram_zernike.py
```

To run on an HPC cluster, adapt and execute `src/data_generation/run.sh` as a PBS batch job.

### Training the Zernike Regression Model

The training pipeline is configured by `src/zernike_prediction/train_config.json` and executed by `src/zernike_prediction/training_zernike.py`.

A typical workflow is:

```bash
cd src/zernike_prediction
python training_zernike.py
```

For cluster execution, `src/zernike_prediction/run_train.sh` provides a template PBS wrapper.

### Model Inference

- Use `src/zernike_prediction/inference_zenrikes.py` to evaluate trained models on stored test datasets.
- Use `src/zernike_prediction/inference_experimental.py` for experimental inference on measured interferograms and Zemax coefficient files.

### Segmentation Inference

Run `src/preprocessing/segmentation/inference.py` to apply the segmentation pipeline on interferogram images and compute mask quality metrics. This script also saves visualizations of predicted masks and overlayed results.

## Research Value

The repository is designed to support a structured investigation into how machine learning can assist optical metrology:

- synthetic interferogram generation for controlled experiments;
- robust preprocessing and mask extraction for real interferometric imagery;
- regression of Zernike coefficients using learned representations;
- hybrid use of classical tilt extraction with learned higher-order aberration prediction.

## Notes

- The repository assumes access to datasets and model checkpoints stored under `meta_output/` and `results/`.
- Several wrappers are tailored to PBS / HPC environments and may require adaptation for local execution.
- The codebase remains research-oriented and is organized around reproducible experiment configuration rather than production packaging.

## License

This repository is part of a master’s thesis project. Please refer to the license file for usage conditions.
