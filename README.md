# Autoencoders for Anomaly Detection (PyTorch)

This repository contains a PyTorch-based anomaly detection project using **autoencoders** on a time-series dataset. The notebook demonstrates how reconstruction-based learning can be used to identify anomalous patterns in data without explicit anomaly labels.

## Python Script:
- A complete anomaly detection pipeline implemented in a single Jupyter Notebook
- Data preprocessing including scaling and data splitting
- Multiple autoencoder architectures built using PyTorch
- Anomaly detection using reconstruction error and percentile-based thresholding
- Visual analysis of reconstruction errors and detected anomalies

## Dataset
The project uses a time-series dataset from the NAB benchmark:
- `ambient_temperature_system_failure.csv`

The notebook references the dataset using a Kaggle-style path:
- `/kaggle/input/a2-input/ambient_temperature_system_failure.csv`

For local execution, the dataset path can be updated to a local directory such as:
- `data/ambient_temperature_system_failure.csv`

## What’s happening in the project
- The model learns normal behavior in the time-series by compressing and reconstructing the input data.
- Since autoencoders are trained mostly on normal patterns, they struggle to accurately reconstruct abnormal data.
- Reconstruction error is calculated as the difference between the input and reconstructed output.
- A threshold is chosen using a percentile of reconstruction error values.
- Data points with reconstruction error above this threshold are classified as anomalies.

## Models implemented
- Fully Connected (Dense) Autoencoder
- Convolutional (Conv1D) Autoencoder
- LSTM Autoencoder

## Requirements
Python 3.9+

Main libraries used:
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- torchinfo
