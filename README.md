# Variational Autoencoder (VAE) for Anomaly Detection in High-Dimensional Time Series

## Objective
This project implements a Variational Autoencoder (VAE) to detect subtle anomalies in high-dimensional time series data. The model is trained only on normal data and identifies anomalous sequences based on reconstruction error and latent space divergence.

## Dataset Generation
A synthetic multivariate time series dataset with 15 features and 10,000 time steps is generated using NumPy. Anomalies (1%) are injected by introducing correlated magnitude shifts across multiple dimensions, simulating real-world sensor or financial irregularities.

## Model Architecture
- Encoder: Fully connected layers producing mean and log-variance
- Latent space: Gaussian latent variables
- Decoder: Symmetric fully connected reconstruction network
- Loss Function:
  - Reconstruction Loss (MSE)
  - KL Divergence with Beta-VAE regularization

## Training Strategy
The model is trained only on normal data. The Beta parameter controls the balance between reconstruction fidelity and latent space regularization. Beta was tuned empirically to improve anomaly separability.

## Anomaly Detection
Anomalies are detected using reconstruction error thresholds derived from training distribution percentiles. Performance is evaluated using Precision, Recall, and F1-score.

## Performance Metrics
| Metric | Score |
|------------|-------|
| Precision | Reported after evaluation |
| Recall | Reported after evaluation |
| F1-score | Reported after evaluation |

## Tools Used
- Python
- NumPy
- PyTorch
- Scikit-learn
