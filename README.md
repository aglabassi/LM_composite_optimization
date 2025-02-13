# Unified Matrix and Tensor Factorization

This repository provides a collection of scripts and utilities for matrix and CP tensor factorization using both **NumPy 1.22.4** (in float64) and **PyTorch 1.10.1**. It includes:

- **Matrix-based experiments** using NumPy (float64 precision) to provide more precision when computing the L2 norm squared.
- **GPU-accelerated factorization** in PyTorch, featuring our approach that supports both matrix and CP tensor factorization (symmetric or asymmetric).
- **Neural network examples** to showcase how our Levenberg-Marquardt subgradient method can integrate with deep learning frameworks.

## Contents

- **Levenberg-Marquardt_sensing.py**  
  - PyTorch-based script providing a unified template for matrix and CP tensor factorization.  
  - Designed to run on GPUs if available.  
  - May exhibit numerical issues when using double precision (float64)  with the L2-nrom squared. However, these issues disapear when using L2 or L1 norm.

- **matrix_experiments/**  
  - Contains multiple scripts (e.g., `main.py`, `main_hadamard.py`, `phase_transition_experiment.py`, etc.) for running matrix factorization tasks purely in NumPy,
  which reduces numerical instability on the L2 norm squared.

- **neural_net_example.py**  
  - Demonstrates how composite optimization can be integrated within a neural network.
  - Serves as a starting point for combining composite optimization with deep learning models.

- **utils.py**  
  - General-purpose utility functions shared across the project.  
  - May include data loading, logging, plotting and other helper routines.

- **experiment_results/**  
  - Directory for saving experimental outputs, logs, and possibly generated plots or performance metrics.

## Getting Started

1. **Install Dependencies**  
   - **Python 3.8+**  
   - **NumPy 1.22.4**  
   - **PyTorch 1.10.1** (for GPU-based scripts)  
   ```bash
   pip install numpy==1.22.4 torch==1.10.1
