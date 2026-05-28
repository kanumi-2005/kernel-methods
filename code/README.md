# Code Guide

This directory contains the Python source code used for the experimental part of the **Kernel Methods** project.

## Directory Structure

```text
code/
├── approximate/
│   └── approx.py                 # Random Fourier feature approximation demo
├── pds-kernel/
│   ├── kernel.py                 # Common kernel matrix functions
│   └── svm.py                    # Kernel SVM visualization
├── transducer/
│   └── weighted_transducer.py    # Weighted transducer composition demo
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Environment

Use Python 3.10 or newer. The code was written for a standard scientific Python environment with NumPy, SciPy, pandas, scikit-learn, Matplotlib, and NetworkX.

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Files

| File | Description |
|---|---|
| `pds-kernel/kernel.py` | Implements common positive definite symmetric kernel functions: linear, polynomial, RBF, Gaussian, and sigmoid kernels. Running this file prints sample kernel matrices for random inputs. |
| `pds-kernel/svm.py` | Trains SVM classifiers with linear, polynomial, RBF, and sigmoid kernels on a synthetic two-moons dataset, then visualizes decision regions, margins, and support vectors. |
| `transducer/weighted_transducer.py` | Implements a small weighted finite-state transducer class, trimming, graph drawing, and transducer composition with an epsilon filter. Running this file draws the input transducers and the composed result. |
| `approximate/approx.py` | Implements exact Gaussian, Laplacian, and Cauchy kernels, samples their associated random Fourier frequencies, and uses one-sample t-tests to compare Monte Carlo estimates with exact kernel values. |

## Running the Demos

Run all commands from the `code/` directory.

### 1. Kernel Matrix Demo

```bash
python pds-kernel/kernel.py
```

Expected output: printed kernel matrices for randomly generated input arrays.

### 2. Kernel SVM Demo

```bash
python pds-kernel/svm.py
```

Expected output: a Matplotlib window showing four SVM decision boundary plots for linear, polynomial, RBF, and sigmoid kernels.

### 3. Weighted Transducer Demo

```bash
python transducer/weighted_transducer.py
```

Expected output: Matplotlib graph windows for transducers `T1`, `T2`, and their epsilon-filtered composition. The terminal also prints the number of generated transitions after filtering.

### 4. Approximate Kernel Feature Map Demo

```bash
python approximate/approx.py
```

Expected output: terminal tables for Gaussian, Laplacian, and Cauchy kernels. Each table reports the exact kernel value, Monte Carlo mean, t-statistic, and p-value for several random vector pairs.

## Reproducibility Notes

- Each demo fixes the NumPy or scikit-learn random seed where randomness is used.
- Visualization scripts require a graphical backend for Matplotlib.
- The approximation demo uses `D = 100000` random samples, so it may take longer than the other scripts.
- No external datasets are required; all experiments use synthetic data generated inside the scripts.
