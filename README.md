# Kernel Methods

> **Project 2** - CSC14005 - Introduction to Machine Learning (Nhập môn Học máy)
> University of Science, Vietnam National University Ho Chi Minh City (VNUHCM-US)
> Faculty of Information Technology

## Table of Contents

- [Overview](#overview)
- [Course Information](#course-information)
- [Selected Chapter](#selected-chapter)
- [Project Structure](#project-structure)
- [What We Implemented](#what-we-implemented)
- [Extensions Beyond the Book](#extensions-beyond-the-book)
- [Code Instructions](#code-instructions)
- [Reference](#reference)
- [Team](#team)
- [License](#license)

## Overview

This project studies **kernel methods**, a core family of techniques in machine learning for modeling nonlinear structure through implicit feature mappings. The selected material is **Chapter 6: Kernel Methods** from *Foundations of Machine Learning* by Mohri, Rostamizadeh, and Talwalkar. The report connects the mathematical foundations of positive definite symmetric kernels, reproducing kernel Hilbert spaces, kernel-based algorithms, sequence kernels, and scalable approximate feature maps.

The experimental code provides small, reproducible demonstrations for the main ideas discussed in the report:

| Experiment | File | Goal |
|---|---|---|
| Kernel matrices | `code/pds-kernel/kernel.py` | Compute common PDS kernels such as linear, polynomial, RBF, Gaussian, and sigmoid kernels |
| Kernel SVM | `code/pds-kernel/svm.py` | Visualize SVM decision boundaries with different kernels on a nonlinear two-moons dataset |
| Weighted transducers | `code/transducer/weighted_transducer.py` | Demonstrate weighted finite-state transducer composition with epsilon filtering |
| Approximate kernel feature maps | `code/approximate/approx.py` | Verify random Fourier feature approximations for Gaussian, Laplacian, and Cauchy kernels |

## Course Information

| | |
|---|---|
| **Course** | Introduction to Machine Learning (Nhập môn Học máy) |
| **Instructor (Theory)** | Dr. Bùi Tiến Lên |
| **Instructor (Lab)** | MSc. Lê Nhựt Nam |
| **Class** | 23_24 |
| **Group** | 13 |
| **Semester** | Semester 2, 2026 |

## Selected Chapter

The selected chapter is **Chapter 6: Kernel Methods** from *Foundations of Machine Learning* by Mohri, Rostamizadeh, and Talwalkar. The project focuses on the following parts:

1. Mathematical preliminaries for metric spaces, compactness, convergence, inner products, Hilbert spaces, and formal languages.
2. Positive definite symmetric kernels and their connection to reproducing kernel Hilbert spaces.
3. Kernel-based learning algorithms, including SVMs with PDS kernels and the representer theorem.
4. Negative definite symmetric kernels.
5. Sequence kernels through weighted transducers and rational kernels.
6. Approximate kernel feature maps through random Fourier features.
7. Recent research directions in scalable kernel learning and kernel-based deep learning.

## Project Structure

```text
kernel-methods/
├── code/
│   ├── approximate/
│   │   └── approx.py                 # Random Fourier feature approximation demo
│   ├── pds-kernel/
│   │   ├── kernel.py                 # Common kernel functions
│   │   └── svm.py                    # SVM decision boundary visualization
│   ├── transducer/
│   │   └── weighted_transducer.py    # Weighted transducer composition demo
│   ├── requirements.txt              # Python dependencies
│   └── README.md                     # Code usage guide
├── report/
│   ├── report.tex                    # Main LaTeX source
│   ├── report.pdf                    # Compiled report
│   └── ...                           # LaTeX sections, tables, and figures
└── README.md                         # Project overview
```

## What We Implemented

The report and code cover both theoretical and practical aspects of kernel methods:

1. A structured theoretical report written in LaTeX.
2. Examples and proofs for PDS kernels, RKHS, kernel algorithms, NDS kernels, sequence kernels, and approximate feature maps.
3. SVM experiments comparing different kernels on a nonlinear classification dataset.
4. A weighted transducer implementation with composition and epsilon filtering.
5. Monte Carlo verification of random Fourier feature approximations for shift-invariant kernels.
6. A short survey of recent research: large kernel models, random Gegenbauer features, and RKHM-based deep learning.

## Extensions Beyond the Book

Compared with the core book material, this project adds:

1. Additional explanations and derivations for lemmas and theorems whose details are brief in the source material.
2. Visual experiments showing how different kernels affect SVM decision boundaries.
3. A concrete weighted-transducer composition implementation for sequence-kernel intuition.
4. Statistical verification of approximate feature maps using one-sample t-tests.
5. A review of recent kernel-method research directions from ICML 2022, ICML 2023, and NeurIPS 2023.

## Code Instructions

See [`code/README.md`](code/README.md).

## Reference

The main theoretical material is based on:

- Mohri, M., Rostamizadeh, A., & Talwalkar, A. (2018). *Foundations of Machine Learning* (2nd ed.). MIT Press.

## Team

| Name | Student ID | Role |
|---|---|---|
| Hoàng Ngọc Phú | 23120010 | Team Lead |
| Hoàng Ngọc Quí | 23120077 | Member |
| Nguyễn Duy Bảo | 23120113 | Member |

## License

This project is developed for educational purposes as part of the Introduction to Machine Learning course at VNUHCM-US.
