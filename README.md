# Manifold-Constrained Counterfactual Explanations for Medical Diagnostics



![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

![Status](https://img.shields.io/badge/Status-Research_Complete-success)



## 📌 Overview



This project implements a **Trustworthy AI framework** for explaining "Black Box" medical models. While standard Deep Learning models can accurately predict Heart Disease, they often fail to explain *why* or *how* a patient can improve.



Standard explanation methods (like gradient-based perturbations) often generate **adversarial examples**—mathematically correct but biologically impossible suggestions (e.g., "Lower your age by 5 years" or "Increase cholesterol while lowering blood pressure").



**This solution uses a Variational Autoencoder (VAE)** to learn the "Patient Manifold" (the natural distribution of realistic physiological data). By performing optimization in the VAE's **Latent Space** rather than pixel space, we generate **Manifold-Constrained Counterfactuals**: actionable, realistic changes that a high-risk patient can actually achieve.



---



## 🔬 Methodology



The framework consists of two coupled neural networks:



1.  **The Predictor (MLP):**

    * A PyTorch Neural Network trained to predict heart disease risk.

    * **Performance:** Achieved **ROC AUC of 0.9132**.

    * **Calibration:** Utilized **Youden’s J Statistic** to optimize the decision threshold (0.1241), reducing False Negatives by **66%** compared to standard thresholds.



2.  **The Explainer (VAE):**

    * A **Variational Autoencoder** trained to map patient data into a compressed 16-dimensional latent space ($z$).

    * **Inference:** To generate a counterfactual, we freeze immutable features (Age, Sex) and optimize a vector $z^*$ in latent space such that:

        1.  The Predictor classifies the decoded patient as "Healthy."

        2.  The generated patient remains close to the original data manifold (realistic).



---



## 📊 Key Results



### 1. Performance Metrics

| Metric | Baseline (Threshold 0.5) | Optimized (Threshold 0.12) |

| :--- | :--- | :--- |

| **ROC AUC** | 0.9132 | **0.9132** |

| **Accuracy** | 78.69% | **85.25%** |

| **False Negatives** | 9 (Dangerous) | **3 (Safe)** |

| **Sensitivity** | 74.2% | **91.4%** |



### 2. Interpretability Validation (Latent Space PCA)

We visualized the counterfactual trajectories using PCA.

* **Baseline Method (Orange):** Moves into low-density regions (off-manifold), creating unrealistic "adversarial" examples.

* **Our VAE Method (Green):** Trajectory remains strictly within the high-density patient cluster (gray), ensuring the suggested changes are biologically plausible.



> *Note: See `results/manifold_visualization.png` for the generated plot.*



---



## 🛠️ Installation & Usage



### Prerequisites

* Python 3.8+

* PyTorch (with CUDA support if available)

* Scikit-Learn, Pandas, NumPy, Matplotlib



