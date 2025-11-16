# Shadow-Structured Cosmology  
## Lexicographic Coherence and the Geometry of Observation

Repository for the computational and reproducibility assets accompanying:

> **Antonios Valamontes (2025).**  
> *Shadow-Structured Cosmology: Lexicographic Coherence and the Geometry of Observation.*  
> Kapodistrian Academy of Science.

This repo provides:

- Minimal, inspectable **Python implementations** of the key constructions:
  - Lexicographic Coherence Operator (LCO)
  - Shadow Transfer Function \(T_s(k)\)
  - Shadow Optical Metric (SOM) and Void Refractivity Tensor (VRT)
  - Dual geometry (physical + cognitive curvature)
- **Synthetic cosmological data** (void fields, mock spectra, noise realizations)
- **Notebooks and scripts** to reproduce plots and diagnostic quantities.

---

## 1. Repository Layout

See `src/`, `data/`, `configs/`, `notebooks/`, `figures/`, and `tests/` for the main components.

---

## 2. Installation

```bash
git clone https://github.com/<username>/ShadowCosmology-LCO.git
cd ShadowCosmology-LCO
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Quick Start

```bash
jupyter notebook
```

Then open the notebooks in `notebooks/` to reproduce the main diagnostics.

For details on file mapping (paper ↔ code ↔ data), see the paper and inline
documentation in the notebooks and source files.

---

## 4. License and Citation

This repository is released under the MIT License.

If you use this repository, please cite:

> Valamontes, A. (2025).  
> *Shadow-Structured Cosmology: Lexicographic Coherence and the Geometry of Observation.*  
> Kapodistrian Academy of Science.



