```markdown
# Colibri_GW_Update

This repository contains an updated and reorganized version of the COLIBRI_GW pipeline, designed to perform cosmological forecasts using gravitational-wave standard sirens, galaxy catalogues, and modified gravity extensions. The folder includes Python modules, utility functions, and notebooks used for data processing, theoretical predictions, and parameter inference.

---

## Overview

The Colibri_GW_Update project provides:

- Updated scripts for running the COLIBRI_GW pipeline  
- Interfaces with hi_class, CLASS, and related cosmology tools  
- Functions for computing:
  - background cosmology  
  - transfer functions  
  - matter power spectra  
  - angular power spectra (Cℓ)  
  - cross-correlations between GWs and galaxies  
- Fisher-matrix forecasting tools  
- Notebooks for exploratory analysis and validation  
- Utilities for scanning modified gravity parameter spaces (e.g., αM, αB)

This folder represents a cleaned and updated version of the original development environment.

---

## Repository Structure

```

Colibri_GW_Update/
│
├── notebooks/          # Jupyter notebooks for analysis and testing
├── scripts/            # Core Python scripts for the GW–LSS pipeline
├── utils/              # Helper functions, plotting utilities, data tools
├── parameters/         # YAML/JSON parameter files for pipeline runs
├── hi_class/           # Modified hi_class interface and configs (if present)
├── results/            # Output files, plots, Fisher results
└── README.md           # This documentation

````

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/MicheleSantoni/COLIBRI-GW-update.git
cd COLIBRI-GW-update/Colibri_GW_Update
````

### Install dependencies

```bash
pip install numpy scipy matplotlib h5py astropy
pip install classy-python
```

Or use:

```bash
pip install -r requirements.txt
```

---

## Usage

A typical workflow includes:

1. Configure parameters in the `parameters/` directory
2. Run CLASS or hi_class
3. Generate matter power spectra and transfer functions
4. Compute angular power spectra (Cℓ)
5. Run Fisher forecasts
6. Inspect results in notebooks

Example:

```bash
python scripts/run_pipeline.py --params parameters/default.yml
```

---

## Outputs

The pipeline produces:

* Auto- and cross-spectra (Cℓ)
* Noise terms (shot noise, GW detector noise)
* Source distributions and window functions
* Fisher matrices and parameter forecasts
* Diagnostic plots and intermediate results

Outputs are stored in the `results/` folder.

---

## Contributing

Contributions are welcome. Possible improvements include:

* Enhancements to the hi_class interface
* New forecasting modules
* Additional modified-gravity parameterizations
* Performance optimizations

Please open an issue or submit a pull request.

---

## License

MIT License (or another license of your choice).

---

## Citation

If you use this updated pipeline in academic work, please cite the original COLIBRI_GW project and relevant CLASS/hi_class papers.

```

---

Let me know if you want a shorter, more formal, or more technical version.
```
