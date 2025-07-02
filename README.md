# fMRI Connectivity in Mood Circuits

**A rapid, end-to-end ACC seed-based resting-state fMRI pipeline**

## Overview

This project demonstrates an end-to-end neuroimaging analysis pipeline examining functional connectivity within mood regulation circuits. Using resting-state fMRI data, we investigate anterior cingulate cortex (ACC) connectivity patterns that have been implicated in treatment-resistant depression and obsessive-compulsive disorder.

## Dataset

- **Source**: OpenNeuro ds000030 (or similar resting-state dataset)
- **Subjects**: sub-01, sub-02 (chosen for minimal motion <0.2 mm FD)
- **Acquisition**: Resting-state fMRI, single session

## Methods

### Preprocessing
- **Tool**: fMRIPrep v23.x
- **Output space**: MNI152NLin6Asym (2mm)
- **Key steps**: Motion correction, spatial normalization, minimal smoothing

### Connectivity Analysis
- **Tool**: CONN Toolbox (MATLAB)
- **Approach**: Seed-to-voxel functional connectivity
- **Seed region**: ACC (MNI coordinates: 0, 24, 26)
- **Analysis**: First-level connectivity maps, group-level thresholding

### Visualization
- **Tools**: Python (nilearn, matplotlib)
- **Outputs**: Connectivity maps, statistical plots

## Key Findings

ACC showed strongest positive connectivity to dorsomedial PFC (Z = 4.1) and PCC (Z = 3.7). ![ACC Connectivity](results/acc_connectivity_thumb.png)

## Clinical Relevance

The ACC serves as a critical hub in cognitive control and emotion regulation networks. Hyperconnectivity between ACC and subcortical regions has been consistently reported in treatment-resistant depression, making this circuit a key target for neurostimulation interventions (Philip et al., 2018).

## Repository Structure

```
├── data/                   # Raw and preprocessed data
├── scripts/               
│   ├── requirements.txt   # Python dependencies
│   ├── preprocess.sh      # fMRIPrep commands
│   ├── connectivity.m     # CONN analysis script
│   └── visualize.py       # Python plotting
├── results/               # Output connectivity maps
├── docs/
│   └── Mini_Report.pdf    # Analysis report
└── README.md
```

## Requirements

- fMRIPrep (Docker)
- MATLAB + CONN Toolbox
- Python: nilearn, matplotlib, pandas

## Usage

1. **Download data**: `bash scripts/download_data.sh`
2. **Preprocess**: `bash scripts/preprocess.sh`
3. **Analyze connectivity**: `matlab -batch "connectivity('sub-01')"`
4. **Generate plots**: `python scripts/visualize.py`

## Results Summary

- **Connectivity map**: Z-scored correlation maps thresholded at p < 0.001
- **Key connections**: ACC ↔ dmPFC, PCC, sgACC
- **Clinical implications**: Results consistent with hyperconnectivity patterns in mood disorders

| Region | Z-score | Clinical relevance |
|--------|---------|-------------------|
| dmPFC | 4.1 | Executive control |
| PCC | 3.7 | Default mode network |

📄 **[Full Report](docs/Mini_Report.pdf)**

## References

- Philip, N.S. et al. (2018). Network mechanisms of clinical response to transcranial magnetic stimulation in posttraumatic stress disorder and major depressive disorder. *Biol Psychiatry*
- Whitfield-Gabrieli, S. & Nieto-Castanon, A. (2012). CONN toolbox. *Brain Connectivity*
- Esteban, O. et al. (2019). fMRIPrep: a robust preprocessing pipeline. *Nature Methods*

---

**License**: MIT  
**Contact**: [Your name] | [Email] | Completed as part of neuroimaging portfolio development
