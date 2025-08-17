# TGN-SVDD: Temporal Graph Networks + Deep SVDD (CIC-IDS2017)

Minimal, reproducible pipeline for unsupervised intrusion detection on dynamic network graphs. The method combines a Temporal Graph Network (TGN) encoder with Deep SVDD anomaly scoring.

## Overview
TGN-SVDD detects intrusions on dynamic graphs by combining a TGN encoder with Deep SVDD anomaly scoring. Trained on normal traffic only and evaluated on CIC-IDS2017 Monday+X day-pairs, it outperforms vanilla TGN and classic baselines under both feature-rich and featureless settings.

Two-phase workflow:
- Data processing (one-time): PCAP → temporal CSVs
- Main experiment: Train/evaluate TGN-SVDD on processed data

## Prerequisites
- Linux
- Conda (Miniconda/Anaconda)
- Two separate environments are used:
  - Data processing: Python 3.8 (env name: `tgn_data`)
  - Main experiment: Python 3.10 (env name: `tgn_svdd_main`)
- GPU optional (CPU works; GPU recommended for speed)
- For data processing: NFStream

## 1) Data processing (one-time)

You can find more information in data_processing/README.md

Download CIC-IDS2017 PCAPs from the official source and convert to CSVs.

```bash
# 1) Create data env and install deps
cd data_processing
bash install_data_env.sh
conda activate tgn_data

# 2) Convert PCAPs → CSV
python cic_2017_preprocess.py --raw-data-dir /path/to/cic2017/pcap --verbose
```

Outputs:
- CSVs under `data/cic_2017_processing/cic_2017_monday_0_0_1/` with files like:
   - `monday_friday_workinghours.csv`
   - `monday_thursday_workinghours.csv`
   - `monday_tuesday_workinghours.csv`
   - `monday_wednesday_workinghours.csv`
- CSV format: `[src_ip_id, dst_ip_id, timestamp_ms, label, attack_name, 61_features...]`


## 2) Main experiment setup
Install the main environment from the repo root.

```bash
# From repo root
bash install_main_env.sh  # creates/activates a conda env and installs src deps
# If the script didn’t auto-activate, run:
# conda activate tgn_svdd_main
```

## 3) Run a minimal experiment
Run from repository root using the module entrypoint, or from `src/` directly.

```bash
# Option A (recommended): from repo root
python -m src.main --quick-test --epochs 3 --verbose

# Option B: from src directory
cd src
python main.py --quick-test --epochs 3 --verbose
```

This performs a short run on a single day (`monday_friday_workinghours`) and writes outputs to `results/TIMESTAMP_*`.

### Common options
- `--day monday_friday_workinghours` to run a specific day
- `--epochs 30` to change epochs
- `--batch-size 200` and `--lr 0.0001` to override training params
- `--results-dir results` to change output directory
- `--save-checkpoints` to write model checkpoints

## Output
- Results and plots: `results/TIMESTAMP_*` 
- Logs: printed to console and saved alongside results

## Troubleshooting
- PyTorch + PyG: the install script installs PyTorch via conda and the rest via pip from `src/requirements.txt` (which includes `torch-geometric`). If installing manually, ensure versions are compatible with your CUDA.
- CUDA version: the script installs `pytorch-cuda=12.1` when a GPU is detected. If your system supports a different CUDA (e.g., 11.8), edit `install_main_env.sh` to match your driver, or run CPU-only.
- Data path: ensure `--raw-data-dir` points to the folder containing the CIC-IDS2017 PCAPs.


## Citation
If you use this repository, please cite the associated paper:

```bibtex
@inproceedings{liuliakov2023one,
  title={One-Class Intrusion Detection with Dynamic Graphs},
  author={Liuliakov, Aleksei and Schulz, Alexander and Hermes, Luca and Hammer, Barbara},
  booktitle={International Conference on Artificial Neural Networks},
  pages={537--549},
  year={2023},
  organization={Springer}
}
```

## License
MIT License. See `LICENSE`.
