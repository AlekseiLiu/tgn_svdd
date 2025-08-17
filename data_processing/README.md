# data_processing/

Scripts and configs for one-time data preparation.

- `cic_2017_preprocess.py`: Preprocess raw CIC-IDS2017 data to temporal event CSVs

## Dataset Download

Download the CIC-IDS2017 dataset from the official source:

**🔗 Dataset URL:** https://www.unb.ca/cic/datasets/ids-2017.html

You'll need the following PCAP files:
- `Monday-WorkingHours.pcap` (baseline/normal traffic)
- `Tuesday-WorkingHours.pcap` (SSH-Patator, FTP-Patator)
- `Wednesday-WorkingHours.pcap` (DoS attacks, Heartbleed)
- `Thursday-WorkingHours.pcap` (Web attacks, Infiltration)
- `Friday-WorkingHours.pcap` (Botnet, PortScan, DDoS)

Extract all PCAP files to a directory (e.g., `/path/to/cic2017/pcap/`).

## Environment Setup

Choose one of the following methods to set up the data processing environment:

### Conda Environment (Recommended)

Create an isolated conda environment with all required packages:

```bash
cd data_processing
bash install_data_env.sh
conda activate tgn_data
```

This creates a dedicated `tgn_data` environment with pandas, numpy, scikit-learn, and nfstream.

### Pip Installation (Alternative)

Install packages directly into your current Python environment:

```bash
cd data_processing
pip install -r requirements.txt
```

### NFStream notes
- We use NFStream for NetFlow extraction; if you encounter install/runtime issues, see the official docs: https://www.nfstream.org/
- Some Linux setups may require additional system packages (see NFStream docs for details).

## Usage

Once your environment is ready, you can run the preprocessing script:

```bash
python cic_2017_preprocess.py --help
python cic_2017_preprocess.py --raw-data-dir /path/to/pcap/files --verbose
```

