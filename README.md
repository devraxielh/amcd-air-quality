# amcd-air-quality

Code for the paper:

> **Latent Emission Sources Can Mislead Causal Discovery: An Autonomic Procedure for Deciding When to Correct, Evaluated on Air-Quality Data**
> Rodrigo Garcia, Adrian Doria, Jessica Polo — Universidad del Sinú

AMCD (autonomic Monitor–Analyze–Plan–Execute causal discovery) wraps the PC algorithm in a MAPE-K loop. It tests for a latent source mixture with diagonal Gaussian mixture models selected by the BIC. If a mixture is found, it removes the estimated source means by two-fold cross-fitted residualization before running PC. If not, it passes the data to PC unchanged.

This repository reproduces every table and figure in the paper. A single command runs the whole pipeline.

## Repository structure

```text
amcd-air-quality/
├── preprocess_pipeline.py        # cleaning of the observed (Kaggle) file
├── data_raw/                     # place air_quality_dataset.csv here (see Data)
├── data_external/beijing/raw/    # place the extracted Beijing CSV files here (see Data)
├── dataset_ml_ready.csv          # cleaned observed file (generated, see step 2)
└── scripts/
    ├── 00_config_and_utils.py    # paths, seeds, AMCD and all baseline implementations
    ├── utils.py
    ├── 01_data_splitting_and_prep.py   # scenarios D_semi, D_real, D_null and 50 subsamples (Table 3, Fig. 4)
    ├── 02_baseline_and_sota_models.py  # descriptive comparison (Table 4, Fig. 5)
    ├── 03_ablation_and_sensitivity.py  # ablation and sensitivity (Table 8, Fig. 8)
    ├── 04_statistical_tests.py         # Friedman, Nemenyi, Wilcoxon-Holm, McNemar (Table 5, Fig. 6)
    ├── 05_explainability_and_error.py  # MAPE-K loop internals and error analysis (Figs. 9-10)
    ├── 06_robustness_generators.py     # confirmatory study, conditions C0-C7 (Tables 6-7, Fig. 7)
    ├── beijing_data.py                 # Beijing loading, daily aggregation and sampling
    ├── 07_real_data_validation.py      # Beijing, meteorology observed (Tables 9-10, Fig. 11)
    ├── 08_real_latent_regime.py        # Beijing, meteorology hidden (Tables 11-12, Fig. 12)
    ├── run_all_experiments.py          # runs 01-08 twice and checks reproducibility
    ├── results/   tables/   figures/   # outputs (regenerated)
    └── experiment_manifest_and_captions.md
```

## Requirements

Python 3.13. The versions used for the paper are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data

Both datasets are public. They are not redistributed here, so download them before running the pipeline.

1. **Observed file (descriptive scenarios).** Air Quality Dataset on Kaggle:
   <https://www.kaggle.com/datasets/srisyra02/air-quality-dataset>.
   Save it as `data_raw/air_quality_dataset.csv`.
   The paper explains why this file is treated only as a calibrated testbed: its content does not match its public description, and it is very likely synthetic.

2. **Beijing Multi-Site Air Quality (real-data validation).** UCI Machine Learning Repository, dataset 501 (CC BY 4.0),
   DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G):

   ```bash
   mkdir -p data_external/beijing/raw
   curl -L -o data_external/beijing/beijing_multisite.zip \
     "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
   unzip data_external/beijing/beijing_multisite.zip -d data_external/beijing/raw
   unzip data_external/beijing/raw/PRSA2017_Data_20130301-20170228.zip -d data_external/beijing/raw
   ```

   The archive used in the paper has SHA-256 `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`.
   The twelve station files must end up in `data_external/beijing/raw/PRSA_Data_20130301-20170228/`.

## Reproducing the results

```bash
# 1. Clean the observed file (writes dataset_ml_ready.csv)
python -c "import pandas as pd; from preprocess_pipeline import deterministic_clean; \
deterministic_clean(pd.read_csv('data_raw/air_quality_dataset.csv')).to_csv('dataset_ml_ready.csv', index=False)"

# 2. Run the full pipeline (two complete runs + reproducibility check)
cd scripts
python run_all_experiments.py
```

`run_all_experiments.py` deletes `results/`, `tables/` and `figures/` and runs scripts 01–08 twice. It then checks that every table, figure and result file was produced and that the two runs are bit-identical (wall-clock times excluded).

Each script can also be run on its own, in numerical order, from inside `scripts/`.

**Runtime.** One full run took about five minutes (302 s and 308 s for the two runs) on an Apple M5 laptop (10 cores, 16 GB RAM, macOS 27.0.1).

**Determinism.** The global seed is 42, and descriptive replicate *r* uses seed 42 + *r*. The independently generated datasets use seed *b* + 100*c* + *r*, with *b* = 10,000 for the development block and *b* = 20,000 for the confirmatory test block. The Beijing replicates use seeds 30,000 + *r* (meteorology observed) and 32,000 + *r* (meteorology hidden). Numerical libraries run single-threaded.

## Outputs

| Output | Paper |
|---|---|
| `tables/table_01` – `table_10` (`.csv` and `.tex`) | Tables 3–12 |
| `figures/fig_01` – `fig_08`, `fig_S1` (`.pdf` and 600-DPI `.png`) | Figures 4–12 |
| `results/*.csv`, `results/amcd_traces.json` | Per-replicate scores and MAPE-K traces |
| `results/exit_criteria.txt` | Reproducibility report |

Figure and table captions, together with the values quoted in the text, are listed in `scripts/experiment_manifest_and_captions.md`.

## Citation

If you use this code, please cite the paper (citation details will be added on publication).
