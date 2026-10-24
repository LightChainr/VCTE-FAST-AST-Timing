# VCTE-FAST-AST-Timing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23612516.svg)](https://doi.org/10.5281/zenodo.23612516)

Analysis code and aggregate results for:

> Yu H-w, Ying H, Zhang Z, Qiao C, Dong C. *Aspartate aminotransferase from another date reclassifies FibroScan-AST more often than Agile: reference change values identify results at risk.* Manuscript submitted to Clinical Biochemistry (2026).

The FibroScan-aspartate aminotransferase (FAST) score combines liver stiffness (LSM) and controlled attenuation parameter (CAP) from vibration-controlled transient elastography with AST. In routine practice the AST is often measured on another date. This study compared FAST calculated with the nearest earlier or later AST (≤90 days) with FAST from same-day AST for the same scan, repeated the comparison for Agile 3+, Agile 4 and FIB-4, and evaluated a laboratory flag based on the reference change value (RCV) of AST.

## Contents

| Path | Description |
|---|---|
| `code/00_link_vcte_lis.py` | Deterministic in-hospital linkage of elastography records to the laboratory information system |
| `code/01_rebuild_and_analyze.py` | AST selection, paired comparisons with same-day AST, patient-level bootstrap, models |
| `code/02_focused_sensitivity.py` | Sensitivity analyses (first scan per patient, unique AST, CK, HBV DNA, interval bands, etiology) |
| `code/02b_rcv_threshold.py` | Scan-specific AST thresholds for FAST 0.35 and 0.67 and the RCV flag |
| `code/02c_clinical_actions.py` | Changes in the implied clinical action |
| `code/03_build_figures.py`, `code/03b_graphical_abstract.py` | Figures and graphical abstract |
| `code/08_hybrid_score_timing.py` | Agile 3+, Agile 4 and FIB-4 on the same comparisons (`results/29_*`; added in v1.1.0) |
| `code/09_flag_benchmark.py` | Post hoc: RCV flag versus simpler rules at equal workload, chemistry line, selection into paired comparisons (`results/30_*`; added in v1.1.0) |
| `results/` | Aggregate result tables (CSV, JSON, TSV) behind every number in the manuscript |
| `results/external_reference/` | FAST perturbation results in six public cohorts |

Scripts are run in the order listed. Bootstrap and cross-validation seeds are fixed in the scripts.

## Data availability

Patient-level data are not included because they contain confidential patient information. The scripts expect the frozen analysis dataset `DS-VCTE-FAST-V13-20261003` from the institutional research data repository, whose location is read from the environment variable `VCTE_FAST_WAREHOUSE`. The dataset is available from the corresponding author upon reasonable request with approval of the Ethics Committee of Shanghai Pudong Hospital. The aggregate results in `results/` contain no patient-level records.

## Environment

Python 3.11.15; numpy 2.3.5; pandas 2.3.3; scipy 1.16.3; scikit-learn 1.9.0; statsmodels 0.14.6; matplotlib 3.10.0.

## Citation

Please cite the archived release. v1.1.2 (Clinical Biochemistry submission): DOI to be added after archiving. v1.1.1: https://doi.org/10.5281/zenodo.23612516. v1.1.0: https://doi.org/10.5281/zenodo.23611397. All versions: https://doi.org/10.5281/zenodo.23319354. v1.0.0: https://doi.org/10.5281/zenodo.23319355.

## Versions

- v1.1.2 (2026-10-24): manuscript title revised; graphical abstract title updated accordingly. Analysis scripts and all results are unchanged from v1.1.0.
- v1.1.1 (2026-10-24): figure scripts only. Main figures reorganized (Figure 1: direction of FAST change and the four-score comparison; Figure 2: how the RCV flag works, its performance and the reporting steps); former panels moved to Supplementary Figs. S5 and S10; graphical abstract simplified. Analysis scripts and all results are unchanged from v1.1.0.
- v1.1.0 (2026-10-24): adds the Agile 3+, Agile 4 and FIB-4 comparison (amendment 29) and post hoc checks of the RCV flag (amendment 30); figures and banner graphical abstract updated; FAST formula written with the natural logarithm (ln), as in the original derivation.
- v1.0.0 (2026-10-12): FAST analyses.

## License

Code: MIT (`LICENSE`). Aggregate results: CC BY 4.0 (`LICENSE-DATA`).

## Contact

Chang Dong, Department of Clinical Laboratory, Shanghai Pudong Hospital (Fudan University Pudong Medical Center), Shanghai, China.

Analysis code was written with the assistance of Claude Code (Anthropic) and was reviewed and run by the authors.
