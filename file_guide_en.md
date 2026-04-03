# Project File Guide

This document classifies and annotates files in the current project directory to help you quickly identify core files and temporary files.

## 1. Core Code (Core System) —— **[Keep / Do Not Delete]**
These are the foundation for the project to run. Missing them will cause the system to fail.

*   **`src/`**: Source code directory.
    *   `processing/`: Data loading and preprocessing logic.
    *   `models/`: Model definitions (RF, TransECA-Net, Hybrid).
    *   `presentation/`: Dashboard code.
*   **`main.py`**: Project entry file.
*   **`train_stage1.py`**: Stage 1 (filter) training script.
*   **`train_stage2.py`**: Stage 2 (classifier) training script.
*   **`run_pipeline.py`**: End-to-end testing and validation script.
*   **`analyze_results.py`**: Results analysis and plotting script.
*   **`run_dashboard.bat`**: Shortcut script to launch the visualization dashboard.

## 2. Models & Data (Models & Data) —— **[Keep / Core Assets]**
The "brain" and "memory" of the system.

*   **`models_chk/`**: Stores trained model weights (`.joblib`, `.pth`) and preprocessing standards (`preprocessor.joblib`).
*   **`archive/`**: Raw dataset (Parquet files).

## 3. Results & Reports (Results & Reports) —— **[Keep / Assignment Submission]**
Files used to showcase work results.

*   **`Project_Proposal.docx`**: **[Important]** Your project proposal (Word version) for submission.
*   **`data_manifest.md`**: **[Important]** Detailed training data statistics report.
*   **`results/`**: Stores generated charts (confusion matrices, feature importance) and text reports.
    *   `stage1_cm.png`, `stage1_report.txt`, etc.

## 4. Auxiliary Docs (Auxiliary Docs) — **[Reference]**
*   **`notebooks/`**: Documentation hub for the project.
    *   `experimental_design.md/en.md`: Experiment design specifications (CN/EN).
    *   `experiment_report_en.md / zh.md`: Full experiment reports (EN/CN).
    *   `verification_logic_chain.md/en.md`: Evidence chain documentation.
    *   `literature_analysis.md`: Literature review.
    *   `experimentalRank / experimentalRank_en`: Experiment priority rankings.
    *   **`experiments/`**: Detailed per-experiment records (E1, E2, E4, E8, E10, E14, E15, E17) — each documenting objectives, methodology, parameters, results, and analysis.
*   **`REPRODUCIBILITY.md`**: Complete guide for reproducing all experiments.
*   **`requirement_en.txt`**: Original project requirements document.

## 5. Temporary Files (Temporary) —— **[Can Delete / Archived]**
*   **`temp/`**: Intermediate temporary files organized into it just now (such as `read_pdf.py`, `debug_etl.py`, etc.). If you no longer need debugging, you can delete the entire folder.

---
**Recommendation**:
When submitting assignments or backing up, you only need to package files in **Group 1, 2, 3** (the `archive/` dataset is usually too large and does not need to be submitted; just submit code reports and `models_chk/`).
