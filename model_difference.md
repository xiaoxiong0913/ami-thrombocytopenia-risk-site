# Model Difference

- The public research-use web calculator reports the final XGBoost model retained in the strong-revision manuscript workflow.
- The deployed XGBoost model uses harmonized first-day ICU variables derived from the study workflow. Feature set: strong_revision_compact_model.
- The benchmark panel summarizes external-validation performance for Admission platelet alone, Admission platelet + hemoglobin + creatinine, and Very simple thrombocytopenia clinical model.
- Benchmark rows are shown as study-level external-validation summaries rather than bedside recalculations.
- Public deployment reuses the locked post-calibrated runtime bundle so the exported web API preserves manuscript-aligned probability calibration.
