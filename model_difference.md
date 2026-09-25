# Model Difference

- The public research-use web calculator reports the final eight-variable CatBoost model described in the BMC MIDM submission manuscript.
- The deployed model uses the eight harmonized first-day ICU variables listed in the manuscript.
- The benchmark panel summarizes external-validation performance for Admission platelet alone, Admission platelet + hemoglobin + creatinine, and Very simple thrombocytopenia clinical model.
- Benchmark rows are shown as study-level external-validation summaries rather than bedside recalculations.
- Public deployment uses the CatBoost model's raw probability and the locked Youden threshold of 0.0605, without a post-calibrator.
