## Experiment table:
Model: Logistic Regression
Notes: Baseline, little preprocessing
Accuracy: 0.851
F1 (overall): 0.57
AUC: 0.84
Key: catches ~46% of defaulters, moderate precision

Model: Random Forest
Notes: 300 trees, basic params
Accuracy: 0.929
F1 (overall): 0.81
AUC: 0.93
Key: catches ~70% of defaulters, very high precision (0.96),
     about 300 more defaulters caught vs baseline, fewer false positives