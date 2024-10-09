This directory contains source code for Volatile Solids Reduction modelling for Anaerobic Digestion.

The module is located under src, and can be installed by running "python install.py"
The scripts and raw data are located under scripts. These consist in:
Scripts:
- adm1_model (brute force calibration of BMP and k_dis for ADM1 by grid search)
- vsr_empirical_model (Leave One Plant Out training and assessment of linear models with engineered features)
- vsr_eq_1 (Assessment of litterature model outputting VSR = 13.7 log(HRT) + 18.9)
Data:
- all_data.csv: a dataframe with a multi index, containing all columns necessary to train the emprical model and containg the data for ALL digesters
- data/Dig{i}.csv: raw data used for ADM1 modelling for Digester i (one file per Digester).
The VSR column coincides between all_data.csv and the data/Dig{i}.csv files.