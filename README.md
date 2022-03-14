# Tabular Data Generation

Out of variety of possible use cases from MIMIC dataset we focus on single use case to predict the number of days a patient stays in the Intensive Care Unit (ICU) to further generate synthetic data. 

This usecase seemed important - it is a benefit for payer to predict the tendency of the length of stay of a patient. It helps in changing the premium charged by the payer according to the comparison of predictions and baseline (the defined no. of days covered by a particular plan of the patient). For this use case the model utilises the total number of diagnosis that occured for different disease category for each patient.

## Dataset

[MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) contains deidentiﬁed data of 383,220 patients admitted to an intensive care unit (ICU) or the emergency department (ED) between 2008 - 2019. The latest version of MIMIC-IV is v0.4 and only provides public access to the electronic health record data of 50,048 patients admitted to the ICU, which is sourced from the clinical information system MetaVision at the BIDMC. 

### Accessing MIMIC
Researchers seeking to use the database must:
- Become a credentialed user on PhysioNet. This involves completion of a training course in human subjects research.
- Sign the data use agreement (DUA). Adherence to the terms of the DUA is paramount.
- Follow the tutorials for direct cloud access (recommended), or download the data locally.

### Difficulties with MIMIC
Data is hard to preprocess - data cleaning procedure need to handle: 

(1) Inconsistent units. 

(2) Multiple recordings at the same time. 

(3) Range of feature values. We use the median of the range as thevalue of the feature.

### Data preparation

Predicting the length of stay in ICU can be based on 4 tables from MIMIC-IV database:

- `patients`: Every unique patient in the database (defines subject_id) Columns like: 'subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod' (count of rows: 382278, count of columns: 6)
- `admissions`: Every unique hospitalization for each patient in the database (defines hadm_id) Columns like: 'subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'admission_location', 'discharge_location', 'insurance', 'language', 'marital_status', 'ethnicity', 'edregtime', 'edouttime', 'hospital_expire_flag' (count of rows: 523740, count of columns: 15)
- `icustays`: Every unique ICU stay in the database (defines icustay_id) Columns like: 'subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'last_careunit', 'intime', 'outtime', 'los' (count of rows: 76540, count of columns: 8)
- `diagnosis_ICD`: Hospital assigned diagnoses, coded using the International Statistical Classification of Diseases and Related Health Problems (ICD) system Columns like: 'subject_id', 'hadm_id', 'seq_num', 'icd_code', 'icd_version' (count of rows: 5280351, count of columns: 5)

More about data preparation and EDA based on chosen features can be found [HERE](./preprocessing/LOS-real-dataset-preparation.ipynb).

## GANs trainings

The [Synthetic Data Vault (SDV)](https://github.com/sdv-dev/SDV) is a Synthetic Data Generation ecosystem of libraries that allows users to easily learn single-table, multi-table and time series datasets to later on generate new Synthetic Data that has the same format and statistical properties as the original dataset.

We tried with CTGAN architecture from the framework.

Package Reference:  [ctgan](https://pypi.org/project/ctgan/)

Documentation : https://sdv-dev.github.io/CTGAN/

Github: https://github.com/sdv-dev/CTGAN

We trained the model for 100 epochs only as the discriminator and generator loss becomes quite low after these many epochs.

Then, we generated 100 000 rows of synthetic data and compare its distribiution with real.

Script to run experiments can be found [HERE](./train.py).

## Synthetic data evaluation

We did a feature by feature comparision between the generated data and the actual data. We used python’s [table_evaluator](https://pypi.org/project/table-evaluator/) library to compare the features.

We call the `visual_evaluation` method to compare the actual data(data) and the generated data(samples).

As its apparent from the visualizations, the similarity between the original data and the synthetic data is quite high. The results give a lot of confidence as we took a random dataset and applied the default implementation without any tweaks or any data preprocessing.

Notebook to run experiments can be found [HERE](./evaluation.ipynb).

## Subsequent task - regresion

Subsequently, we can use original train data and synthetic train data to predict case Length of Stay.
This is regresion task, in which algorithms like Regression Tree, Random Forest, XGBoost, Support Vector Machine and K-Nearest Neighbor, can be used.

We tried with simple neural network:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_15 (Dense)             (None, 100)               4300      
_________________________________________________________________
dense_16 (Dense)             (None, 50)                5050      
_________________________________________________________________
dense_17 (Dense)             (None, 50)                2550      
_________________________________________________________________
dense_18 (Dense)             (None, 50)                2550      
_________________________________________________________________
dense_19 (Dense)             (None, 1)                 51        
=================================================================
Total params: 14,501
Trainable params: 14,501
Non-trainable params: 0
_________________________________________________________________
```

We compared metrics as: Mean Squared Error, Root Mean Squared Error, Mean Absolute Error, Mean Absolute Percentage Error.

|     Dataset     | RMSE |  MSE  |  MAE |
|:---------------:|:----:|:-----:|:----:|
| orginal         | 4.05 | 16.41 | 2.44 |
| synthetic CTGAN | 7.05 | 49.75 | 5.04 | 

Script to run experiments can be found [HERE](./nn.py).
