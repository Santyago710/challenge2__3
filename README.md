# Environment Setup

## Create a virtual environment and install dependencies.
```bash
python -m venv .venv

source .venv/bin/activate
```
# Install required packages:
```bash
pip install -r requirements.txt
```

# Air Quality Machine Learning Challenge

## Project Overview

This project focuses on building a machine learning pipeline to analyze and classify air pollutant measurements using the **Air Quality dataset** obtained from Data.gov.

The dataset contains environmental measurements collected across multiple geographic locations and time periods. The goal is to develop models capable of learning patterns in pollutant data and correctly identifying pollutant categories.

The project is organized as a collaborative workflow where each team member is responsible for a specific stage of the machine learning pipeline.

---

# Project Structure

challenge2__3
│
├── data
│ ├── Air_Quality.csv
│ ├── air_quality_processed.csv
│ ├── X_labeled.pkl
│ ├── X_unlabeled.pkl
│ ├── X_test.pkl
│ ├── y_labeled.pkl
│ └── y_test.pkl
│
├── docs
│ └── features.md
│
├── notebooks
│ └── EDA.ipynb
│ └── semi_supervised.ipynb
│ └── baseline_models.ipynb
│
├── src
│ └── preprocessing.py
│ └── ssl_methods.py
│ └── evaluation.py
│ └── models.py
│
└── README.md

---

# Dataset Description

The dataset contains air pollution measurements collected across different geographic regions.

Each observation includes:

- pollutant type
- measurement value
- location
- measurement period
- timestamp information

Detailed documentation of the dataset features can be found in:
docs/features.md


---

# Work Completed (Person 1)

The following steps have already been completed:

### 1. Exploratory Data Analysis (EDA)

File:
notebooks/EDA.ipynb

Tasks performed:

- dataset exploration
- statistical analysis
- pollutant distribution analysis
- temporal analysis
- outlier visualization
- identification of preprocessing requirements

---

### 2. Data Preprocessing Pipeline

File:
src/preprocessing.py

This script performs the following transformations:

**Data Cleaning**

- removal of irrelevant columns
  - `Unique ID`
  - `Message`
  - `Geo Type Name`
  - `Time Period`
  - `Indicator ID`
  - `Measure`

**Temporal Feature Engineering**

- extraction of:
  - `year`
  - `month`
  - `day_of_week`

**Missing Value Handling**

- missing pollutant measurements are filled using the **mean of the same pollutant category**

**Outlier Removal**

- outliers are filtered using the **Interquartile Range (IQR) method**

**Target Encoding**

- pollutant names are encoded using **LabelEncoder**

---

### 3. Processed Dataset

The preprocessing pipeline generates the final dataset:
data/air_quality_processed.csv

This dataset should be used for model training.

---

# Work To Be Completed (Person 2)

Person 2 is responsible for the **Model Training and Evaluation stage**.

The following datasets are available:

| Dataset | Description |
|------|------|
| X_labeled.pkl | Features for labeled training data |
| y_labeled.pkl | Labels for labeled data |
| X_unlabeled.pkl | Unlabeled features for semi-supervised learning and evaluation |
| y_unlabeled.pkl | Unlabeled labels for semi-supervised learning and evaluation |

---
## Running the Pipeline
Preprocessing

python src/preprocessing.py


### 1. Load Training Data

Load the dataset using Python and do some standardization for the columns we need:

```python
df = pd.read_csv("../data/air_quality_processed.csv")
df = df.drop(columns=["Name","Start_Date"])
scaler=StandardScaler()
columnas=["Geo Join ID","Data Value","year","month","day_of_week"]
df[columnas] = scaler.fit_transform(df[columnas])
```
### 2. Split the dataset and prepare it
We split the dataset into labeled and unlabeled making sure that the labeled date is the 20% of all the dataset and the remaining is in the other set (unlabeled).
Then we split our x data (the features we are going to use for the machine prediction) and y data (our goal), obviously for both types of data (unlabeled and labeled).
And in case we need this date later we put it in a .pkl file with:
```python
with open("../data/X_labeled.pkl", "wb") as f:
    pickle.dump(X_labeled, f)
with open("../data/y_labeled.pkl", "wb") as f:
    pickle.dump(y_labeled, f)
with open("../data/X_unlabeled.pkl", "wb") as f:
    pickle.dump(X_unlabeled, f)
with open("../data/y_unlabeled.pkl", "wb") as f:
    pickle.dump(y_unlabeled, f)
```
### 3. Training The Models
We use the data we already prepared for the train and evaluation of the models but before that we put a different set of seeds that in the end will give us some metrics like:
- The Accuracy
- The Precision
- The Recall
- The F1-Score

That to have a summarized information for later comparition. All this can be found in the notebooks/baseline_models.ipynb

# Work Completed (Person 3) - Semi-Supervised Stage

Implemented the full **modeling, semi-supervised learning, and evaluation** stage.

### 1. Semi-Supervised Notebook

File:
notebooks/semi_supervised.ipynb

Implemented workflow:

- load processed data and required splits (`X_labeled`, `X_unlabeled`, `X_test`, `y_*`)
- auto-generate split files if they are missing
- train supervised baselines for comparison
- train semi-supervised methods:
  - Self-Training (pseudo-labeling)
  - Label Spreading
- evaluate using weighted metrics and save outputs
- generate comparison plot (`data/ssl_comparison.png`)

### 2. Evaluation Utilities

File:
src/evaluation.py

Responsibilities:

- compute standard classification metrics:
  - accuracy
  - weighted precision
  - weighted recall
  - weighted F1
- convert result rows into a sorted DataFrame for reporting

### 3. Model Builders

File:
src/models.py

Implemented model factory functions:

- supervised baseline: `RandomForestClassifier`
- SSL self-training model: `SelfTrainingClassifier` with logistic regression base estimator
- SSL graph model: `LabelSpreading`

This keeps hyperparameters centralized and reusable across notebook and scripts.

### 4. End-to-End SSL Script

File:
src/ssl_methods.py

The script performs end-to-end execution:

- load `data/air_quality_processed.csv`
- validate required columns
- encode categorical features (`get_dummies`)
- scale selected numeric features
- split train/test and labeled/unlabeled sets
- serialize splits to `.pkl`
- train and evaluate:
  - SupervisedBaseline(RandomForest)
  - SemiSupervised(SelfTraining)
  - SemiSupervised(LabelSpreading)
- save final report to `data/ssl_results.csv`

### 5. Outputs

Generated artifacts:

- data/X_labeled.pkl
- data/y_labeled.pkl
- data/X_unlabeled.pkl
- data/y_unlabeled.pkl
- data/X_test.pkl
- data/y_test.pkl
- data/ssl_results.csv
- data/ssl_comparison.png
