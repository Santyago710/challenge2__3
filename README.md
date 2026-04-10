# Environment Setup

## Create a virtual environment and install dependencies.

python -m venv .venv

source .venv/bin/activate

# Install required packages:

pip install -r requirements.txt


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
│
├── src
│ └── preprocessing.py
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
  - `Geo Join ID`
  - `Message`

**Temporal Feature Engineering**

- extraction of:
  - `year`
  - `month`
  - `day_of_week`

**Missing Value Handling**

- missing pollutant measurements are filled using the **mean of the same pollutant category**

**Outlier Removal**

- outliers are filtered using the **Interquartile Range (IQR) method**

**Categorical Encoding**

- One-Hot Encoding applied to:
  - Measure
  - Geo Type Name
  - Geo Place Name
  - Time Period

**Target Encoding**

- pollutant names are encoded using **LabelEncoder**

**Feature Scaling**

- pollutant measurement values normalized using **StandardScaler**

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
| X_unlabeled.pkl | Unlabeled data for semi-supervised learning |
| X_test.pkl | Test features |
| y_test.pkl | Test labels |

---
## Running the Pipeline
Preprocessing

python src/preprocessing.py


### 1. Load Training Data

Load the datasets using Python:

```python
import pickle

with open("data/X_labeled.pkl", "rb") as f:
    X_labeled = pickle.load(f)

with open("data/y_labeled.pkl", "rb") as f:
    y_labeled = pickle.load(f)