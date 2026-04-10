# Feature Documentation — Air Quality Dataset

## Overview

This document describes the features used in the Air Quality dataset after preprocessing.

The dataset contains environmental measurements of different air pollutants collected across geographic locations and time periods.

During preprocessing, several transformations were applied including:

- removal of irrelevant columns
- temporal feature engineering
- categorical encoding
- outlier filtering
- feature scaling

These steps ensure that the dataset is suitable for machine learning models.

---

# Original Features

| Feature | Type | Description |
|------|------|------|
| Unique ID | Identifier | Unique identifier for each observation. Removed during preprocessing because it has no predictive value. |
| Indicator ID | Numeric | Identifier of the pollutant indicator. |
| Name | Categorical | Name of the pollutant being measured (PM2.5, NO2, O3, etc.). |
| Measure | Categorical | Measurement type (e.g., Mean). |
| Measure Info | Categorical | Unit of measurement (e.g., mcg/m³, ppb). |
| Geo Type Name | Categorical | Type of geographic area where the measurement was recorded. |
| Geo Join ID | Identifier | Geographic join identifier used for spatial linking. Removed during preprocessing. |
| Geo Place Name | Categorical | Name of the geographic area where the measurement was taken. |
| Time Period | Categorical | Time interval during which the measurement was recorded. |
| Start_Date | Date | Date when the measurement period started. |
| Data Value | Numeric | Recorded pollutant measurement value. |
| Message | Text | Empty column in the dataset. Removed during preprocessing. |

---

# Engineered Features

To improve the predictive power of the dataset, several new features were created.

| Feature | Type | Description |
|------|------|------|
| year | Numeric | Year extracted from the `Start_Date`. |
| month | Numeric | Month extracted from the `Start_Date`. |
| day_of_week | Numeric | Day of the week derived from the `Start_Date`. |

These features allow the model to capture **temporal patterns in pollution levels**.

---

# Target Variable

The target variable corresponds to the pollutant type.

| Feature | Description |
|------|------|
| target | Numerical representation of the pollutant name generated using LabelEncoder. |

Example transformation:

| Pollutant | Encoded Value |
|------|------|
| PM2.5 | 0 |
| NO2 | 1 |
| O3 | 2 |

---

# Preprocessing Transformations

The following preprocessing steps were applied to prepare the dataset for machine learning.

## Column Removal

The following columns were removed because they do not provide predictive information:

- `Unique ID`
- `Geo Join ID`
- `Message`

---

## Missing Value Handling

Missing values in the pollutant measurement column were filled using the **mean value of the same pollutant category**. This approach preserves pollutant-specific distributions and avoids bias introduced by global averages.

---

## Outlier Filtering

Outliers were detected using the **Interquartile Range (IQR) method**.

Values outside the interval:

[Q1 − 1.5 × IQR , Q3 + 1.5 × IQR]


were removed in order to reduce noise caused by extreme sensor readings.

---

## Categorical Encoding

Categorical variables were transformed using **One-Hot Encoding**. The following columns were encoded:

- `Measure`
- `Geo Type Name`
- `Geo Place Name`
- `Time Period`

This transformation allows machine learning models to interpret categorical variables correctly.

---

## Feature Scaling

The pollutant measurement variable (`Data Value`) was standardized using **StandardScaler**.

This ensures that variables operating on different scales do not bias the learning algorithm.

---

# Final Dataset

After preprocessing, the dataset contains:

- cleaned features
- engineered temporal variables
- encoded categorical attributes
- normalized measurement values

The processed dataset is stored in:
data/air_quality_processed.csv
and is used as the input for the machine learning training pipeline.
---