# Traffic Demand Prediction using Stacked Ensemble Learning

## Overview

This project focuses on predicting traffic demand using geospatial, temporal, weather, and road infrastructure data. The solution leverages advanced feature engineering, Out-of-Fold (OOF) target encoding, and a stacked ensemble of gradient boosting models to achieve strong predictive performance on a real-world traffic forecasting dataset.

The final solution combines CatBoost, LightGBM, and XGBoost predictions through a meta-learner and incorporates log-transformed target modeling to improve robustness against demand outliers.

---

## Problem Statement

Accurate traffic demand prediction is critical for:

* Smart city planning
* Traffic signal optimization
* Ride-hailing demand estimation
* Resource allocation
* Congestion management

The objective is to predict normalized traffic demand based on location, time, weather conditions, and road characteristics.

---

## Dataset Features

### Spatial Features

* Geohash
* Latitude
* Longitude
* Geo3
* Geo4
* Geo5

### Temporal Features

* Day
* Hour
* Minute
* Time Slot
* Rush Hour Indicators
* Night Indicators
* Cyclical Time Encoding (Sin/Cos)

### Road Features

* Road Type
* Number of Lanes
* Landmarks
* Large Vehicles

### Environmental Features

* Weather
* Temperature

---

## Feature Engineering

### Geospatial Processing

* Decoded geohashes into latitude and longitude coordinates
* Generated hierarchical geohash representations:

  * Geo3
  * Geo4
  * Geo5

### Time-Based Features

* Hour extraction
* Minute extraction
* Time slots
* Peak traffic indicators
* Night indicators
* Cyclical encoding using sine and cosine transformations

### Target Encoding

Leakage-safe Out-of-Fold target encoding was applied on:

* Geohash
* Geo4
* Geo5

Statistics generated:

* Mean demand
* Demand standard deviation

This allowed the model to capture historical demand behavior while minimizing overfitting.

---

## Model Architecture

### Base Models

#### CatBoost Regressor

* Native categorical handling
* Robust performance on tabular data

#### LightGBM Regressor

* Fast gradient boosting
* Efficient handling of large feature spaces

#### XGBoost Regressor

* Regularized boosting framework
* Strong generalization capability

---

## Stacking Ensemble

Predictions from all base models were used as inputs to a meta-learner.

```text
CatBoost
     \
      \
LightGBM ----> Meta Learner ----> Final Prediction
      /
     /
XGBoost
```

This stacking strategy enabled the system to capture complementary strengths from different boosting algorithms.

---

## Target Transformation

To reduce the impact of extreme demand values:

```python
y = log(1 + demand)
```

Predictions were transformed back using:

```python
demand = exp(prediction) - 1
```

Benefits:

* Reduced influence of outliers
* Improved model stability
* Better leaderboard performance

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* CatBoost
* LightGBM
* XGBoost
* PyGeoHash

---

## Results

### Key Achievements

* Advanced feature engineering pipeline
* Leakage-safe OOF target encoding
* Stacked ensemble learning
* Log-transformed target optimization
* Leaderboard score exceeding 91

---

## Project Structure

```text
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── training.ipynb
│
├── models/
│   └── saved_models/
│
├── submissions/
│   └── submission.csv
│
├── README.md
└── requirements.txt
```

---

## Future Improvements

* Geospatial clustering using DBSCAN/K-Means
* Graph-based road network features
* Real-time traffic forecasting API
* Interactive traffic demand dashboard
* Model deployment using FastAPI



---

## License

This project is intended for educational, research, and competition purposes.
