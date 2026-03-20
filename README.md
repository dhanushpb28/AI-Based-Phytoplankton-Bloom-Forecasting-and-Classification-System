# 🌊 AI-Based Phytoplankton Bloom Forecasting & HAB Risk Prediction System

## Overview

This project presents an AI-integrated framework for monitoring, forecasting, and analyzing phytoplankton blooms using satellite-derived oceanographic data.

The system leverages spatio-temporal deep learning (ConvLSTM) to forecast chlorophyll concentration and bloom dynamics, combined with an XGBoost-based machine learning model to classify harmful algal bloom (HAB) risk levels.

To improve interpretability, SHAP (SHapley Additive exPlanations) is used to identify key environmental drivers influencing bloom formation. The entire pipeline is integrated into an interactive Streamlit dashboard for real-time visualization and analysis.

<br>

---

## 🚀 Features

- 🌍 Satellite-based ocean monitoring (Copernicus Marine data)
- 📈 Chlorophyll forecasting using ConvLSTM
- ⚠️ HAB risk classification using XGBoost
- 🧠 Explainable AI using SHAP
- 🗺️ Bloom detection using threshold-based mapping
- 📊 Environmental variable analysis & correlation
- 📉 Temporal trend visualization
- 🖥️ Interactive Streamlit dashboard
- ☁️ AWS S3 integration for scalable data storage

<br>

---

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras (ConvLSTM)
- XGBoost
- SHAP
- Streamlit
- NumPy, Pandas, Xarray
- Matplotlib / Seaborn
- AWS S3 (data storage)
- NetCDF4 (data handling)

<br>

---

## 📂 Dataset

- Source: **Copernicus Marine Environment Monitoring Service (CMEMS)**
- Format: NetCDF
- Variables Used:
  - Chlorophyll concentration (chl)
  - Phytoplankton biomass (phyc)
  - Primary productivity (nppv)
  - Nitrate (no3)
  - Phosphate (po4)
  - Sea surface temperature anomaly
  - Ocean currents (uo, vo)

<br>

---
## 🔑 Environment Variables Setup (Required)

This project requires credentials for Copernicus Marine data access and AWS S3 integration.

### 1. Create a `.env` file in the project root:

```bash
COPERNICUS_USERNAME=your_username
COPERNICUS_PASSWORD=your_password

AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=your_region
```

## ⚙️ Installation & Setup

1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Based-Phytoplankton-Bloom-Forecasting-and-Classification-System.git
```
2. Navigate into the Project Folder
    ```sh
    cd ai-travel-planner
    ```
3. Create Virtual Environment:
    ```sh
    python -m venv venv
    ```
4. Activate Virtual Environment:

    Windows
    ```sh
    venv\Scripts\activate
    ```
    macOS/Linux
    ```sh
    source venv/bin/activate
    ```

5. Install Dependencies:
    ```sh
    pip install -r requirements.txt
    ```
6. Run the Application:
    ```sh
    streamlit run app.py
    ```
The application will open automatically in your browser.

<br>

## How It Works
1. ### Data Acquisition

- Fetch oceanographic datasets from Copernicus Marine

- Store datasets in NetCDF format (locally / AWS S3)

2. Data Preprocessing

- Remove land pixels using ocean mask

- Apply log transformation for skewed variables

- Normalize features

- Align all variables to a common grid

3. Forecasting (ConvLSTM)

- Input: Previous 4 days of environmental data

- Output: Next 2 days chlorophyll prediction

- Spatial attention improves bloom localization

4. HAB Risk Classification

- XGBoost model classifies:

    - No Risk

    - Moderate Risk

    - High Risk

5. Explainability (SHAP)

- Identifies key environmental drivers:

    - Nutrients

    - Temperature anomaly

    - Primary productivity

6. Visualization (Streamlit)

- Bloom maps

- Forecast maps

- Environmental trends

- Interactive controls

<br>
