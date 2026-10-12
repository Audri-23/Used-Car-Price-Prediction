# Quantifying Used-Car Depreciation Beyond Static Specifications: A Rate-Engineered Machine Learning Approach

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Course](https://img.shields.io/badge/Course-CSE437--Data--Science-informational.svg)]()

> **Course**: CSE437 - Data Science  
> **Dataset**: [Used Car Dataset (Kaggle)](https://www.kaggle.com/datasets/rishabhkarn/used-car-dataset) by Rishabh Karn  
> **Target Variable**: `price_cleaned` (Resale Price in Lakhs INR, where $1\text{ Lakh} = 100,000\text{ INR}$)

---

## 📌 Executive Summary

Traditional vehicle appraisal models evaluate raw, static specifications in isolation—such as total odometer reading, nominal engine displacement, or manufacturing year. This approach fails to capture dynamic physical wear: a car driven 50,000 km in one year experiences vastly different annual strain and depreciation compared to a car driven 50,000 km across a decade.

This project introduces a **rate-engineered machine learning framework** that models dynamic usage intensity, engine power density, registration delays, and brand tiers. Across seven regression algorithms spanning linear baselines and tree ensembles, rate-engineered features yield consistent performance gains, with the **Voting Ensemble** and **Tree Ensembles** achieving an **$R^2 \approx 0.79$** and reducing prediction error to **$\approx 8.14\text{ Lakhs INR}$**.

---

## 📂 Repository Contents

```text
├── Quantifying_Used_Car_Depreciation_Paper.pdf     # Full academic research paper (PDF)
├── Quantifying_Used_Car_Depreciation_Notebook.ipynb  # End-to-end Jupyter Notebook with executable code & plots
├── requirements.txt                                # Python package dependencies
└── README.md                                       # Project documentation
```

---

## 🔬 Methodology & Feature Engineering

### 1. Data Cleaning & Scraping Artifact Resolution
The dataset contains raw web-scraping artifacts, including shifted rows, mixed date formats, and corrupted columns:
- **Structural Alignment**: Shifted rows where `mileage(kmpl)` held engine displacement and `torque(Nm)` held peak power were programmatically parsed and corrected.
- **Year Parsing**: Date formats (e.g., `Month-YY`) were converted to 4-digit Gregorian years (`reg_year_clean`), defaulting to `manufacturing_year` when unrecorded.
- **Horsepower Extraction**: Scaled Brake Horsepower (`real_max_power_bhp`) was parsed and verified against manufacturer specifications.
- **Price Scaling**: Raw INR values were standardized to Lakhs INR ($100,000\text{ INR} = 1.0\text{ Lakh}$).
- **Outlier Capping & Imputation**: Continuous numeric columns were bounded to physical thresholds and imputed using column medians to prevent distributional skewing.

### 2. Rate-Engineered Domain Features
| Feature | Formulation | Domain Rationale |
| :--- | :--- | :--- |
| `vehicle_age` | $\max(2024 - \text{mfg\_year}, 1)$ | Captures physical temporal aging relative to evaluation baseline. |
| `usage_intensity` | $\frac{\text{kms\_driven}}{\text{vehicle\_age}}$ | **Annual wear rate (km/year)**: Distinguishes high-intensity fleet cars from lightly used older vehicles. |
| `power_density` | $\frac{\text{real\_max\_power\_bhp}}{\text{engine(cc)}}$ | Specific engine power efficiency metric ($bhp/cc$) reflecting tuning quality. |
| `registration_delay` | $\text{clip}(\text{reg\_year} - \text{mfg\_year}, 0, 5)$ | Captures showroom holding time and pre-registration delays. |
| `brand_segment` | Luxury vs. Mainstream | Segregates premium prestige brands (*Mercedes-Benz, BMW, Audi, Porsche, etc.*) to model non-linear price drops. |
| `log_transformations`| $\log(1 + x)$ | Compresses long-tailed distributions for skewed numerical attributes. |

---

## 📊 Key Results & Comparative Benchmarks

### 1. Comparative Model Performance (Raw Baseline vs. Rate-Engineered)
All models were evaluated on an **unseen 20% test split** ($N = 222$) with **5-Fold Cross-Validation** on the training split ($N = 886$):

| Model | 5-Fold CV $R^2$ | Raw Test $R^2$ | Engineered Test $R^2$ | $R^2$ Gain (%) | Raw RMSE (Lakhs) | Eng RMSE (Lakhs) | Eng MAE (Lakhs) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Voting Ensemble (RF+ET+XGB+LGB)** | **0.7550** | **0.7182** | **0.7899** | **+9.99%** | **9.43** | **8.14** | **3.46** |
| **XGBoost Regressor** | 0.7381 | 0.6961 | 0.7770 | +11.62% | 9.79 | 8.39 | 3.59 |
| **Extra Trees Regressor** | 0.7659 | 0.7044 | 0.7760 | +10.16% | 9.66 | 8.41 | 3.43 |
| **Random Forest Regressor** | 0.7296 | 0.6483 | 0.7745 | +19.46% | 10.53 | 8.43 | 3.50 |
| **LightGBM Regressor** | 0.7152 | 0.6771 | 0.7642 | +12.88% | 10.09 | 8.62 | 3.81 |
| **Gradient Boosting (GBR)** | 0.7324 | 0.6933 | 0.7312 | +5.47% | 9.84 | 9.21 | 3.66 |
| **Ridge Regression (Baseline)** | 0.6137 | 0.4465 | 0.5770 | +29.24% | 13.21 | 11.55 | 5.67 |

### 2. Domain-Level Feature Ablation Study
Removing top features iteratively from the best-performing model validates the individual importance of engineered variables:

| Removed Feature | Original $R^2$ | Ablated $R^2$ | Drop ($\Delta R^2$) | Impact Rating |
| :--- | :---: | :---: | :---: | :---: |
| `brand_segment` | 0.7745 | 0.6780 | **-0.0965** | **Critical** |
| `vehicle_age` | 0.7745 | 0.7533 | **-0.0212** | **High** |
| `power_density` | 0.7745 | 0.7619 | **-0.0126** | **Moderate** |
| `usage_intensity` | 0.7745 | 0.7638 | **-0.0108** | **Moderate** |
| `real_max_power_bhp` | 0.7745 | 0.7651 | **-0.0094** | **Moderate** |
| `transmission` | 0.7745 | 0.7698 | **-0.0048** | **Minor** |
| `registration_delay` | 0.7745 | 0.7765 | **-0.0019** | **Minor** |

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook Quantifying_Used_Car_Depreciation_Notebook.ipynb
```

---

## 🚀 Running in Google Colab
To run directly in Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `Quantifying_Used_Car_Depreciation_Notebook.ipynb`.
3. The dataset is configured to download automatically via the Google Drive link embedded in Cell 3.
4. Run all cells (`Runtime > Run all`).

---

## 📈 Key Takeaways for Automotive Pricing
1. **Usage Intensity Matters More Than Raw Odometer**: An annualized run rate provides tree models with the context needed to distinguish highway fleet vehicles from low-mileage city commuters.
2. **Prestige Premium Dynamics**: Brand segmenting is the strongest predictor of resale value. Luxury cars experience steeper early depreciation curves (steep non-linear decay in years 1–4) compared to mainstream cars.
3. **Power-to-Displacement Efficiency**: The `power_density` ratio ($bhp/cc$) acts as a strong pricing proxy for engine performance grade and trim level without introducing excessive model variance.
