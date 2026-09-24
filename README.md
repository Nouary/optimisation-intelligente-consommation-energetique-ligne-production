# Optimisation intelligente de la consommation énergétique d’une ligne de production

🧠 **Machine‑learning powered optimisation** of energy consumption for a steel‑line production process.

The repository contains the full data‑science pipeline, from exploratory data analysis to a predictive model and a downstream optimisation routine, together with a Streamlit web app to explore results interactively.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Getting Started](#getting-started)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Dependencies](#dependencies)
- [License](#license)

---

## Project Overview
The **Steel Line Energy** dataset (`data/steel_line_energy_dataset.csv`) records 15‑minute energy consumption together with operational parameters (temperature, load, etc.).
The pipeline consists of:

1. **EDA** – quick visual inspection (`01_eda.py`).
2. **Pre‑processing** – train/test split, feature engineering (`02_preprocessing.py`).
3. **Modeling** – baseline regression model (`03_modeling.py`).
4. **Prediction & Optimisation** – custom optimisation script (`04_prediction_and_optimization.py`).
5. **Streamlit UI** – interactive dashboard (`app.py`).

---

## Folder Structure
```
.
├─ energy_ml_project/
│   ├─ data/                         # CSV dataset(s)
│   ├─ models/                       # Saved model files (if any)
│   ├─ pages/                        # Optional Streamlit pages
│   ├─ 01_eda.py
│   ├─ 02_preprocessing.py
│   ├─ 03_modeling.py
│   ├─ 04_prediction_and_optimization.py
│   └─ app.py                        # Streamlit entry point
├─ README.md                        # ← this file
└─ LICENSE                          # MIT licence
```

---

## Getting Started
### 1️⃣ Clone the repo
```bash
git clone https://github.com/Nouary/optimisation-intelligente-consommation-energetique-ligne-production.git
cd optimisation-intelligente-consommation-energetique-ligne-production
```
### 2️⃣ Set up a Python environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell
pip install --upgrade pip
pip install -r requirements.txt   # (see next section for a sample requirements file)
```
### 3️⃣ Run the pipeline (quick test)
```powershell
python energy_ml_project\01_eda.py
python energy_ml_project\02_preprocessing.py
python energy_ml_project\03_modeling.py
python energy_ml_project\04_prediction_and_optimization.py
```
---

## Running the Streamlit App
```powershell
streamlit run energy_ml_project\app.py
```
Open the URL shown by Streamlit (usually `http://localhost:8501`) in a browser.

---

## Dependencies
| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Modelling |
| `streamlit` | Web UI |
| `matplotlib` / `seaborn` | Plotting |
| *(add any other libraries you use)* | |

You can generate a full list with `pip freeze > requirements.txt` after installing the needed packages.

---

## License
This project is released under the **MIT License** – see the `LICENSE` file for details.
