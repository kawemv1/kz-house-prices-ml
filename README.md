# 🏠 Kazakhstan Apartment Price Predictor (ML Project)

Predict apartment prices (₸) across **Kazakhstan** using **Machine Learning**.  
This project uses real housing data and a **CatBoost Regressor** model to estimate apartment prices based on **city**, **district**, **area**, and other property features.

![Housing Banner](https://img.shields.io/badge/ML-CatBoost-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![GitHub last commit](https://img.shields.io/github/last-commit/kawemv1/kz-house-prices-ml?color=blue)

---

## ✨ Overview

This project aims to help **buyers**, **sellers**, and **real estate analysts** understand apartment pricing trends across Kazakhstan.  
It predicts the **total price** (₸) or **price per square meter** given key input features.

### 🎯 Example use case
> “If I live in *Алматы*, in the *Медеуский район*, and want a 68 m² apartment with 3 rooms on the 9th floor — how much will it cost?”

👉 The model estimates:  
**🏠 ≈ 42,286,000 ₸**

---

## 🧠 Model Summary

| Metric | Result |
|---------|---------|
| **Model** | CatBoostRegressor |
| **R² Score** | 0.859 |
| **MAE** | 3.48 million ₸ |
| **RMSE** | 5.44 million ₸ |
| **Training Data Size** | ~700 listings |
| **Features Used** | `city`, `microdistrict`, `area`, `rooms`, `floor`, `total_floors` |

📈 The model explains ~86% of price variance — excellent for real estate predictions with limited data.

---

## 📊 Features

- 🏙️ City & Microdistrict based prediction  
- 📐 Supports area (m²), rooms, floors, and building height  
- 🤖 Powered by **CatBoost** for high accuracy with categorical features  
- 💡 Outputs both **total price** and **price per m²**  
- 🧩 Includes visualization of price distributions by city  

---

## 🧪 Example Prediction (from Colab)

```python
import pandas as pd
from catboost import CatBoostRegressor

# Load your trained model
model = CatBoostRegressor()
model.load_model("kz_apartment_model.cbm")

# Example input
new_flat = pd.DataFrame([{
    'city': 'Алматы',
    'microdistrict': 'Медеуский р-н',
    'area': 68,
    'rooms': 3,
    'floor': 9,
    'total_floors': 9
}])

predicted_price = model.predict(new_flat)[0]
print(f"🏠 Estimated price: {predicted_price:,.0f} ₸")
