# 🌱 Carbon Footprint Optimization in Supply Chain Logistics

This project uses deep learning to optimize delivery routes in logistics by minimizing **carbon emissions**, helping companies adopt environmentally sustainable practices.

---

## 🚀 Objectives

- Predict carbon emissions using route and cargo data
- Suggest routes that are fuel-efficient and eco-friendly
- Provide insights using evaluation metrics and visualizations

---

## 📦 Features Used

- `route_distance` (km)  
- `fuel_used` (liters)  
- `weather_index`  
- `traffic_index`  
- `cargo_weight` (kg)

---

## 🧠 Tech Stack

- **Python**  
- **TensorFlow/Keras** (Deep Learning)  
- **Pandas, NumPy** (Data Handling)  
- **Scikit-learn** (Evaluation)  
- **Matplotlib** (Visualization)

---

## 🔁 Workflow

1. **Data Collection** – logistics, weather, traffic data  
2. **Preprocessing** – normalization, missing values  
3. **Model Training** – deep learning regression model  
4. **Evaluation** – MAE, RMSE, % Error  
5. **Route Suggestion** – selects route with least emissions

---

## 📁 Project Files

- `data1.csv.xlsx` – dataset file  
- `carbon-footprints.py` – model and route logic  
- `carbon_emission_model.h5` – trained model
- 'lib'-https://drive.google.com/drive/folders/1vHYQJzIHt205BCgvmhBih9PygyCluIZW?usp=sharing
- `README.md` – project guide

---

## ▶️ How to Run

1. **Install Libraries:**
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib openpyxl
