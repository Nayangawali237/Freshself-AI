# 🥗 FreshShelf AI Dashboard

**Automated shelf-life estimation, biochemical decay analysis, and dynamic markdown triage.**

FreshShelf AI is a predictive analytics and inventory management tool designed for the retail grocery and agricultural supply chain sectors. It transforms fresh produce management from a reactive process into a proactive, data-driven strategy. By predicting decay before it happens, it helps store managers take immediate action to reduce food waste and maximize revenue.

## ✨ Key Features

*   **🤖 AI Shelf-Life Prediction:** Uses a Machine Learning regression model to estimate exactly how many days a batch of produce has left before spoiling.
*   **🚥 Dynamic Triage System:** Automatically categorizes inventory into 4 states (Optimal, Moderate, Urgent, Critical) and generates step-by-step action plans for staff.
*   **💰 Economic Salvage Estimator:** Calculates potential financial waste and advises when to apply 30%-50% markdowns to recover lost revenue.
*   **🌡️ Physical Stress Indicators:** Calculates the Vapor Pressure Deficit (VPD) and Thermal Stress to warn users if the storage room is too hot or too dry.
*   **🧠 Explainable AI (SHAP):** Features an interactive SHAP waterfall chart that visually explains exactly *why* the AI made its prediction, building trust with non-technical users.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **Machine Learning:** Scikit-Learn (Random Forest Regressor)
*   **Data Processing:** Pandas, NumPy
*   **Explainable AI:** SHAP
*   **Containerization:** Docker

## 📂 Project Structure

```text
FreshShelf/
│
├── app.py                  # Main Streamlit dashboard application
├── train_model.py          # Script to generate and train the ML model
├── requirements.txt        # Python package dependencies
├── Dockerfile              # Docker container configuration
├── README.md               # Project documentation
└── models/                 # Directory where the trained .joblib model is saved