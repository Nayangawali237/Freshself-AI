import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    svp = 0.61078 * np.exp((17.27 * data["storage_temp"]) / (data["storage_temp"] + 237.3))
    avp = svp * (data["relative_humidity"] / 100.0)
    data["vpd_kpa"] = np.round(svp - avp, 3)
    data["thermal_stress"] = np.where(data["storage_temp"] > 8.0, data["storage_temp"] - 8.0, 0.0)
    return data

def build_and_train():
    if not os.path.exists("produce_shelf_life.csv"):
        import generate_data
        generate_data.generate_produce_dataset().to_csv("produce_shelf_life.csv", index=False)

    raw_df = pd.read_csv("produce_shelf_life.csv")
    df = add_engineered_features(raw_df)

    X = df.drop(columns=["shelf_life_days"])
    y = df["shelf_life_days"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    categorical_cols = ["produce_type", "ethylene_exposure", "packaging_type"]
    numeric_cols = ["initial_maturity", "storage_temp", "relative_humidity", 
                    "transit_days", "vpd_kpa", "thermal_stress"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42)
    }

    best_model_name = None
    best_score = float("inf")
    best_pipeline = None

    for name, regressor in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"[{name}] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

        if mae < best_score:
            best_score = mae
            best_model_name = name
            best_pipeline = pipeline

    os.makedirs("models", exist_ok=True)
    export_path = os.path.join("models", "freshshelf_pipeline.joblib")
    bundle = {
        "pipeline": best_pipeline,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "best_model_name": best_model_name
    }
    joblib.dump(bundle, export_path)
    print(f"Saved optimal model ({best_model_name}) to {export_path}")

if __name__ == "__main__":
    build_and_train()
