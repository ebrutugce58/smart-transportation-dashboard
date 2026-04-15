import pickle
import random
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "stop_arrivals.csv"
MODEL_PATH = "model.pkl"

# Preferred features (use only those available in the CSV).
PREFERRED_FEATURES = [
    "line_id",
    "stop_id",
    "hour_of_day",
    "day_of_week",
    "weather_condition",
    "traffic_level",
    "cumulative_delay_min",
    "speed_factor",
]
TARGET_COLUMN = "delay_min"


def create_synthetic_dataset(rows=500):
    """Create a simple synthetic dataset for quick model training."""
    line_ids = [f"L{str(i).zfill(2)}" for i in range(1, 21)]
    stop_ids = [f"S{str(i).zfill(3)}" for i in range(1, 81)]
    weather_options = ["clear", "rain", "windy", "fog"]
    traffic_options = ["low", "medium", "high"]

    records = []
    for _ in range(rows):
        hour_of_day = random.randint(0, 23)
        day_of_week = random.randint(0, 6)
        weather_condition = random.choice(weather_options)

        if hour_of_day in (7, 8, 9, 17, 18, 19):
            traffic_level = random.choices(traffic_options, weights=[1, 3, 6])[0]
        elif hour_of_day in (6, 10, 16, 20):
            traffic_level = random.choices(traffic_options, weights=[2, 6, 2])[0]
        else:
            traffic_level = random.choices(traffic_options, weights=[6, 3, 1])[0]

        speed_factor = round(random.uniform(0.7, 1.3), 2)
        cumulative_delay_min = round(random.uniform(0, 20), 1)

        traffic_penalty = {"low": 1.0, "medium": 3.0, "high": 6.0}[traffic_level]
        weather_penalty = {"clear": 0.0, "windy": 1.0, "fog": 2.0, "rain": 2.5}[weather_condition]
        rush_hour_penalty = 2.0 if hour_of_day in (7, 8, 9, 17, 18, 19) else 0.0
        weekend_bonus = -0.5 if day_of_week >= 5 else 0.0
        speed_penalty = (1.0 - speed_factor) * 6.0
        random_noise = random.uniform(-1.5, 1.5)

        delay_min = max(
            0.5,
            round(
                2.0
                + traffic_penalty
                + weather_penalty
                + rush_hour_penalty
                + weekend_bonus
                + 0.35 * cumulative_delay_min
                + speed_penalty
                + random_noise,
                1,
            ),
        )

        records.append(
            {
                "line_id": random.choice(line_ids),
                "stop_id": random.choice(stop_ids),
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "weather_condition": weather_condition,
                "traffic_level": traffic_level,
                "cumulative_delay_min": cumulative_delay_min,
                "speed_factor": speed_factor,
                "delay_min": delay_min,
            }
        )

    return pd.DataFrame(records)


def main():
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        print("stop_arrivals.csv not found, using synthetic dataset.")
        synthetic_df = create_synthetic_dataset(rows=500)
        synthetic_df.to_csv(data_file, index=False)
        print(f"Saved synthetic dataset to {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Dataset must contain 'delay_min' column.")

    feature_columns = [feature for feature in PREFERRED_FEATURES if feature in df.columns]
    if not feature_columns:
        raise ValueError("None of the preferred features were found in the dataset.")

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    categorical_features = [col for col in feature_columns if X[col].dtype == "object"]
    numeric_features = [col for col in feature_columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained successfully. MAE: {mae:.2f}")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump({"model": model, "features": feature_columns}, model_file)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
