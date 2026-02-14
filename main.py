import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


# ==========================
# 1. LOAD DATA (Memory Safe)
# ==========================

def load_data():
    print("Loading data...")

    train = pd.read_csv("data/train.csv", nrows=800000)
    test = pd.read_csv("data/test.csv")
    stores = pd.read_csv("data/stores.csv")
    oil = pd.read_csv("data/oil.csv")
    holidays = pd.read_csv("data/holidays_events.csv")

    # Convert date
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])
    oil["date"] = pd.to_datetime(oil["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    # Optimize datatypes
    train["store_nbr"] = train["store_nbr"].astype("int16")
    train["family"] = train["family"].astype("category")
    train["sales"] = train["sales"].astype("float32")
    train["onpromotion"] = train["onpromotion"].astype("int16")

    test["store_nbr"] = test["store_nbr"].astype("int16")
    test["family"] = test["family"].astype("category")
    test["onpromotion"] = test["onpromotion"].astype("int16")

    return train, test, stores, oil, holidays


# ==========================
# 2. MERGE EXTERNAL FEATURES
# ==========================

def merge_external(train, test, stores, oil, holidays):
    print("Merging external data...")

    # Merge stores
    train = train.merge(stores, on="store_nbr", how="left")
    test = test.merge(stores, on="store_nbr", how="left")

    # Merge oil
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill()

    train = train.merge(oil, on="date", how="left")
    test = test.merge(oil, on="date", how="left")

    train["dcoilwtico"] = train["dcoilwtico"].ffill()
    test["dcoilwtico"] = test["dcoilwtico"].ffill()

    # Simple holiday feature
    holidays = holidays[holidays["transferred"] == False]
    holidays = holidays[["date", "type"]]
    holidays["is_holiday"] = 1

    train = train.merge(holidays[["date", "is_holiday"]], on="date", how="left")
    test = test.merge(holidays[["date", "is_holiday"]], on="date", how="left")

    train["is_holiday"] = train["is_holiday"].fillna(0)
    test["is_holiday"] = test["is_holiday"].fillna(0)

    return train, test


# ==========================
# 3. FEATURE ENGINEERING
# ==========================

def create_features(df):
    print("Creating features...")

    df = df.sort_values(["store_nbr", "family", "date"])

    # Date features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek

    # Encode categorical
    df["family"] = df["family"].cat.codes
    df["city"] = df["city"].astype("category").cat.codes
    df["state"] = df["state"].astype("category").cat.codes
    df["type"] = df["type"].astype("category").cat.codes

    # Lag feature (only one to stay safe)
    if "sales" in df.columns:
        df["lag_7"] = df.groupby(["store_nbr", "family"])["sales"].shift(7)

    return df


# ==========================
# 4. TRAIN MODEL
# ==========================

def train_model(train):
    print("Training model...")

    # Drop only rows where lag is missing
    train = train.dropna(subset=["lag_7"])

    print("Full train shape after preprocessing:", train.shape)

    # Sort by date for proper time split
    train = train.sort_values("date")

    split_index = int(len(train) * 0.8)

    train_data = train.iloc[:split_index]
    valid_data = train.iloc[split_index:]

    print("Train_data shape:", train_data.shape)
    print("Valid_data shape:", valid_data.shape)

    features = [
        "store_nbr", "family", "onpromotion",
        "year", "month", "day", "dayofweek",
        "city", "state", "type",
        "dcoilwtico", "is_holiday",
        "lag_7"
    ]

    X_train = train_data[features]
    y_train = train_data["sales"]

    X_valid = valid_data[features]
    y_valid = valid_data["sales"]

    # Model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    print("Validation RMSE:", rmse)

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Save metrics
    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Validation RMSE: {rmse}")

    # ----------------------------
    # Validation Plot
    # ----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(y_valid.values[:200], label="Actual")
    plt.plot(preds[:200], label="Predicted")
    plt.legend()
    plt.title("Validation: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("outputs/validation_plot.png")
    plt.close()

    # ----------------------------
    # Feature Importance Plot
    # ----------------------------
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance")

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    return model, features



# ==========================
# 5. CREATE SUBMISSION
# ==========================

def create_submission(model, train, test, features):
    print("Creating submission...")

    # Get last 7 days of sales from train
    last_sales = (
        train.sort_values("date")
        .groupby(["store_nbr", "family"])["sales"]
        .last()
        .reset_index()
    )

    test = test.merge(last_sales, on=["store_nbr", "family"], how="left")

    test["lag_7"] = test["sales"]
    test.drop(columns=["sales"], inplace=True)

    X_test = test[features]

    preds = model.predict(X_test)

    submission = pd.DataFrame({
        "id": test["id"],
        "sales": preds
    })

    os.makedirs("outputs", exist_ok=True)
    submission.to_csv("outputs/submission.csv", index=False)

    print("Submission saved to outputs/submission.csv")


# ==========================
# MAIN
# ==========================

if __name__ == "__main__":

    train, test, stores, oil, holidays = load_data()

    train, test = merge_external(train, test, stores, oil, holidays)

    train = create_features(train)
    test = create_features(test)

    model, features = train_model(train)

    create_submission(model, train, test, features)

    print("All tasks completed successfully.")
