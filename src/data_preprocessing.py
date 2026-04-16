import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Remove missing values
    df = df.dropna()

    label_encoders = {}

    # Encode categorical columns
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, "outputs/scaler.pkl")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test