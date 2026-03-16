import joblib
from sklearn.linear_model import LinearRegression
from data_preprocessing import load_data, preprocess_data

def train():

    df = load_data("data/housing.csv")

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    model = LinearRegression()

    model.fit(X_train, y_train)

    joblib.dump(model, "models/linear_regression_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_names, "models/features.pkl")

    print("Model trained and saved")

if __name__ == "__main__":
    train()