import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data_preprocessing import load_data, preprocess_data
import numpy as np


def evaluate():

    df = load_data("data/housing.csv")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    model = joblib.load("models/linear_regression_model.pkl")

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("R2 Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)


if __name__ == "__main__":
    evaluate()