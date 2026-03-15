import joblib
import numpy as np


def predict(area, bedrooms, bathrooms):

    model = joblib.load("models/linear_regression_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    data = np.array([[area, bedrooms, bathrooms]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    return prediction[0]


if __name__ == "__main__":

    price = predict(2000, 3, 2)

    print("Predicted Price:", price)