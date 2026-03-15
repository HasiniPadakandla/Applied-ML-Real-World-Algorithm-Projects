import joblib
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import load_data


def plot_feature_importance():

    df = load_data("data/housing.csv")

    model = joblib.load("models/linear_regression_model.pkl")

    features = df.drop("SalePrice", axis=1).columns

    importance = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    })

    importance = importance.sort_values(by="Coefficient")

    plt.barh(importance["Feature"], importance["Coefficient"])
    plt.title("Feature Importance")
    plt.show()


if __name__ == "__main__":
    plot_feature_importance()