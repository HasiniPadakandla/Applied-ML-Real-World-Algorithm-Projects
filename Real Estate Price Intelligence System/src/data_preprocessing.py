import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):

    df = df[[
        "Gr Liv Area",
        "Bedroom AbvGr",
        "Full Bath",
        "Garage Cars",
        "Overall Qual",
        "Year Built",
        "Neighborhood",
        "SalePrice"
    ]]

    df = df.fillna(df.median(numeric_only=True))

    # encode locality
    df = pd.get_dummies(df, columns=["Neighborhood"], drop_first=True)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler, X.columns