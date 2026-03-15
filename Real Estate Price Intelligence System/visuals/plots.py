import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def price_distribution(df):

    sns.histplot(df["price"], kde=True)

    plt.title("Price Distribution")

    plt.show()


def area_vs_price(df):

    sns.scatterplot(x="area", y="price", data=df)

    plt.title("Area vs Price")

    plt.show()


def correlation_heatmap(df):

    plt.figure(figsize=(10,6))

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

    plt.title("Feature Correlation Heatmap")

    plt.show()