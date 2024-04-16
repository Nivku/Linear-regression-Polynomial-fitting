import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import matplotlib.pyplot as plt







def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename,parse_dates=["Date"])

    # remove duplicates and Nan values
    df = df.dropna()
    df = df.drop_duplicates()

    # Date checking
    df = df[df['Year'].isin(range(2024))]
    df = df[df['Month'].isin(range(1, 13))]
    df = df[df['Day'].isin(range(1, 32))]

    df = df[df['Temp'] > - 10]
    df = df[df['Temp'] < 50]

    df["DayOfYear"] = df["Date"].dt.dayofyear

    return df





if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")


    # Question 2 - Exploring data for specific israel
    df_is = df[df["Country"] == "Israel"]
    years = df_is["Year"].unique()
    for year in (years):
        df_year = df_is[df_is["Year"] == year]
        plt.scatter(df_year["DayOfYear"], df_year["Temp"], label=year,s=1,)

    # Add a title and axis labels
    plt.title("Daily Temperature in israel by day")
    plt.xlabel("DayofYear")
    plt.ylabel("Temperature")
    # Add a legend
    plt.legend(title="Year")
    plt.legend(loc='right')
    # Show the plot
    plt.show()
    plt.close()


    #Std by month for israel
    df_is_month = df_is.groupby('Month').agg('std')
    plt.bar(np.array(range(1,13)),df_is_month['Temp'])
    plt.xlabel("Month")
    plt.ylabel("std")
    plt.title("The std of the temperature over years for every month in israel")
    plt.show()


    # Question 3 - Exploring differences between countries
    df_country_month = df.groupby(['Country', 'Month'], as_index=False).agg({'Temp': ['mean', 'std']})
    df_country_month.columns = df_country_month.columns.droplevel(level=0)
    # Rename remaining column level
    df_country_month = df_country_month.set_axis(['Country', 'Month', 'mean', 'std'], axis=1)
    fig = px.line(df_country_month, x ='Month', y ='mean', error_y='std', color='Country')
    fig.show()




    # Question 4 - Fitting model for different values of `k` for israel

    train_X, train_y, test_X, test_y = split_train_test(df_is["DayOfYear"],df_is["Temp"],0.75)
    loss_arr = np.zeros(10)
    min = None
    for i in range(1,11):
        fitted_model = PolynomialFitting(i)
        fitted_model.fit(train_X.to_numpy(), train_y.to_numpy())
        k_loss = fitted_model.loss(test_X,test_y)
        loss_arr[i-1] = np.round(k_loss,2)
        print(f"The MSE of {i} degree: {np.round(k_loss,2)}")
    plt.bar(np.array(range(1, 11)), loss_arr)
    plt.xlabel("K degree")
    plt.ylabel("MSE")
    plt.title("The MSE as function of K degree for israel")
    plt.show()



    # Question 5 - Evaluating the MSE for different countries using the k-degree of israel

    best_k = np.argmin(loss_arr) + 1
    fitted_model = PolynomialFitting(best_k)
    fitted_model.fit(train_X.to_numpy(), train_y.to_numpy())
    country_loss = []
    countries = df["Country"].unique()
    for country in countries:
        df_country = df[df["Country"] == country]
        train_X, train_y, test_X, test_y = split_train_test(df_country["DayOfYear"], df_country["Temp"], 0.75)
        loss = fitted_model.loss(test_X,test_y)
        country_loss.append(loss)
        print(f"The MSE of {country} is: {np.round(loss,2)}")
    plt.bar(countries,country_loss)
    plt.xlabel("country")
    plt.ylabel("MSE")
    plt.title("The MSE of the k model on different countries")
    plt.show()












