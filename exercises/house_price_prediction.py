from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # dummy values
    X['zipcode'].fillna(0,inplace= True)
    X['zipcode'] = X['zipcode'].astype(int)
    X = pd.get_dummies(X, prefix='zipcode', columns=['zipcode'])

    # Operations for both train and test
    X = X.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_lot",], axis=1)

    # Check 1 - negative or null
    if y is None:

        coulms = ['bedrooms', 'bathrooms', 'floors', 'sqft_basement', 'yr_renovated',
              'sqft_living', 'sqft_living15', 'sqft_above']

        for i in coulms:
            X.loc[X[i] < 0, i] = X[i].mean()
            X[i].fillna(X[i].mean(), inplace=True)

    # Check 2 -  specific range of values
        X.loc[~X['waterfront'].isin([0, 1]), 'waterfront'] = df['waterfront'].mean()
        X.loc[~X['view'].isin([0, 1, 2, 3, 4]), 'view'] = df['view'].mean()
        X.loc[~X['grade'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), 'grade'] = df['grade'].mean()
        X.loc[~X['condition'].isin([1, 2, 3, 4, 5]), 'condition'] = df['condition'].mean()

        return X
    else:
        X = X.dropna()
        X = X.drop_duplicates()
        X = X[(X >= 0).all(axis=1)]
        y = y[X.index]
        return X,y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    for col_name, col_data in X.iteritems():
        if "zipcode" in col_name:
            continue
        pearson_cor = np.round(np.cov(col_data, y)[1,0] / (np.std(col_data) * np.std(y)),3)
        plt.scatter(col_data,y, s=1 )
        plt.xlabel(col_name)
        plt.ylabel("Price")
        plt.title(f" Pearson correlation: {pearson_cor}")
        plt.savefig(f"{col_name}_Price_cor.png")
        plt.close()






if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # remove Nan prices and price <= 0
    df = df[df['price'] > 0 ]

    # get the price coulm
    price_col = df['price']

    # remove the price coulmn from df
    df.drop('price', axis=1, inplace=True)

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df, price_col, 0.75)



    #QUESTION 2

    test_X = preprocess_data(test_X)
    train_X,train_y = preprocess_data(train_X, train_y)
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)




    # # Question 3 - Feature evaluation with respect to response

    feature_evaluation(train_X, train_y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    mean_loss  = np.zeros(91)
    std_loss = np.zeros(91)
    for i in range(10,101):
        loss_std_p = np.zeros(10)
        for j in range(10):
            partial_train_X = train_X.sample(frac= i / 100)
            partial_train_y = train_y.loc[partial_train_X.index]
            fitted_model = LinearRegression(include_intercept=True)
            fitted_model.fit(partial_train_X,partial_train_y)
            loss = fitted_model.loss(test_X,test_y)
            loss_std_p[j] = loss
        mean_loss[i - 10] = loss_std_p.mean()
        std_loss[i - 10] = loss_std_p.std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(10, 101)), y=mean_loss +2*std_loss,fill=None,name='mean(loss) +2*std(loss)'))
    fig.add_trace(go.Scatter(x=np.array(range(10, 101)), y=mean_loss - 2 * std_loss, fill='tonexty',name='mean(loss) -2*std(loss)'))
    fig.add_trace(go.Scatter(x= np.array(range(10, 101)), y=mean_loss, fill='none',line=dict(color='black'), name='mean(loss)'))
    fig.update_layout(title='The MSE as function of training precentege')
    fig.update_layout(
        title='The MSE as function of training percentage',
        xaxis=dict(title='The percentage of the training data'),  # Set x-axis label
        yaxis=dict(title='The MSE')  # Set y-axis label
    )

    fig.show()
















