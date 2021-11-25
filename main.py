import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv('bodyPerformance.csv')

def error_function(x, y, slope, intercept):
    '''Defining error function'''
    n = float(len(x))
    total_error = 0.0
    for i in range(n):
        total_error += (y[i] - (slope * x[i] + intercept))**2
    return total_error / n


def get_line(x, y, i = 1000):
    '''getting the regression line'''
    slope  = 3          #Como fazer a ligação entre este slope e o definido na função error_function
    intercept = 7       # "" p/ intercept

    learning_rate = 0.0001
    iterations = i

    n = float(len(x))

    for i in range(iterations):
        y_prediction = slope * x + intercept
        derivate_slope = (-2/n) * sum(x *(y - y_prediction))
        derivate_intercept = (-2/n) * (y - y_prediction)
        slope = slope - learning_rate * derivate_slope
        intercept = intercept - learning_rate * derivate_intercept

    y_prediction = slope * x + intercept

    plt.scatter(x,y)
    plt.plot([min(x), max(x)], [min(y_prediction), max(y_prediction)], color='red')


def predict_value(x, y, value_x):
    '''Returning predicted dependent variable value for given independent variable value'''
    x_mean = x.mean()

    ds['diff_x'] = x_mean - x             #creating new column and getting differences

    ds['diff_x_squared'] = ds.diff_x ** 2 #squaring the differences
    ssxx = ds.diff_x_squared.sum()

    y_mean = y.mean()
    ds['diff_y'] = y_mean - y

    ssxy = (ds.diff_x*ds.diff_y).sum()

    slope = ssxy/ssxx

    intercept = y_mean - slope * x_mean

    predict = slope * value_x + intercept
    plt.plot(value_x, predict, 'go')

get_line(ds.weight_kg, ds.height_cm)
predict_value(ds.weight_kg, ds.height_cm, range(50, 85, 2))


plt.show()


