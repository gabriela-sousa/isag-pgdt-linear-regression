import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv('bodyPerformance.csv')

def get_line(x, y, i = 1000):
    slope  = 0
    intercept = 0

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
    plt.show()




get_line(ds.weight_kg, ds.height_cm, 500)
