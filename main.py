import pandas as pd
import matplotlib.pyplot as plt

def get_input(file_name):
    return pd.read_csv(file_name)

dataset = get_input("bodyPerformance.csv")
x = dataset.weight_kg
y = dataset.height_cm

plt.scatter(x,y)


slope = 0
intercept = 0
learning_rate = 0.00001

iterations = 1_000

number_elements_x = float(len(x))

for i in range(iterations):
    y_prediction = slope * x + intercept
    derivate_slope = (-2 / number_elements_x) * sum(x * (y - y_prediction))
    derivate_intercept = (-2 / number_elements_x) * (y - y_prediction)
    slope = slope - learning_rate * derivate_slope
    intercept = intercept - learning_rate * derivate_intercept
    print(slope, intercept)


y_prediction = slope * x + intercept
plt.plot([min(x), max(x)], [min(y_prediction), max(y_prediction)], color='red')

plt.show()
