import matplotlib.pyplot as plt
import pandas as pd

class DataInputs:
    """We use that class to store data related to the inputs"""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class LinearRegressionParameters:
    """We use this class to pass the parameters that will impact the Linear Regression calculations"""
    def __init__(self, number_of_iteration, learning_rate, minimal_error):
        self.number_of_iteration = number_of_iteration
        self.learning_rate = learning_rate
        self.minimal_error = minimal_error


class LinearRegressionResults:
    """We use this class to represent the results of  Gradiant Descendent Algorithm
        of the Linear Regression"""
    def __init__(self, final_slope, final_intercept, cost):
        self.final_slope = final_slope
        self.final_intercept = final_intercept
        self.cost = cost


class LinearRegression:
    """We use this class for all the computation of the algorithm"""
    def __compute_cost(self, y, y_approx):
        n = float(len(y))
        return sum([data ** 2 for data in (y - y_approx)]) / n

    def __get_slope_derivative(self, x, y, y_approx):
        n = float(len(y))
        return -(2 / n) * sum(x * (y - y_approx))

    def __get_intercept_derivative(self, y, y_approx):
        n = float(len(y))
        return -(2 / n) * sum(y - y_approx)

    def __get_y_approx(self, x, slope_current, intercept_current):
        return (slope_current * x) + intercept_current

    def __get_updated_slope(self, slope_current, learning_rate, slope_derivative):
        return slope_current - (learning_rate * slope_derivative)

    def __get_updated_intercept(self, intercept_current, learning_rate, intercept_derivative):
        return intercept_current - (learning_rate * intercept_derivative)

    def calculate(self, data_inputs, linear_regression_parameters):
        slope = 0
        intercept = 0
        cost = 0
        for i in range(linear_regression_parameters.number_of_iteration):
            y_approx = self.__get_y_approx(data_inputs.x, slope, intercept)
            slope_derivative = self.__get_slope_derivative(data_inputs.x, data_inputs.y, y_approx)
            intercept_derivative = self.__get_intercept_derivative(data_inputs.y, y_approx)
            slope = self.__get_updated_slope(slope, linear_regression_parameters.learning_rate, slope_derivative)
            intercept = self.__get_updated_intercept(intercept, linear_regression_parameters.learning_rate, intercept_derivative)
            cost = self.__compute_cost(data_inputs.y, y_approx)
            if cost <= linear_regression_parameters.minimal_error:
                break

        return LinearRegressionResults(slope, intercept, cost)

def __get_input(file_name):
    dataset = pd.read_csv(file_name)
    x = dataset.iloc[:, 0]
    y = dataset.iloc[:, 1]
    return DataInputs(x, y)


def main():
    inputs = __get_input("dataset.csv")

    parameters = LinearRegressionParameters(number_of_iteration=1000, learning_rate=0.0001, minimal_error=1)
    print("####### PARAMETERS #######")
    print(f" - number_of_iteration={parameters.number_of_iteration}")
    print(f" - learning_rate={parameters.learning_rate}")
    print(f" - minimal_error={parameters.minimal_error}")

    results = LinearRegression().calculate(inputs, parameters)

    print("\n####### RESULTS #######")
    print(f"lowest cost achieved: {results.cost}")
    print(f"final_slope: {results.final_slope}")
    print(f"final_intercept: {results.final_intercept}")

    plt.scatter(inputs.x, inputs.y)
    linear_outcome = results.final_slope * inputs.x + results.final_intercept
    plt.plot(inputs.x, linear_outcome, '-r')
    plt.show()


if __name__ == "__main__":
    main()
