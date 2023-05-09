# sjwl
import numpy as np

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)
        self.weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            output = self.predict(inputs)
            error = outputs - output
            adjustment = np.dot(inputs.T, error * self.sigmoid_derivative(output))
            self.weights += adjustment

    def predict(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

if __name__ == "__main__":
    neural_network = NeuralNetwork()
    print("Random starting weights:")
    print(neural_network.weights)

    inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(inputs, outputs, 10000)

    print("New weights after training:")
    print(neural_network.weights)

    print("Predictions for new data:")
    print(neural_network.predict(np.array([[1, 0, 0]])))
