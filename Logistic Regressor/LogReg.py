import numpy as np  # linear algebra
import matplotlib.pyplot as plt  # visualization import


class LogistRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training
    def fit(self, X, Y):

        self.m, self.n = X.shape

        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    # Helper function to update weights in gradient descent
    def update_weights(self):
        A = self.sigmoid((self.X.dot(self.W) + self.b))

        # calculate gradients
        tmp = (A - self.Y.T)
        tmp = np.reshape(tmp, self.m)
        dW = np.dot(self.X.T, tmp) / self.m
        db = np.sum(tmp) / self.m

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def predict(self, X):
        Z = self.sigmoid((X.dot(self.W) + self.b))
        Y = np.where(Z > 0.5, 1, 0)
        return Y


# Driver code
# 287 1000000
# 286 500000
def main():
    # Importing dataset
    trainData_X = np.loadtxt('train.csv', delimiter=',', dtype=float, skiprows=1, usecols=np.arange(1, 286))
    trainData_Y = np.loadtxt('train.csv', delimiter=',', dtype=float, skiprows=1, usecols=286)

    # testData_X = np.genfromtxt('test.csv', delimiter=',', dtype=float, skip_header=400001, usecols=np.arange(1, 285))
    # testData_Y = np.genfromtxt('test.csv', delimiter=',', dtype=float, skip_header=400001, usecols=285)

    # Splitting dataset into train and test set
    X_train, X_test = trainData_X[:800000, :], trainData_X[800000:, :]
    Y_train, Y_test = trainData_Y[:800000, ], trainData_Y[800000:, ]

    accuracies = []
    epoch = 0
    for epoch in range(50):
        # Model training
        model = LogistRegression(learning_rate=0.01, iterations=epoch + 1)
        model.fit(X_train, Y_train)

        # Prediction on test set
        Y_pred = model.predict(X_test)

        correctly_classified = 0.0
        count = 0
        for count in range(np.size(Y_pred)):
            if Y_test[count] == Y_pred[count]:
                correctly_classified += 1
            count += 1

        accuracy = (correctly_classified / count) * 100
        accuracies.append(accuracy)

    epochs = np.arange(0, 50)
    plt.plot(epochs, accuracies)
    plt.title("Curve plotted using the given points")
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracies")
    plt.show()


if __name__ == "__main__":
    main()
