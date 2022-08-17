class LinearRegressor():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.W = [1,2]
        self.b = 0

        self.m = len(X)
        
        for _ in range(self.iterations):
            self._update_weights()
        return self

    # Update weights in gradient descent
    def _update_weights(self):
        Y_pred = self.predict(self.X)

        Y_residual = []
        for i in range(self.m):
            Y_residual.append(Y_pred[i] - self.Y[i])

        dW = []
        dW_sum0 = dW_sum1 = 0
        for i in range(self.m):
            dW_sum0 += self.X[i][0] * Y_residual[i]
            dW_sum1 += self.X[i][1] * Y_residual[i]
        dW0 = dW_sum0 * (2 / self.m)
        dW1 = dW_sum1 * (2 / self.m)
        dW = [dW0, dW1]

        db =  (2 / self.m) * self._sum(Y_residual) 

        # Update weights
        self.W = [self.W[0] - self.learning_rate * dW[0], self.W[1] - self.learning_rate * dW[1]]
        self.b = self.b - self.learning_rate * db

        return self

    # Returns sum of the array
    def _sum(self, array):
        sum = 0
        for i in array:
            sum += i
        return sum

    def predict(self, X):
        Y_pred = []
        for i in range(len(X)):
            Y_pred.append(X[i][0] * self.W[0] + X[i][1] * self.W[1] + self.b)
        return Y_pred