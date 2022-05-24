import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000) -> None:
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters

        self.w = None
        self.b = None
    
    def fit(self, X,y):
        y_ = np.where(y <=0, -1, 1)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.w = np.zeros(n_features) # this is not a good way to initialize weights to zero
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx]* (np.dot(self.w, x_i)- self.b) >= 1
                if condition:
                    self.w -= self.learning_rate*2*self.lambda_param*self.w
                else:
                    self.w -= self.learning_rate*(2*self.lambda_param*self.w - y_[idx]*x_i)
                    self.b -=self.learning_rate*y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

if __name__ == "__main__":

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=50, n_features=2, centers =2, cluster_std=1.5, random_state=40)

    y = np.where(y==0, -1, 1)

    svm = SVM()
    svm.fit(X, y)
    



    