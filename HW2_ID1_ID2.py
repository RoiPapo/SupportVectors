import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


class PerceptronClassifier:
    def __init__(self):
        """
        Constructor for the PerceptronClassifier.
        """
        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (316327451, 206230021)
        self.W = 0
        self.number_of_clases = 0;

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This method trains a multiclass perceptron classifier on a given training set X with label set y.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
        Array datatype is guaranteed to be np.uint8.
        """
        # rand = np.random.RandomState(self.random_state)
        ones = np.ones((X.shape[0], 1))
        X = np.c_[X, ones]
        self.numer_of_classes = len(np.unique(y, return_counts=False))
        Wmat = np.zeros((self.numer_of_classes, X.shape[1] - 1))
        ones = np.ones((self.numer_of_classes, 1))
        self.W = np.c_[Wmat, ones]
        convergence = 0
        while convergence == 0:
            convergence = 1
            for j, example in enumerate(X):
                maxArg = [0, 0]
                for i, label in enumerate(np.unique(y, return_counts=False)):
                    curr_y = np.inner(example, self.W[i])
                    if curr_y > maxArg[0]:
                        maxArg[0] = curr_y
                        maxArg[1] = label
                if maxArg[1] != y[j]:
                    self.W[y[j]] = self.W[y[j]] + example
                    self.W[maxArg[1]] = self.W[maxArg[1]] - example
                    convergence = 0
        # print(self.W)

    # self.W = np.zeros(X.shape[1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call PerceptronClassifier.fit before calling this method.
        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
        Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """
        ones = np.ones((X.shape[0], 1))
        X = np.c_[X, ones]
        Prediction = []
        for j, example in enumerate(X):
            maxArg = [0, 0]
            for i in range(self.numer_of_classes):
                curr_y = np.dot(example, self.W[i])
                if curr_y > maxArg[0]:
                    maxArg[0] = curr_y
                    maxArg[1] = i
            Prediction.append(maxArg[1])
        return np.array(Prediction)

        ### Example code - don't use this:
        # return np.random.randint(low=0, high=2, size=len(X), dtype=np.uint8)


if __name__ == "__main__":
    print("*" * 20)
    print("Started HW2_ID1_ID2.py")
    # Parsing script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', type=str, help='Input csv file path')
    args = parser.parse_args()

    print("Processed input arguments:")
    print(f"csv = {args.csv}")

    print("Initiating PerceptronClassifier")
    model = PerceptronClassifier()
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {args.csv}...")
    data = pd.read_csv(args.csv, header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    is_separable = model.fit(X, y)
    print("Done")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y.ravel()) / y.shape[0]
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)
