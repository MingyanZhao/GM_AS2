import numpy as np

def normalizeStd(X):
    means = X.mean(0)
    stds = X.std(0)
    X[:, 1:] = (X[:, 1:] - means[1:]) / stds[1:]

    return X

def normalizeMax(X):
    maxs = X.max(1)

    X[1:] = X[1:] / maxs[1:]

    return X

def normalizeGen(X):
    maxs = X.max(1)
    mins = X.min(1)

    X[1:] = (X[1:] - mins[1:]) / (maxs[1:] - mins[1:])

    return X