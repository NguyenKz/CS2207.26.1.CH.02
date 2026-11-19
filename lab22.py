import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(2025)

def generate_data(n, p, true_beta):
    X = np.random.normal(loc=0, scale=1, size=(n, p))
    true_beta = np.reshape(true_beta, (p, 1))
    noise = np.random.normal(loc=0, scale=1, size=(n, 1))

    y = np.dot(X, true_beta) + noise

    return X, y

def run():
    n = 100
    p = 1
    true_beta = [0.0]
    X, y = generate_data(n, p, true_beta)

    # Your code here to compute the p-value

    

if __name__ == '__main__':
    
    p_value = run()
    print("p-value:", p_value)