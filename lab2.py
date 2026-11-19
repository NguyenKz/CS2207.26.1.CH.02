import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def run():
    # generate synthetic data
    n = 100
    mu = 0
    std = 1  # standard deviation
    x = np.random.normal(loc=mu, scale=std, size=(n, 1))
    
    # we want to the following hypotheses
    # H_0: mu = 0   vs.   H_1: mu != 0

    # construct test-statistic
    
    T = np.mean(x)
    
    cdf = norm.cdf(T,loc=0,scale=np.sqrt(1/n))
    
    p_value = 2 * (1 - cdf)
    
    return p_value


if __name__ == '__main__':

    # Exercise 1: Your code to compute p_value
    p_value = run()
    print("p-value:", p_value)

    # Exercise 2: Your code to plot the distribution of the p_value
    p_values = np.array([run() for _ in range(1000)])
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    ax.hist(p_values, bins=np.linspace(0, 1, 21), color='#4C78A8',
            edgecolor='white', linewidth=1.2)
    ax.axhline(1000 / 20, color='#E45756', linestyle='--',
               linewidth=1.5, label='Expected count if uniform')
    ax.set(
        title='P-value distribution (1,000 simulations)',
        xlabel='P-value',
        ylabel='Number of simulations',
        xlim=(0, 1),
    )
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(axis='y', alpha=0.25)
    ax.legend(frameon=False)
    plt.show()
    

    # Exercise 3: Your code to compute the false positive rate with
    # 1000 iterations
    
    alpha = 0.05
    false_positive_rate = np.mean(
        np.array(p_values) < alpha
    )

    print("False positive rate:", false_positive_rate)
    




    