import numpy as np
from scipy.stats import shapiro, norm



def berkowitz_test(data, cdf, alpha=0.05):
    """
    Perform the Berkowitz test for goodness of fit of estimated distributions to future returns.

    Parameters:
    - data: array-like, the data to test
    - cdf: list of callable, the CDFs for each sample
    - alpha: float, significance level

    Returns:
    - decision: str, "reject H0" or "fail to reject H0"
    - p_value: float, p-value of the test
    """
    n = len(data)
    u = np.zeros(n)

    for i in range(n):
        u[i] = cdf[i](data[i])

    # Transform to normal
    zs = norm.ppf(u)

    # Perform the Shapiro-Wilk test on the transformed data
    stat, p_value = shapiro(zs)
    decision = "reject H0" if p_value < alpha else "fail to reject H0"

    return decision, p_value


if __name__ == "__main__":
    # Example usage
    data = np.random.normal(0, 1, 1000)
    train_data = data[:800]
    test_data = data[800:]
    berkowitz_data = []
    berkowitz_cdfs = []
    for i in range(len(test_data)):
        curr_data = np.concatenate((train_data[i:], test_data[:i]))
        mu, sigma = norm.fit(curr_data)
        cdf = norm(mu, sigma).cdf
        berkowitz_data.append(test_data[i])
        berkowitz_cdfs.append(cdf)
    print(berkowitz_test(berkowitz_data, berkowitz_cdfs, alpha=0.05))
