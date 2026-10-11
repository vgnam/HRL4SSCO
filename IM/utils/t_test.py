from scipy import stats


def t_test(mean1, std_dev1, n1, mean2, std_dev2, n2):
    """
    Calculate the p-value for a two-sample independent t-test.
    
    Parameters:
    mean1 (float): Mean of sample 1
    std_dev1 (float): Standard deviation of sample 1
    n1 (int): Sample size of sample 1
    mean2 (float): Mean of sample 2
    std_dev2 (float): Standard deviation of sample 2
    n2 (int): Sample size of sample 2

    Returns:
    p_value (float): Two-tailed p-value for the t-test
    """
    # Calculate the standard error for samples 1 and 2
    se = ((std_dev1 ** 2) / n1 + (std_dev2 ** 2) / n2) ** 0.5

    # Calculate the t-statistic
    t_statistic = (mean1 - mean2) / se

    # Calculate the degrees of freedom
    df = n1 + n2 - 2

    # Use scipy.stats to calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

    return p_value
