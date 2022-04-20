import numpy as np
import scipy.stats


def get_steady_state(markov_matrix):
    n = markov_matrix.shape[0]
    aux = np.vstack((markov_matrix - np.eye(n), np.ones((1, n))))
    return np.linalg.solve(aux.T @ aux, np.ones(n))


def compute_log_binomial_probability_matrix(ntrials, probabilities, observations):
    """
    Computes the logarithm of binomial probabilities of each pair of probabilities and observations.
    :param ntrials: number of binomial trials
    :param probabilities: vector with probabilities
    :param observations: vector with integers (observations)
    :return log_matrix: |probabilities| x |observations| matrix with the log binomial probabilities
    """
    probabilities = np.array(probabilities)
    if any(probabilities > 0):
        probabilities[probabilities == 0] = min(probabilities[probabilities > 0]) / 100  # To avoid numerical errors. An error would mean the adversary information is very off.
    else:
        probabilities += 1e-10
    column_term = np.array([np.log(probabilities) - np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    last_term = np.array([ntrials * np.log(1 - np.array(probabilities))]).T  # COLUMN TERM
    log_matrix = np.array(observations) * column_term + last_term
    return log_matrix


def compute_log_binomial_plus_laplacian_probability_matrix(ntrials, probabilities, observations, lap_mean, lap_scale):
    def prob_laplacian_rounded_up_is_x(mu, b, x):
        if x <= mu:
            return 0.5 * (np.exp((x - mu) / b) - np.exp((x - 1 - mu) / b))
        elif mu < x < mu + 1:
            return 1 - 0.5 * (np.exp(-(x - mu) / b) + np.exp((x - 1 - mu) / b))
        else:
            return 0.5 * (np.exp(-(x - 1 - mu) / b) - np.exp(-(x - mu) / b))

    pmf_discrete_laplacian = [prob_laplacian_rounded_up_is_x(lap_mean, lap_scale, x) for x in range(int(2 * lap_mean) + 1)]
    log_matrix = np.zeros((len(probabilities), len(observations)))
    for i_row, prob in enumerate(probabilities):
        pmf_binomial = scipy.stats.binom(ntrials, prob).pmf(range(ntrials))
        pmf_sum = np.convolve(pmf_binomial, pmf_discrete_laplacian)
        log_matrix[i_row, :] = [np.log(pmf_sum[obs]) if pmf_sum[obs] > 0 else np.nan_to_num(-np.inf) for obs in observations]
    return log_matrix


def compute_log_binomial_with_power_rounding(ntrials, probabilities, observations, x):
    log_matrix = np.zeros((len(probabilities), len(observations)))
    round_limits = [0] + [x ** i for i in range(int(np.ceil(np.log(ntrials) / np.log(x))) + 1)]
    for i_row, prob in enumerate(probabilities):
        pmf_binomial = scipy.stats.binom(ntrials, prob).pmf(range(ntrials))
        pmf_rounded_dict = {round_limits[i]: sum(pmf_binomial[round_limits[i - 1] + 1:round_limits[i] + 1]) for i in range(1, len(round_limits))}
        log_matrix[i_row, :] = [np.log(pmf_rounded_dict[obs]) if pmf_rounded_dict[obs] > 0 else np.nan_to_num(-np.inf) for obs in observations]
    return log_matrix


def compute_pancake_parameters(nkw, true_dist):
    replicas_per_kw = np.ceil(true_dist * nkw)
    replicas_per_kw = np.append(replicas_per_kw, 2 * nkw - np.sum(replicas_per_kw)).astype(np.int64)

    prob_reals = np.append(true_dist, 0)
    if replicas_per_kw[-1] == 0:
        prob_dummies = np.append(replicas_per_kw[:-1] / nkw - prob_reals[:-1], 0)
    else:
        prob_dummies = replicas_per_kw / nkw - prob_reals

    return prob_reals, prob_dummies, replicas_per_kw
