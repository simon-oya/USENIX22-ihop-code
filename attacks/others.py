import numpy as np
from processing.process_obs import process_traces, compute_Vobs
from processing.process_aux import get_Vexp
from scipy.optimize import linear_sum_assignment as hungarian


def umemaya_attack(obs, aux, exp_params):

    def_params = exp_params.def_params
    ndocs = obs['ndocs']
    token_trace, token_info = process_traces(obs, aux, def_params)
    Vobs = compute_Vobs(obs['trace_type'], token_info, ndocs)
    Vexp = get_Vexp(aux, exp_params.def_params)

    # Umemaya
    eigs, u_train = np.linalg.eig(Vexp)
    idx = eigs.argsort()[::-1]
    u_train = u_train[:, idx]
    eigs, u_test = np.linalg.eig(Vobs)
    idx = eigs.argsort()[::-1]
    u_test = u_test[:, idx]
    new_c_matrix = np.abs(u_train) @ np.abs(u_test).T
    row_vals, col_vals = hungarian(-new_c_matrix)

    keyword_predictions_for_each_token = {}
    for tag, keyword in zip(col_vals, row_vals):
        keyword_predictions_for_each_token[tag] = keyword
    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]

    return keyword_predictions_for_each_query


def _fastPFP(mode, n, n2, matrices, alpha=0, step=0.5, threshold_global=1e-10, threshold_proj=1e-10, maxiter_global=200, maxiter_proj=200, verbose=False):
    """Solves (1-alpha)||A-P A' P^T||+alpha tr(P^T C) using the fastFPF algorithm"""

    def project_into_partially_double_stochastic(Y, threshold_proj, maxiter_proj):
        n = Y.shape[0]
        Yold = Y + 1 + threshold_proj
        niter_proj = 0
        while np.max(np.abs(Y - Yold)) > threshold_proj and niter_proj < maxiter_proj:
            Yold = Y
            Y = Y + (1 + np.sum(Y) / n - Y.sum(0) - Y.sum(1).reshape(n, 1)) / n
            Y = (Y + np.abs(Y)) / 2
            niter_proj += 1
        return Y, niter_proj

    assert mode in ('mse', 'ml')
    assert n2 <= n
    if mode == 'mse':
        assert matrices[0].shape == (n, n)
        assert matrices[1].shape == (n2, n2)
        assert matrices[2].shape == (n, n2)
    elif mode == 'ml':
        assert all(group[0].shape == (n, n) and group[1].shape == (n2, n2) for group in matrices)

    # n = A.shape[0]
    # n2 = Ap.shape[0]  # np < n
    lamb = 2*alpha/(1-alpha)

    X = np.ones((n, n2)) / (n * n2)
    Y = np.zeros((n, n))
    Xold = (threshold_global + 1) + np.ones((n, n2))
    niter_global = 0

    while np.max(np.abs(X-Xold)) > threshold_global and niter_global < maxiter_global:
        if mode == 'mse':
            Y[:n, :n2] = matrices[0] @ X @ matrices[1] + lamb * matrices[2]
        elif mode == 'ml':
            aux = - np.sum([group[0] @ X @ group[1] + group[0].T @ X @ group[1].T for group in matrices], 0)
            Y[:n, :n2] = aux
        Y, niter_proj = project_into_partially_double_stochastic(Y, threshold_proj, maxiter_proj)
        Xold = X
        X = (1 - step) * Xold + step * Y[:n, :n2]
        X = X / np.max(X)
        niter_global += 1
    P = X
    if verbose:
        print("epsilon1 = {:e}, niters1={:d}".format(np.max(np.abs(X-Xold)), niter_global))
    return P


def _greedy_assignment(X):
    """A simple greedy algorithm for the assignment problem as
    proposed in the paper of fastPFP. It creates a proper partial
    permutation matrix (P) from the result (X) of the optimization
    algorithm fastPFP.
    Taken from: https://github.com/emanuele/fastPFP/blob/master/fastPFP.py
    """
    XX = X.copy()
    min = XX.min() - 1.0
    P = np.zeros(X.shape)
    while (XX > min).any():
        row, col = np.unravel_index(XX.argmax(), XX.shape)
        P[row, col] = 1.0
        XX[row, :] = min
        XX[:, col] = min

    return P


def fastfpf_attack(obs, aux, exp_params):

    def_params = exp_params.def_params
    ndocs = obs['ndocs']
    token_trace, token_info = process_traces(obs, aux, def_params)
    Vobs = compute_Vobs(obs['trace_type'], token_info, ndocs)
    Vexp = get_Vexp(aux, exp_params.def_params)
    nkw = len(aux['keywords'])
    ntok = len(token_info)

    P = _fastPFP('mse', Vexp.shape[0], Vobs.shape[1], (Vexp, Vobs, np.zeros((ntok, nkw))))
    P = _greedy_assignment(P)
    keyword_predictions_for_each_token = {}
    for token in range(ntok):
        keyword_predictions_for_each_token[token] = np.where(P[:, token])[0][0]
    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]

    return keyword_predictions_for_each_query
