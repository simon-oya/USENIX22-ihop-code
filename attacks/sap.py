import numpy as np
from processing.process_obs import process_traces, compute_fobs, compute_vobs
from processing.process_aux import get_faux, get_vaux
import utils
from scipy.optimize import linear_sum_assignment as hungarian


def _run_algorithm(c_matrix):
    """Runs the Hungarian algorithm with the given cost matrix
    :param c_matrix: cost matrix, (n_keywords x n_tokens)"""

    row_ind, col_ind = hungarian(c_matrix)

    query_predictions_for_each_tag = {}
    for tag, keyword in zip(col_ind, row_ind):
        query_predictions_for_each_tag[tag] = keyword

    return query_predictions_for_each_tag


def _build_cost_freq(faux, fobs, nqr):
    log_c_matrix = nqr * fobs * np.log(np.array([faux]).T)
    return -log_c_matrix


def _build_cost_vol(vaux, vobs, ndocs_obs, def_params, naive_flag=False):

    nkw = len(vaux)
    counts_obs = np.array([vol * ndocs_obs for vol in vobs])

    # Computing the cost matrix
    if naive_flag or def_params['name'] in ('none'):
        log_prob_matrix = utils.compute_log_binomial_probability_matrix(ndocs_obs, vaux, counts_obs)
    elif def_params['name'] in ('osse', 'clrz'):
        tpr, fpr = def_params['tpr'], def_params['fpr']
        vaux_mod = [prob * (tpr - fpr) + fpr for prob in vaux]
        log_prob_matrix = utils.compute_log_binomial_probability_matrix(ndocs_obs, vaux_mod, counts_obs)
    elif def_params['name'] in ('ppyy',):
        epsilon = def_params['epsilon']
        lap_mean = 2 / epsilon * (64 * np.log(2) + np.log(nkw))
        lap_scale = 2 / epsilon
        log_prob_matrix = utils.compute_log_binomial_plus_laplacian_probability_matrix(ndocs_obs, vaux, counts_obs, lap_mean, lap_scale)
    elif def_params['name'] in ('seal',):
        x = def_params['x']
        log_prob_matrix = utils.compute_log_binomial_with_power_rounding(int(ndocs_obs / x), vaux, counts_obs, x)
    else:
        raise ValueError('def name {:s} not recognized for the SAP attack'.format(def_params['name']))
    cost_vol = - log_prob_matrix
    return cost_vol


def sap_attack(obs, aux, exp_params):

    att_params = exp_params.att_params
    naive_flag = att_params['naive']
    alpha = att_params['alpha']

    token_trace, token_info = process_traces(obs, aux, exp_params.def_params)
    fobs = compute_fobs(exp_params.def_params['name'], token_trace, len(token_info))
    vobs = compute_vobs(obs['trace_type'], token_info, obs['ndocs'])
    faux = get_faux(aux)
    vaux = get_vaux(aux)

    if alpha == 1:
        c_matrix = _build_cost_freq(faux, fobs, len(obs['traces']))
    elif alpha == 0:
        c_matrix = _build_cost_vol(vaux, vobs, obs['ndocs'], exp_params.def_params, naive_flag)
    else:
        cost_freq = _build_cost_freq(faux, fobs, len(obs['traces']))
        cost_vol = _build_cost_vol(vaux, vobs, obs['ndocs'], exp_params.def_params, naive_flag)
        c_matrix = cost_freq * alpha + cost_vol * (1 - alpha)

    keyword_predictions_for_each_token = _run_algorithm(c_matrix)
    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]

    return keyword_predictions_for_each_query

