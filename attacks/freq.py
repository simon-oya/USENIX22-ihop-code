import numpy as np
from processing.process_obs import process_traces, compute_fobs
from processing.process_aux import get_faux


def freq_attack(obs, aux, exp_params):
    """Simple frequency attack by Liu et al.

    Performs query recovery by comparing auxiliary frequency information with the observed frequency information.
    The original Paper: Chang Liu, Liehuang Zhu, Mingzhong Wang, and Yu-An Tan.
    Search pattern leakage in searchable encryption: Attacks and new construction.
    Information Sciences, 2014.

    :param dict obs: Attack observations
    :param dict aux: Attack auxiliary information
    :param dict exp_params: Experiment information, such as defense parameters
    :return: Keyword predictions for each query (a list of keyword id's)
    :rtype: dict
     """

    def_params = exp_params.def_params

    token_trace, token_info = process_traces(obs, aux, def_params)
    n_tokens = len(token_info)
    fobs = compute_fobs(def_params['name'], token_trace, n_tokens)
    faux = get_faux(aux)

    keyword_predictions_for_each_token = {}
    for j in range(n_tokens):
        keyword_predictions_for_each_token[j] = np.argmin(np.abs(fobs[j] - faux))
    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]

    return keyword_predictions_for_each_query
