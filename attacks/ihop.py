import numpy as np
from processing.process_obs import process_traces, compute_fobs, compute_Vobs, compute_Fobs
from processing.process_aux import get_faux, get_Vaux, get_Fexp_and_mapping, get_Vexp
import utils
from scipy.optimize import linear_sum_assignment as hungarian


def get_update_coefficients_functions(token_trace, token_info, aux, obs, exp_params):

    def _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
        cost_vol = -utils.compute_log_binomial_probability_matrix(ndocs, np.diagonal(Vexp)[free_keywords], np.diagonal(Vobs)[free_tags] * ndocs)
        for tag, kw in zip(fixed_tags, fixed_keywords):
            cost_vol -= utils.compute_log_binomial_probability_matrix(ndocs, Vexp[kw, free_keywords], Vobs[tag, free_tags] * ndocs)
        return cost_vol

    def _build_cost_freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):
        cost_freq = - (nqr * fobs[free_tags]) * np.log(np.array([fexp[free_keywords]]).T)
        return cost_freq

    def _build_cost_Freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags):

        cost_matrix = np.zeros((len(free_keywords), len(free_tags)))
        ss_aux = utils.get_steady_state(Fexp)

        cost_matrix -= Fobs_counts[np.ix_(free_tags, free_tags)].diagonal() * np.log(np.array([Fexp[np.ix_(free_keywords, free_keywords)].diagonal()]).T)

        ss_from_others_train = (Fexp[np.ix_(free_keywords, free_keywords)] *
                                (np.ones((len(free_keywords), len(free_keywords))) - np.eye(len(free_keywords)))) @ ss_aux[free_keywords]
        ss_from_others_train = ss_from_others_train / (np.sum(ss_aux[free_keywords]) - ss_aux[free_keywords])
        counts_from_others_test = Fobs_counts[np.ix_(free_tags, free_tags)].sum(axis=1) - Fobs_counts[np.ix_(free_tags, free_tags)].diagonal()
        cost_matrix -= counts_from_others_test * np.log(np.array([ss_from_others_train]).T)

        for tag, kw in zip(fixed_tags, fixed_keywords):
            cost_matrix -= Fobs_counts[free_tags, tag] * np.log(np.array([Fexp[free_keywords, kw]]).T)
            cost_matrix -= Fobs_counts[tag, free_tags] * np.log(np.array([Fexp[kw, free_keywords]]).T)

        return cost_matrix

    att_params = exp_params.att_params
    mode = att_params['mode']
    ndocs = obs['ndocs']
    nqr = len(token_trace)

    # Observations
    Vobs = compute_Vobs(obs['trace_type'], token_info, ndocs)
    fobs = compute_fobs(exp_params.def_params['name'], token_trace, len(token_info))
    nq_per_tok, Fobs = compute_Fobs(exp_params.def_params['name'], token_trace, len(token_info))
    Fobs_counts = Fobs * nq_per_tok

    # Auxiliary info
    fexp = get_faux(aux)
    Fexp, rep_to_kw = get_Fexp_and_mapping(aux, exp_params.def_params, att_params['naive'])
    Vexp = get_Vexp(aux, exp_params.def_params, att_params['naive'])

    if mode == 'Vol':
        return _build_cost_Vol_some_fixed, rep_to_kw
    elif mode == 'Vol_freq':
        def compute_cost(free_keywords, free_tags, fixed_keywords, fixed_tags):
            return _build_cost_Vol_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags) + \
                   _build_cost_freq_some_fixed(free_keywords, free_tags, fixed_keywords, fixed_tags)
        return compute_cost, rep_to_kw
    elif mode == 'Freq':
        return _build_cost_Freq_some_fixed, rep_to_kw
    else:
        raise ValueError("Mode '{:s}' not recognized".format(mode))


def ihop_attack(obs, aux, exp_params):

    token_trace, token_info = process_traces(obs, aux, exp_params.def_params)

    compute_coef_matrix, rep_to_kw = get_update_coefficients_functions(token_trace, token_info, aux, obs, exp_params)
    att_params = exp_params.att_params

    pct_free = att_params['pfree']
    n_iters = att_params['niters']
    nrep = len(rep_to_kw)
    ntok = len(token_info)

    # 1) PROCESS GROUND-TRUTH INFORMATION
    if 'ground_truth_queries' in aux and len(aux['ground_truth_queries']) > 0:
        raise NotImplementedError
        # TODO: this is not part of the IHOP paper, but a future extension.
        # TODO: map the ground truth info to token-replica format
        # ground_truth_tokens, ground_truth_reps = [list(val) for val in zip(*obs['known_queries'].items())]
    else:
        ground_truth_tokens, ground_truth_reps = [], []
    unknown_toks = [i for i in range(ntok) if i not in ground_truth_tokens]
    unknown_reps = [i for i in range(nrep) if i not in ground_truth_reps]

    # First matching:
    c_matrix_original = compute_coef_matrix(unknown_reps, unknown_toks, ground_truth_reps, ground_truth_tokens)
    row_ind, col_ind = hungarian(c_matrix_original)
    replica_predictions_for_each_token = {token: rep for token, rep in zip(ground_truth_tokens, ground_truth_reps)}
    for j, i in zip(col_ind, row_ind):
        replica_predictions_for_each_token[unknown_toks[j]] = unknown_reps[i]

    if 'niter_list' in att_params:
        run_multiple_niters, niter_list, rep_pred_tok_list = True, att_params['niter_list'], []
        if 0 in niter_list:
            rep_pred_tok_list.append(replica_predictions_for_each_token.copy())
    else:
        run_multiple_niters, niter_list, rep_pred_tok_list = False, [], []

    # Iterate using co-occurrence:
    n_free = int(pct_free * len(unknown_toks))
    assert n_free > 1
    for k in range(n_iters):
        random_unknown_tokens = list(np.random.permutation(unknown_toks))
        free_tokens = random_unknown_tokens[:n_free]
        fixed_tokens = random_unknown_tokens[n_free:] + ground_truth_tokens
        fixed_reps = [replica_predictions_for_each_token[token] for token in fixed_tokens]
        free_replicas = [rep for rep in unknown_reps if rep not in fixed_reps]

        c_matrix = compute_coef_matrix(free_replicas, free_tokens, fixed_reps, fixed_tokens)

        row_ind, col_ind = hungarian(c_matrix)
        for j, i in zip(col_ind, row_ind):
            replica_predictions_for_each_token[free_tokens[j]] = free_replicas[i]

        if run_multiple_niters and k + 1 in niter_list:
            rep_pred_tok_list.append(replica_predictions_for_each_token.copy())

        if (k + 1) % (n_iters // 10) == 0:
            print("{:d}".format(((k + 1) // (n_iters // 10)) - 1), end='', flush=True)

    if not run_multiple_niters:
        keyword_predictions_for_each_query = [rep_to_kw[replica_predictions_for_each_token[token]] for token in token_trace]
        return keyword_predictions_for_each_query
    else:
        kw_pred_for_each_query_list = []
        for replica_predictions_for_each_token in rep_pred_tok_list:
            kw_pred_for_each_query_list.append([rep_to_kw[replica_predictions_for_each_token[token]] for token in token_trace])
        return kw_pred_for_each_query_list
