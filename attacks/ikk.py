import numpy as np
from processing.process_obs import process_traces, compute_Vobs
from processing.process_aux import get_Vexp


def _run_simmulated_annealing(remaining_tags, remaining_keywords, initial_state, m_matrix, m_prime_matrix, unique_flag,
                              initial_temp=200, cooling_rate=0.999, reject_threshold=1500):

    def compute_cost(state, m_matrix, m_prime_matrix):
        total_cost = np.sum((m_matrix[np.ix_(state, state)] - m_prime_matrix) ** 2)
        return total_cost

    current_state = initial_state[:]  # copy

    current_cost = compute_cost(current_state, m_matrix, m_prime_matrix)
    succ_reject = 0
    current_temp = initial_temp

    n_iters = 0
    n_max_iters = int((np.log(initial_temp) - np.log(1e-10)) / -np.log(cooling_rate))
    # print("  Starting annealing (init={:.2f}, cool={:e}, rej_th={:d} -- nmax_iters = {:d}... ".format(initial_temp, cooling_rate, reject_threshold, n_max_iters), flush=True)
    while current_temp > 1e-10 and succ_reject < reject_threshold:

        next_state = current_state[:]  # copy
        tag_to_replace = np.random.choice(remaining_tags)
        old_kw = next_state[tag_to_replace]
        new_kw = np.random.choice(remaining_keywords)
        if unique_flag and new_kw in next_state:
            next_state[next_state.index(new_kw)] = old_kw
        next_state[tag_to_replace] = new_kw

        next_cost = compute_cost(next_state, m_matrix, m_prime_matrix)
        if next_cost < current_cost or np.random.rand() < np.exp(-(next_cost - current_cost) / current_temp):
            current_state = next_state
            current_cost = next_cost
            succ_reject = 0
        else:
            succ_reject += 1
        current_temp *= cooling_rate
        n_iters += 1
        if n_iters % int(n_max_iters / 10) == 0:
            print(int(n_iters // (n_max_iters / 10)), end='', flush=True)
        #     print("({:d}%)  n_iters={:d}/{:d}...".format(int(n_iters * 10 // (n_max_iters / 10)), n_iters, n_max_iters), flush=True)

    # print("  Done!", flush=True)
    return current_state


def ikk_attack(obs, aux, exp_params):
    def_params = exp_params.def_params
    att_params = exp_params.att_params
    naive_flag = att_params['naive']
    unique_flag = att_params['unique']
    cooling_rate = att_params['cooling']

    ndocs = obs['ndocs']
    token_trace, token_info = process_traces(obs, aux, def_params)
    ntok = len(token_info)
    nkw = len(aux['keywords'])
    Vobs = compute_Vobs(obs['trace_type'], token_info, ndocs)
    Vexp = get_Vexp(aux, exp_params.def_params, naive_flag)

    known_tokens, known_keywords = [], []  # In the IHOP paper there are no ground-truth queries
    remaining_tokens = [j for j in range(ntok) if j not in known_tokens]
    remaining_keywords = [i for i in range(nkw) if i not in known_keywords]

    token_list = list(known_tokens) + remaining_tokens
    if unique_flag:
        keyword_list = list(known_keywords) + list(np.random.choice(remaining_keywords, size=len(remaining_tokens), replace=False))
    else:
        keyword_list = list(known_keywords) + list(np.random.choice(range(nkw), size=len(remaining_tokens), replace=True))
    initial_state = [kw_id for _, kw_id in sorted(zip(token_list, keyword_list))]

    if len(remaining_tokens) == 0:
        final_state = initial_state
    elif unique_flag:
        final_state = _run_simmulated_annealing(remaining_tokens, remaining_keywords, initial_state, Vexp, Vobs, unique_flag, cooling_rate=cooling_rate)
    else:
        final_state = _run_simmulated_annealing(remaining_tokens, range(nkw), initial_state, Vexp, Vobs, unique_flag, cooling_rate=cooling_rate)

    keyword_predictions_for_each_token = {}
    for tag_id in range(ntok):
        keyword_predictions_for_each_token[tag_id] = final_state[tag_id]
    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]
    return keyword_predictions_for_each_query