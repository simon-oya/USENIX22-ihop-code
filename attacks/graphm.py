import numpy as np
from processing.process_obs import process_traces, compute_Vobs
from processing.process_aux import get_Vexp
import utils
import os
import stat
import subprocess
import tempfile

GRAPHM_PATH = './graphm-0.52/bin'
if not os.path.exists('tmp/'):
    os.makedirs('tmp/')


def _write_matrix_to_file_ascii(file, matrix):
    for row in matrix:
        row_str = ' '.join("{:.6f}".format(val) for val in row) + '\n'
        file.write(row_str.encode('ascii'))


def _return_config_text(algorithms_list, alpha, relpath_experiments, out_filename):
    """relpath_experiments: path from where we run graphm to where the graph files are"""

    config_text = """//*********************GRAPHS**********************************
//graph_1,graph_2 are graph adjacency matrices,
//C_matrix is the matrix of local similarities  between vertices of graph_1 and graph_2.
//If graph_1 is NxN and graph_2 is MxM then C_matrix should be NxM
graph_1={relpath:s}/graph_1 s
graph_2={relpath:s}/graph_2 s
C_matrix={relpath:s}/c_matrix s
//*******************ALGORITHMS********************************
//used algorithms and what should be used as initial solution in corresponding algorithms
algo={alg:s} s
algo_init_sol={init:s} s
solution_file=solution_im.txt s
//coeficient of linear combination between (1-alpha_ldh)*||graph_1-P*graph_2*P^T||^2_F +alpha_ldh*C_matrix
alpha_ldh={alpha:.6f} d
cdesc_matrix=A c
cscore_matrix=A c
//**************PARAMETERS SECTION*****************************
hungarian_max=10000 d
algo_fw_xeps=0.01 d
algo_fw_feps=0.01 d
//0 - just add a set of isolated nodes to the smallest graph, 1 - double size
dummy_nodes=0 i
// fill for dummy nodes (0.5 - these nodes will be connected with all other by edges of weight 0.5(min_weight+max_weight))
dummy_nodes_fill=0 d
// fill for linear matrix C, usually that's the minimum (dummy_nodes_c_coef=0),
// but may be the maximum (dummy_nodes_c_coef=1)
dummy_nodes_c_coef=0.01 d

qcvqcc_lambda_M=10 d
qcvqcc_lambda_min=1e-5 d


//0 - all matching are possible, 1-only matching with positive local similarity are possible
blast_match=0 i
blast_match_proj=0 i


//****************OUTPUT***************************************
//output file and its format
exp_out_file={relpath:s}/{out:s} s
exp_out_format=Parameters Compact Permutation s
//other
debugprint=0 i
debugprint_file=debug.txt s
verbose_mode=1 i
//verbose file may be a file or just a screen:cout
verbose_file=cout s
""".format(alg=" ".join(alg for alg in algorithms_list), init=" ".join("unif" for _ in algorithms_list),
           out=out_filename, alpha=alpha, relpath=relpath_experiments)
    return config_text


def _run_path_algorithm(m_matrix, m_prime_matrix, c_matrix, alpha):

    temp_dir = tempfile.TemporaryDirectory(suffix=None, prefix='graphm_', dir='tmp/')
    # print(temp_dir.name)

    with open(os.path.join(temp_dir.name, 'graph_1'), 'wb') as f:
        _write_matrix_to_file_ascii(f, m_matrix)

    with open(os.path.join(temp_dir.name, 'graph_2'), 'wb') as f:
        _write_matrix_to_file_ascii(f, m_prime_matrix)

    if alpha > 0:
        with open(os.path.join(temp_dir.name, 'c_matrix'), 'wb') as f:
            _write_matrix_to_file_ascii(f, c_matrix)

    with open(os.path.join(temp_dir.name, 'config.txt'), 'w') as f:
        f.write(_return_config_text(['PATH'], alpha, os.path.relpath(temp_dir.name, '.'), 'graphm_output'))

    test_script_path = os.path.join(temp_dir.name, 'run_script')
    with open(test_script_path, 'w') as f:
        f.write("#!/bin/sh\n")
        f.write("{:s}/graphm {:s}/config.txt\n".format(os.path.relpath(GRAPHM_PATH, ''), os.path.relpath(temp_dir.name, '.')))
    st = os.stat(test_script_path)
    os.chmod(test_script_path, st.st_mode | stat.S_IEXEC)

    # RUN THE ATTACK
    subprocess.run([os.path.join(temp_dir.name, "run_script")], capture_output=True)

    results = []
    with open(os.path.relpath(temp_dir.name, '.') + '/graphm_output', 'r') as f:
        while f.readline() != "Permutations:\n":
            pass
        f.readline()  # This is the line with the attack names (only PATH, in theory)
        for line in f:
            results.append(int(line) - 1)  # Line should be a single integer now

    # COMPUTE PREDICTIONS
    # A result = is a list, where the i-th value (j) means that the i-th training keyword is the j-th testing keyword.
    # This following code reverts that, so that query_predictions_for_each_obs[attack] is a vector that contains the indices of the training
    # keyword for each testing keyword.
    keyword_predictions_for_each_token = {}
    for tag in range(m_prime_matrix.shape[0]):
        keyword_predictions_for_each_token[tag] = results.index(tag)

    temp_dir.cleanup()

    return keyword_predictions_for_each_token


def graphm_attack(obs, aux, exp_params):

    def_params = exp_params.def_params
    att_params = exp_params.att_params
    naive_flag = att_params['naive']
    alpha = att_params['alpha']
    ndocs = obs['ndocs']
    token_trace, token_info = process_traces(obs, aux, def_params)
    Vobs = compute_Vobs(obs['trace_type'], token_info, ndocs)
    Vexp = get_Vexp(aux, exp_params.def_params, naive_flag)
    vobs = np.diagonal(Vobs).copy()
    vexp = np.diagonal(Vexp).copy()
    score_matrix = np.exp(utils.compute_log_binomial_probability_matrix(ndocs, vexp, vobs * ndocs))
    np.fill_diagonal(Vobs, 0)
    np.fill_diagonal(Vexp, 0)

    keyword_predictions_for_each_token = _run_path_algorithm(Vexp, Vobs, score_matrix, alpha)

    keyword_predictions_for_each_query = [keyword_predictions_for_each_token[token] for token in token_trace]
    return keyword_predictions_for_each_query