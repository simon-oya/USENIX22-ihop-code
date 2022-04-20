import numpy as np
import utils
from collections import Counter
import scipy.stats
from sys import float_info
from sklearn.cluster import KMeans

epsilon = 1e-20


def get_faux(aux):
    nkw = len(aux['keywords'])
    if aux['mode_query'] == 'iid':
        faux = (aux['frequencies'] + epsilon / nkw) / (1 + epsilon * 2 / nkw)
    elif aux['mode_query'] == 'markov':
        faux = utils.get_steady_state(aux['frequencies'])
        faux[faux < 0] = 0
        faux = (faux + epsilon / nkw) / (1 + 2 * epsilon / nkw)
    elif aux['mode_query'] == 'each':
        faux = np.ones(nkw) / nkw
    else:
        raise ValueError("freq_type = {:s} nor recognized by freq".format(aux['freq_type']))
    return faux


def get_Faux(aux):
    nkw = len(aux['keywords'])
    if aux['mode_query'] == 'iid':
        Faux = np.tile(((aux['frequencies'] + epsilon / nkw) / (1 + 2 * epsilon / nkw)).reshape(nkw, 1), nkw)
    elif aux['mode_query'] == 'markov':
        Faux = (aux['frequencies'] + epsilon / nkw) / (1 + 2 * epsilon / nkw)
    elif aux['mode_query'] == 'each':
        Faux = np.ones((nkw, nkw)) / nkw
    else:
        raise ValueError("freq_type = {:s} nor recognized by freq".format(aux['freq_type']))
    return Faux


def get_Fexp_and_mapping(aux, def_params, naive_flag=False):
    if not naive_flag and def_params['name'] == 'pancake':

        nkw = len(aux['keywords'])
        nrep = 2 * nkw
        Faux = get_Faux(aux)
        faux = utils.get_steady_state(Faux)
        prob_reals, prob_dummies, replicas_per_kw = utils.compute_pancake_parameters(nkw, faux)
        aux = [0] + list(np.cumsum(replicas_per_kw, dtype=int))
        kw_id_to_replicas = [tuple(range(aux[i], aux[i + 1])) for i in range(len(aux) - 1)]
        rep_to_kw = {rep: kw for kw, replica_list in enumerate(kw_id_to_replicas) for rep in replica_list}

        c1 = 0.105
        m2 = Faux @ Faux
        m3 = m2 @ Faux

        mjoint = np.zeros((nkw + 1, nkw + 1))
        mjoint[:nkw, :nkw] = (0.81 * Faux + 0.17 * m2 + 0.02 * m3) * faux
        mjoint_keywords = c1 * mjoint + (0.25 - c1) * prob_reals * prob_reals.reshape(nkw + 1, 1) + \
                          0.25 * (prob_reals * prob_dummies.reshape(nkw + 1, 1) + prob_dummies * prob_reals.reshape(nkw + 1, 1)) + \
                          0.25 * prob_dummies * prob_dummies.reshape(nkw + 1, 1)
        mj_theo = np.zeros((nrep, nrep))
        for i in range(nrep):
            for j in range(nrep):
                mj_theo[i, j] = mjoint_keywords[rep_to_kw[i], rep_to_kw[j]] / (len(kw_id_to_replicas[rep_to_kw[i]]) * len(kw_id_to_replicas[rep_to_kw[j]]))
        mm_theo = mj_theo / np.sum(mj_theo, axis=0)

        Fexp = mm_theo
        return Fexp, rep_to_kw
    else:
        nkw = len(aux['keywords'])
        rep_to_kw = {rep: rep for rep in range(nkw)}
        return get_Faux(aux), rep_to_kw


def get_binary_matrix(dataset, keywords):
    ## This one only works with the new datasets (index-based)
    ndocs = len(dataset)
    nkw_max = 3000
    binary_database_matrix = np.zeros((ndocs, nkw_max))
    for i, doc in enumerate(dataset):
        binary_database_matrix[i, doc] = 1
    binary_database_matrix = binary_database_matrix[:, keywords]
    return binary_database_matrix  # TODO: add this to the code after the experiments?


def get_Vaux(aux):
    dataset = aux['dataset']
    ndocs = len(dataset)
    keywords = aux['keywords']
    nkw = len(keywords)
    binary_database_matrix = np.zeros((ndocs, nkw))  # TODO: do this more efficiently
    for i_doc, doc in enumerate(dataset):
        for keyword in doc:
            if keyword in keywords:
                i_kw = keywords.index(keyword)
                binary_database_matrix[i_doc, i_kw] = 1
    epsilon = 0  # Value to control that there are no zero elements
    Vaux = (np.matmul(binary_database_matrix.T, binary_database_matrix) + epsilon) / (ndocs + 2 * epsilon)
    return Vaux


def get_vaux(aux):
    dataset = aux['dataset']
    ndocs = len(dataset)
    keywords = aux['keywords']
    keyword_counter_train = Counter([kw for document in dataset for kw in document])
    vaux = [keyword_counter_train[kw] / ndocs for kw in keywords]
    return vaux


def get_Vexp(aux, def_params, naive_flag=False):
    dataset = aux['dataset']
    ndocs = len(dataset)
    keywords = aux['keywords']
    nkw = len(keywords)
    binary_database_matrix = np.zeros((ndocs, nkw))  # TODO: do this more efficiently
    for i_doc, doc in enumerate(dataset):
        for keyword in doc:
            if keyword in keywords:
                i_kw = keywords.index(keyword)
                binary_database_matrix[i_doc, i_kw] = 1

    epsilon = 0  # Value to control that there are no zero elements
    if naive_flag or def_params['name'] in ('none',):
        Vaux = (np.matmul(binary_database_matrix.T, binary_database_matrix) + epsilon) / (ndocs + 2 * epsilon)
    elif def_params['name'] in ('clrz', 'osse'):
        # TODO: Adjust this with epsilon so that there are no zero or one values
        tpr, fpr = def_params['tpr'], def_params['fpr']
        common_elements = np.matmul(binary_database_matrix.T, binary_database_matrix)
        common_not_elements = np.matmul((1 - binary_database_matrix).T, 1 - binary_database_matrix)
        Vaux = common_elements * tpr * (tpr - fpr) + common_not_elements * fpr * (fpr - tpr) + ndocs * tpr * fpr
        np.fill_diagonal(Vaux, np.diag(common_elements) * tpr + np.diag(common_not_elements) * fpr)
        Vaux = Vaux / ndocs
    elif def_params['name'] == 'pancake':
        Vaux = np.zeros((2 * nkw, 2 * nkw))
    else:
        raise ValueError("Def name '{:s}' not recognized".format(def_params['name']))
    return Vaux
