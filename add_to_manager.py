from manager import Manager
from exp_params import ExpParams
import os
import pickle
import pandas as pd
import numpy as np


def add_to_manager_wiki():
    """This was for the wiki datasets (old)"""

    manager_filename = 'manager_data1.pkl'
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    category_list = ['privacy', 'politics', 'cryptography', 'activism', 'security']

    exp_params = ExpParams()
    for def_name in ['none', 'pancake']:

        exp_params.set_defense_params(def_name)
        # niters_max = 10_000
        for niters_max in [500, 10_000]:

            for mode_fs in ['same', 'past', 'past1', 'same1']:
                for category in category_list:
                    dataset = 'wiki_{:s}'.format(category[:3])
                    for nq in [int(100e3), int(500e3)]:
                        exp_params.set_general_params(dataset=dataset, nkw=500, nqr=nq, freq='file',
                                                      mode_ds='same', mode_fs=mode_fs, mode_kw='rand', mode_query='markov')
                        for niters in [i * (niters_max // 10) for i in range(0, 11)]:
                            exp_params.set_attack_params('ihop', mode='Freq', niters=niters, pfree=0.25)
                            manager.initialize_or_add_runs(exp_params, target_runs=30)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)


def add_to_manager3():
    """This was to evaluate all the non-pancake experiments in the paper with the new code (old)"""

    manager_filename = 'manager_data3.pkl'
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    exp_params = ExpParams()

    attack_list = [
        ('sap', {'alpha': 0.5}),
        # ('umemaya', {}),
        # ('fastpfp', {}),
        ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
        ('ihop', {'mode': 'Vol', 'niters': 10000, 'pfree': 0.25}),
        # ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.1}),
        # ('ihop', {'mode': 'Vol', 'niters': 5000, 'pfree': 0.1}),
        # ('graphm', {'alpha': 0.}),
        ('graphm', {'alpha': 0.5}),
        # ('graphm', {'alpha': 1.}),
        ('ikk', {'cooling': 0.999}),
        # ('ikk', {'cooling': 0.9999}),
        ('ikk', {'cooling': 0.99995}),
        # ('ikk', {'cooling': 0.99999}),
    ]

    dataset_and_pct_list = [('enron-full', '')]
    dataset_list = [('enron-full', 'lucene', 'bow-nytimes', 'articles1', 'book-summaries', 'movie-plots')]
    exp_params.set_defense_params('none')
    for dataset_name in ['book-summaries']:
        for pct_adv in [50]:
            exp_params.set_general_params(dataset=dataset_name, nkw=500, nqr=500, freq='none', mode_ds='split{:d}'.format(pct_adv), mode_fs='past', mode_kw='rand', mode_query='each')
            for attack_name, attack_params in attack_list:
                exp_params.set_attack_params(attack_name, **attack_params)
                manager.initialize_or_add_runs(exp_params, target_runs=30)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)


def add_to_manager4():
    """Oct 2021: new datasets, new processing"""


    manager_filename = 'manager_data4.pkl'
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    exp_params = ExpParams()

    attack_list = [
        ('sap', {'alpha': 0.5}),
        # ('umemaya', {}),
        # ('fastpfp', {}),
        ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
        ('ihop', {'mode': 'Vol', 'niters': 10000, 'pfree': 0.25}),
        # ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.1}),
        # ('ihop', {'mode': 'Vol', 'niters': 5000, 'pfree': 0.1}),
        # ('graphm', {'alpha': 0.}),
        ('graphm', {'alpha': 0.5}),
        # ('graphm', {'alpha': 1.}),
        ('ikk', {'cooling': 0.999}),
        # ('ikk', {'cooling': 0.9999}),
        ('ikk', {'cooling': 0.99995}),
        # ('ikk', {'cooling': 0.99999}),
    ]

    # dataset_and_pct_list = (('articles1', (50, 25, 10)),
    #                         ('book-summaries', (50, 25, 10)),
    #                         ('movie-plots', (50, 25, 10)),
    #                         ('enron-full', (50, 25, 10)),
    #                         ('lucene', (50, 25, 10)),
    #                         ('bow-nytimes', (50, 10, 2, 1)))
    dataset_and_pct_list = (('enron-full', (10, 50)),
                            )
    exp_params.set_defense_params('none')
    # dataset_list = ('articles1', 'book-summaries', 'movie-plots', 'enron-full', 'lucene', 'bow-nytimes')
    # for dataset_name in dataset_list:
    #     for pct_adv in [50]:
    nkw_list = [500, 600, 750, 1000]
    for nkw in nkw_list:
        for dataset_name, pct_adv_list in dataset_and_pct_list:
            for pct_adv in pct_adv_list:
                exp_params.set_general_params(dataset=dataset_name, nkw=nkw, nqr=nkw, freq='none', mode_ds='split{:d}'.format(pct_adv), mode_fs='past', mode_kw='rand', mode_query='each')
                for attack_name, attack_params in attack_list:
                    exp_params.set_attack_params(attack_name, **attack_params)
                    manager.initialize_or_add_runs(exp_params, target_runs=30)
                exp_params.set_general_params(dataset=dataset_name, nkw=nkw, nqr=500, freq='none', mode_ds='split{:d}'.format(pct_adv), mode_fs='past', mode_kw='rand', mode_query='each')
                for attack_name, attack_params in attack_list:
                    exp_params.set_attack_params(attack_name, **attack_params)
                    manager.initialize_or_add_runs(exp_params, target_runs=30)


    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)


def add_to_manager_no_def(ndoc_cli_list=(20_000,), ndoc_adv_list=(500, 1000, 5000, 10000), nkw_list=(500,), nqr_equals_nkw=True,
                          nqr_val=500, dataset='all', attacks='all', manager_number=6):
    """Oct 25nd, 2021: a single script for all experiment 1 experiments"""

    manager_filename = 'manager_data{:d}.pkl'.format(manager_number)
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    exp_params = ExpParams()

    if attacks == 'all':
        attack_list = [
            ('sap', {'alpha': 0.5}),
            ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
            ('graphm', {'alpha': 0.5}),
            ('ikk', {'cooling': 0.99995}),
        ]
    elif attacks == 'fast':
        attack_list = [
            ('freq', {}),
            ('sap', {'alpha': 0.5}),
            ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
            ('ihop', {'mode': 'Vol_freq', 'niters': 1000, 'pfree': 0.25}),
        ]
    elif attacks == 'all_freq':
        attack_list = [
            ('freq', {}),
            ('sap', {'alpha': 0.5}),
            ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
            ('ihop', {'mode': 'Vol_freq', 'niters': 1000, 'pfree': 0.25}),
            ('graphm', {'alpha': 0.5}),
            ('ikk', {'cooling': 0.99995}),
        ]

    exp_params.set_defense_params('none')
    dataset_list = ('enron-full', 'lucene', 'movie-plots', 'articles1', 'bow-nytimes') if dataset == 'all' else [dataset]
    nqr_list = nkw_list if nqr_equals_nkw else [nqr_val] * len(nkw_list)
    for nkw, nqr in zip(nkw_list, nqr_list):
        for dataset_name in dataset_list:
            for ndoc_cli in ndoc_cli_list:
                for ndoc_adv in ndoc_adv_list:
                    exp_params.set_general_params(dataset=dataset_name, nkw=nkw, nqr=nqr, ndoc=ndoc_cli + ndoc_adv, freq='none', mode_ds='splitn{:d}'.format(ndoc_adv), mode_fs='past', mode_kw='rand',
                                                  mode_query='each')
                    for attack_name, attack_params in attack_list:
                        exp_params.set_attack_params(attack_name, **attack_params)
                        manager.initialize_or_add_runs(exp_params, target_runs=30)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)


def add_to_manager_iid_queries(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw_list=(500,), nqr_list=(500,), tpr_list=(0.9999,), fpr_list=(0.01, 0.02, 0.05),
                               dataset='all', attacks='all', defense_list=('none', 'clrz', 'osse'), manager_number=7):
    """Oct 25th -> Nov 3rd"""

    manager_filename = 'manager_data{:d}.pkl'.format(manager_number)
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    exp_params = ExpParams()

    if attacks == 'all':
        attack_list = [
            ('freq', {}),
            ('sap', {'alpha': 0.5}),
            # ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
            ('ihop', {'mode': 'Vol_freq', 'niters': 1000, 'pfree': 0.25}),
            ('graphm', {'alpha': 0.5}),
            ('ikk', {'cooling': 0.99995}),
        ]
    elif attacks == 'fast':
        attack_list = [
            ('freq', {}),
            ('sap', {'alpha': 0.5}),
            # ('ihop', {'mode': 'Vol', 'niters': 1000, 'pfree': 0.25}),
            ('ihop', {'mode': 'Vol_freq', 'niters': 1000, 'pfree': 0.25}),
        ]
    else:
        raise ValueError(attacks)

    dataset_list = ('enron-full', 'lucene', 'movie-plots', 'articles1', 'bow-nytimes') if dataset == 'all' else [dataset]
    for nkw in nkw_list:
        for nqr in nqr_list:
            for dataset_name in dataset_list:
                for ndoc_cli in ndoc_cli_list:
                    for ndoc_adv in ndoc_adv_list:
                        exp_params.set_general_params(dataset=dataset_name, nkw=nkw, nqr=nqr, ndoc=ndoc_cli + ndoc_adv, freq='file',
                                                      mode_ds='splitn{:d}'.format(ndoc_adv), mode_fs='past', mode_kw='rand', mode_query='iid')
                        for attack_name, attack_params in attack_list:
                            exp_params.set_attack_params(attack_name, **attack_params)
                            for defense in defense_list:
                                if defense == 'none':
                                    exp_params.set_defense_params('none')
                                    manager.initialize_or_add_runs(exp_params, target_runs=30)
                                else:
                                    for tpr in tpr_list:
                                        for fpr in fpr_list:
                                            exp_params.set_defense_params(defense, tpr=tpr, fpr=fpr)
                                            manager.initialize_or_add_runs(exp_params, target_runs=30)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)


def add_to_manager_ihop_niters(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw=500, nqr=500,
                               pfree_list=(0.1, 0.25, 0.5, 0.75),
                               dataset='all',  manager_number=7):

    manager_filename = 'manager_data{:d}.pkl'.format(manager_number)
    if not os.path.exists(manager_filename):
        manager = Manager()
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)

    exp_params = ExpParams()
    exp_params.set_defense_params('none')
    dataset_list = ('enron-full', 'lucene', 'movie-plots', 'articles1', 'bow-nytimes') if dataset == 'all' else [dataset]
    for dataset_name in dataset_list:
        for ndoc_cli in ndoc_cli_list:
            for ndoc_adv in ndoc_adv_list:
                exp_params.set_general_params(dataset=dataset_name, nkw=nkw, nqr=nqr, ndoc=ndoc_cli + ndoc_adv, freq='none', mode_ds='splitn{:d}'.format(ndoc_adv), mode_fs='past', mode_kw='rand',
                                              mode_query='each')
                for pfree in pfree_list:
                    for niters_max in [100, 1000, 10000]:
                        for niters in [i * (niters_max // 10) for i in range(0, 11)]:
                            exp_params.set_attack_params('ihop', mode='Vol', niters=niters, pfree=pfree)
                            manager.initialize_or_add_runs(exp_params, target_runs=30)

    print("Saving ResultsManager...")
    with open(manager_filename, 'wb') as f:
        pickle.dump(manager, f)


def add_to_manager_e0():
    add_to_manager_ihop_niters(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw=500, nqr=500,
                               pfree_list=(0.1, 0.25, 0.5, 0.75), dataset='all', manager_number=8)
    add_to_manager_ihop_niters(ndoc_cli_list=(20_000,), ndoc_adv_list=(5000,), nkw=500, nqr=500,
                               pfree_list=(0.1, 0.25, 0.5, 0.75), dataset='all', manager_number=8)

def add_to_manager_e10():
    add_to_manager_no_def(ndoc_cli_list=(20_000,), ndoc_adv_list=(500, 1000, 2000, 3000, 5000, 10000), nkw_list=(500,), manager_number=6)


def add_to_manager_e11():
    add_to_manager_no_def(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw_list=(500, 600, 750, 1000), manager_number=6)


def add_to_manager_e12():
    add_to_manager_no_def(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw_list=(500, 600, 750, 1000), nqr_equals_nkw=False, nqr_val=500, manager_number=6)


def add_to_manager_e13():
    add_to_manager_no_def(ndoc_cli_list=(20_000, 50_000, 100_000, 200_000), ndoc_adv_list=(10000,), nkw_list=(500,), dataset='bow-nytimes', manager_number=6)


def add_to_manager_e2():
    add_to_manager_iid_queries(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw_list=(500,), nqr_list=(500,), tpr_list=(0.9999,),
                               fpr_list=(0.01, 0.02, 0.05), dataset='all', attacks='all', manager_number=7)
    add_to_manager_iid_queries(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw_list=(3000,), nqr_list=(500, 5000, 50000, 500000), tpr_list=(),
                               fpr_list=(), dataset='all', attacks='fast', defense_list=('none',), manager_number=7)
    # add_to_manager_iid_queries(ndoc_cli_list=(20_000,), ndoc_adv_list=(10000,), nkw_list=(3000,), nqr_list=(500,), tpr_list=(0.9999,),
    #                            fpr_list=(0.01, 0.02, 0.05), dataset='all', attacks='fast', manager_number=7)




if __name__ == "__main__":
    os.system('mesg n')
    # add_to_manager_e0()
    add_to_manager_e10()
    # add_to_manager_e11()
    # add_to_manager_e12()
    # add_to_manager_e13()
    # add_to_manager_e2()
