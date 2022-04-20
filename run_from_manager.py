import numpy as np
import os
import time
import multiprocessing
from experiment import run_experiment
import pickle
import pytz
from exp_params import ExpParams
import argparse


def print_exp_to_run(d, n_runs=-1, exp_id=-1):
    if exp_id > 0:
        print('***********************[ {:d} ]***********************'.format(exp_id))
    else:
        print('********************************************************')
    print('Setting: {:s} {:s} nkw={:d}'.format(d['dataset'], str(d['gen_p']), d['nkw']))
    print('Def: {:s} {:s}'.format(d['def'], str(d['def_p'])))
    print('Att: {:s} {:s}'.format(d['att'], str(d['att_p'])))
    if n_runs > 0:
        print("* Number of runs: {:d}".format(n_runs))
    print('********************************************************', flush=True)


def run_and_save_experiment_all_together(exp_params, exp_id, seed, results_path, multi=False):
    # We only print the full experiments when we launch the first experiment of this series:
    if seed == 0:
        # time.sleep(np.random.randint(0, 3))  # Some delay to avoid printing overlaps
        print('')
        print_exp_to_run(exp_params, exp_id=exp_id)
    else:
        time.sleep(0.1)
    print('[{:d}-{:d}]'.format(exp_id, seed), end='', flush=True)

    dir_to_save = os.path.join(results_path, 'done_runs_{:04}'.format(exp_id))
    while not os.path.exists(dir_to_save):
        try:
            os.makedirs(dir_to_save)
        except FileExistsError:
            print("Directory {:s} already existed, gonna sleep a bit".format(dir_to_save), flush=True)
            time.sleep(np.random.randint(0, 3))
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print("Could not create directory {:s}, gonna sleep a bit".format(dir_to_save), flush=True)
            time.sleep(np.random.randint(0, 3))

    temp_filename = 'results_{:03d}.temp'.format(seed)

    if os.path.exists(os.path.join(dir_to_save, temp_filename)) or os.path.exists(os.path.join(dir_to_save, 'results_{:03d}.pkl'.format(seed))) \
            or os.path.exists(os.path.join(dir_to_save, 'results_{:03d}_0.pkl'.format(seed))):
        print("[{:d}-{:d} already existed!]".format(exp_id, seed), end='', flush=True)
        return -1
    else:
        with open(os.path.join(dir_to_save, temp_filename), 'w') as f:
            pass

        exp_params_object = ExpParams(exp_params)
        if exp_params_object.att_params['name'] == 'ihop' and multi:
            niters_max = exp_params_object.att_params['niters']
            niters_list = [i * (niters_max // 10) for i in range(0, 11)]
            exp_params_object.att_params['niter_list'] = niters_list
            acc_list, accu_list, time_exp = run_experiment(exp_params_object, seed, debug_mode=False)
            del exp_params_object.att_params['niter_list']
            for acc, accu, niters in zip(acc_list, accu_list, niters_list):
                results_filename = 'results_{:06d}_{:d}.pkl'.format(seed, niters)
                exp_params_object.att_params['niters'] = niters
                with open(os.path.join(dir_to_save, results_filename), 'wb') as f:
                    res_dict = {'seed': seed, 'accuracy': acc, 'accuracy_un': accu, 'time': time_exp}
                    pickle.dump((exp_params_object, res_dict), f)
        else:
            acc, accu, time_exp = run_experiment(exp_params_object, seed, debug_mode=False)
            results_filename = 'results_{:03d}.pkl'.format(seed)
            with open(os.path.join(dir_to_save, results_filename), 'wb') as f:
                res_dict = {'seed': seed, 'accuracy': acc, 'accuracy_un': accu, 'time': time_exp}
                pickle.dump((exp_params_object, res_dict), f)
        os.remove(os.path.join(dir_to_save, temp_filename))
        print("[!{:d}-{:d}]".format(exp_id, seed), end='', flush=True)
        return 1


PRO_DATASETS_PATH = 'datasets_pro'
EXPERIMENTS_PATH = 'results'

if __name__ == "__main__":
    """If ran normally, it queues all the experiments that have not been ran from the manager.
    By default, the manager number is 1, but it can be changed with the first argument.
    The flag --multi, if not -1, loads experiments with args.multi number of iteratiosn only"""

    parser = argparse.ArgumentParser()
    parser.add_argument('manager_number', type=int, nargs='?', default=1)
    parser.add_argument('--multi', help='store results for ihop during multiple runs', type=int, nargs='?', default=-1)
    args = parser.parse_args()
    print(args)
    multi_bool = False if args.multi == -1 else True

    # quit()

    manager_filename = 'manager_data{:d}.pkl'.format(args.manager_number)
    if not os.path.exists(manager_filename):
        raise ValueError("File {:s} does not exist!".format(manager_filename))
    else:
        with open(manager_filename, 'rb') as f:
            manager = pickle.load(f)
        print("Loaded {:s}".format(manager_filename))

    time_init = time.time()
    results_path = 'results'
    tz_ON = pytz.timezone('Canada/Eastern')

    params_list = []
    for index, row in manager.experiments.iterrows():
        # if not args.multiold or args.multiold and index % 11 == 10:
        exp_dict = row.to_dict()
        del exp_dict['target_runs']
        del exp_dict['res_pointer']
        exp_id = row['res_pointer']
        # print(exp_dict)
        if args.multi == -1 or ExpParams(exp_dict).att_params['niters'] == args.multi:
            for seed in range(row['target_runs']):
                if seed not in manager.results[exp_id]['seed'].values:
                    params_list.append((exp_dict.copy(), exp_id, seed, results_path, multi_bool))

    print("Done gathering experiments, we have a total of {:d} seeds:".format(len(params_list)), flush=True)
    for _, exp_id, seed, _, _ in params_list:
        print("({:d}-{:d})".format(exp_id, seed), end='', flush=True)
    print("")

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
    # pool = multiprocessing.Pool(60)
    pool.starmap(run_and_save_experiment_all_together, params_list, chunksize=1)
    print("DONE!!!")
