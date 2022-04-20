import numpy as np
import os
import pickle
import pandas as pd
import argparse


def get_odd(x):
    return [x[i] for i in range(1, len(x), 2)]


def print_console_options():
    print("""
--------------------------------------
[pa] Prints ALL experiments
[pr] Prints REMAINING experiments
[pp <int>] Print a PARTICULAR experiment given an index
[p <col=value> ...] Print experiments that match the column values
[w] Write all pending experiments to run to a file
[w <int> <int>] Write experiments between two indices (included) to run
[eat] Eat pickles        
[reset <int> <int>] Reset target runs and results of experiments between two indices (included)
[cancel <int> <int>] Sets target runs to current number of done runs between two indices (included)
[remove <int> <int>] Removes experiments and results between two indices (included)
[e] Exit""")


exp_param_dict_column = ['dataset', 'nkw', 'gen_p', 'def', 'def_p', 'att', 'att_p', 'target_runs', 'res_pointer']


class Manager:

    def __init__(self):
        """
        self.experiments is dataframe with the experiment description
        self.results is a dictionary with the results stored in a dataframe format (seed, accuracy, time....)
        """
        self.experiments = pd.DataFrame(columns=exp_param_dict_column)
        self.results = {}

    def _get_new_pointer(self):
        new_pointer = len(self.results)
        if new_pointer in self.results:
            new_pointer = 0
            while new_pointer in self.results:
                new_pointer += 1
        return new_pointer

    def _find_indices(self, exp_param_dict):
        if len(self.experiments.index) == 0:
            return []
        mask = pd.Series([True] * len(self.experiments.index))
        for key, value in exp_param_dict.items():
            mask &= self.experiments[key] == value
        if any(mask):
            return self.experiments[mask].index
        else:
            return []

    def _find_pointer(self, exp_param_dict):
        indices = self._find_indices(exp_param_dict)
        if len(indices) == 1:  # Probably error because this is a list?
            return self.experiments.at[indices[0], 'res_pointer']
        else:
            return -1

    def _create_new_experiment(self, exp_param_dict, target_runs):
        dataframe_row = exp_param_dict.copy()
        indices = self._find_indices(dataframe_row)
        if len(indices) == 1:
            print("WARNING! Experiment already existed, cannot create")
            return -1
        elif len(indices) > 1:
            print("WARNING! Experiment was duplicated, there is something wrong here")
            return -1
        else:
            dataframe_row['target_runs'] = target_runs
            new_pointer = self._get_new_pointer()
            dataframe_row['res_pointer'] = new_pointer
            self.experiments = self.experiments.append(dataframe_row, ignore_index=True)
            self.results[new_pointer] = pd.DataFrame(columns=['seed', 'accuracy', 'accuracy_un', 'time'])
            return new_pointer

    def _add_results(self, exp_param_dict, res_dict):
        indices = self._find_indices(exp_param_dict)
        if len(indices) > 1:
            print("We found multiple experiments that match the query, cannot add results")
            return -1
        elif len(indices) == 0:
            print("Experiment did not exist, creating it")
            pointer = self._create_new_experiment(exp_param_dict, 0)
        else:  # len(index) = 1
            index = indices[0]
            pointer = self.experiments.loc[index]['res_pointer']

        if res_dict['seed'] not in self.results[pointer]['seed'].values:
            self.results[pointer] = self.results[pointer].append(res_dict, ignore_index=True)
        else:
            print("Cannot add experiment because this seed is already registered")

    # Called from plot
    def get_results(self, exp_params):
        pointer = self._find_pointer(exp_params.return_as_dict())
        if pointer >= 0:
            return list(self.results[pointer]['accuracy'].values), \
                   list(self.results[pointer]['accuracy_un'].values), \
                   list(self.results[pointer]['time'].values)
        else:
            return [], [], []

    def initialize_or_add_runs(self, exp_params, target_runs):
        exp_param_dict = exp_params.return_as_dict()
        indices = self._find_indices(exp_param_dict)
        if len(indices) == 1:
            index = indices[0]
            if self.experiments.loc[index]['target_runs'] >= target_runs:
                print("Experiment already existed and had more target runs, ignoring")
            else:
                print("Experiment already existed, increasing target runs")
                self.experiments.loc[index]['target_runs'] = target_runs
        elif len(indices) == 0:
            self._create_new_experiment(exp_param_dict, target_runs)
            print("Created experiment")
        pass

    def write_pending_experiments_request(self, experiments_path, minidx=0, maxidx=-1):
        if maxidx == -1:
            maxidx = len(self.experiments.index)
        for index, row in self.experiments.iterrows():
            if minidx <= index <= maxidx:
                seeds_to_run = []
                pointer = row['res_pointer']
                for seed in range(row['target_runs']):
                    if seed not in self.results[pointer]['seed'].values:
                        seeds_to_run.append(seed)
                if len(seeds_to_run) > 0:
                    with open(os.path.join(experiments_path, 'todo_{:04d}.pkl'.format(index)), 'wb') as f:
                        exp_dict = row.to_dict()
                        del exp_dict['target_runs']
                        del exp_dict['res_pointer']
                        pickle.dump((exp_dict, seeds_to_run), f)
                        print("Created todo_{:04d}.pkl".format(index))

    def eat_pickles(self, experiments_path):
        """eats pickles in given directory and all subdirectories"""
        count = 0
        subdirectories = os.scandir(experiments_path)
        for subdir in subdirectories:
            if subdir.is_dir():
                for file in os.scandir(subdir):
                    if file.name.startswith('results') and file.name.endswith('.pkl'):
                        with open(file, 'rb') as f:
                            exp_params, res_dict = pickle.load(f)
                            self._add_results(exp_params.return_as_dict(), res_dict)
                        os.remove(file)
                        count += 1
                try:
                    os.rmdir(subdir)  # Can only delete it if it's empty
                except OSError:
                    print("Dir not empty, not removing")
        if count > 0:
            print("Yum x{:d}".format(count))
        else:
            print("Nothing to eat")
        return count

    def reset_results(self, exp_params):
        print(exp_params)
        indices = self._find_indices(exp_params.return_as_dict())
        response = input("We found {:d} indices, type reset to continue!".format(len(indices)))
        if response == 'reset':
            self.reset_results_given_indices(indices)
        else:
            print("Aborting")

    def reset_results_given_indices(self, indices):
        for index in indices:
            pointer = self.experiments.loc[index]['res_pointer']
            self.results[pointer] = pd.DataFrame(columns=['seed', 'accuracy', 'accuracy_un', 'time'])
            print('reseted {:d}'.format(index))

    def reset_experiments_between_indices(self, start, end, target_runs=0):
        for index in range(start, end + 1):
            pointer = self.experiments.loc[index]['res_pointer']
            self.results[pointer] = pd.DataFrame(columns=['seed', 'accuracy', 'accuracy_un', 'time'])
            self.experiments.loc[index]['target_runs'] = target_runs
            print('reseted {:d}'.format(index))
        print("Done!")

    def cancel_experiments_between_indices(self, start, end):
        for index in range(start, end + 1):
            pointer = self.experiments.loc[index]['res_pointer']
            if len(self.results[pointer]['seed']) == 0:
                current_max_runs = 0
            else:
                current_max_runs = int(max(self.results[pointer]['seed'])) + 1
            print("current max runs = {:d}".format(current_max_runs))
            self.experiments.loc[index]['target_runs'] = current_max_runs
            print('cancelled {:d}'.format(index))
        print("Done!")

    def remove_experiments_between_indices(self, start, end):
        # self.print_pending_experiments()
        for index in range(start, end + 1):
            pointer = self.experiments.loc[index]['res_pointer']
            del self.results[pointer]
            self.experiments.drop(index=index, inplace=True)
        print("Deleted!")

    ########## PRINT FUNCTIONS ##########
    def print_all(self):

        results_table_rows = [[np.round(self.results[pointer]["accuracy"].mean(), 3),
                               np.round(self.results[pointer]["accuracy_un"].mean(), 3),
                               np.round(self.results[pointer]["time"].mean(), 2),
                               len(self.results[pointer].index)] for pointer in self.experiments['res_pointer']]
        # print(self.experiments.columns)
        results_table = pd.DataFrame(np.array(results_table_rows), columns=['acc', 'acc_un', 'time', 'runs'])
        print(pd.concat([self.experiments['dataset'], self.experiments[['nkw']], self.experiments['gen_p'].apply(get_odd),
                         self.experiments[['def']], self.experiments['def_p'].apply(get_odd),
                         self.experiments[['att']], self.experiments['att_p'].apply(get_odd),
                         self.experiments['target_runs'], results_table], axis=1).to_string())

    def print_results_given_indices(self, indices):
        smaller_df = self.experiments.loc[indices].copy()
        results_table_rows = [[np.round(self.results[pointer]["accuracy"].mean(), 2),
                               np.round(self.results[pointer]["accuracy_un"].mean(), 2),
                               np.round(self.results[pointer]["time"].mean(), 2),
                               len(self.results[pointer].index)] for pointer in smaller_df['res_pointer']]
        results_df = pd.DataFrame(np.array(results_table_rows), index=indices, columns=['acc', 'acc_un', 'time', 'runs'])
        # concat_df = pd.concat([smaller_df, results_df], axis=1)
        print(pd.concat([smaller_df[['nkw']], smaller_df['gen_p'].apply(get_odd),
                         smaller_df[['def']], smaller_df['def_p'].apply(get_odd),
                         smaller_df[['att']], smaller_df['att_p'].apply(get_odd),
                         smaller_df['target_runs'], results_df], axis=1).to_string())
        # print(concat_df.to_string())

    def print_given_dict(self, exp_dict):
        print(exp_dict)
        indices = self._find_indices(exp_dict)
        self.print_results_given_indices(indices)

    def print_pending_experiments(self):
        unfinished_indices = []
        for index, row in self.experiments.iterrows():
            pointer = row['res_pointer']
            if len(self.results[pointer]["accuracy"]) < row['target_runs']:
                unfinished_indices.append(index)
        if len(unfinished_indices) > 1:
            self.print_results_given_indices(unfinished_indices)
        else:
            print("Nothing unfinished!")

    def print_results_table_given_index(self, index):
        pointer = self.experiments.loc[index]['res_pointer']
        df = self.results[pointer].sort_values(by=['seed'])
        print(df)


def add_ndoc_to_old_manager(manager):
    print("hi")
    tuple_example = manager.experiments['gen_p'][0]
    if '-ndoc' in tuple_example:
        print("-ndoc was already in the manager, skipping this...")
    else:
        print("Adding (-ndoc, 'full') to the manager")
        manager.experiments['gen_p'] = manager.experiments['gen_p'].apply(lambda x: x[:2] + ('-ndoc', 'full') + x[2:])
    return


if __name__ == "__main__":

    EXPERIMENTS_PATH = 'results'

    parser = argparse.ArgumentParser()
    parser.add_argument('manager_number', type=int, nargs='?', default=7)
    args = parser.parse_args()

    manager_data_filename = 'manager_data{:d}.pkl'.format(args.manager_number)
    if not os.path.exists(manager_data_filename):
        raise ValueError("File {:s} does not exist!".format(manager_data_filename))
    else:
        with open(manager_data_filename, 'rb') as f:
            manager = pickle.load(f)
        print("Loaded {:s}".format(manager_data_filename))

    if not os.path.exists(EXPERIMENTS_PATH):
        os.makedirs(EXPERIMENTS_PATH)

    while True:
        print_console_options()
        print(exp_param_dict_column)
        choice = input("Enter your option: ").lower().split(' ')
        if choice[0] in ('e',):
            print("Saving ResultsManager...")
            with open(manager_data_filename, 'wb') as f:
                pickle.dump(manager, f)
            break
        elif choice[0] == 'p':
            exp_dict = {}
            for vals in choice[1:]:
                if len(vals.split('=')) == 2:
                    key, val = vals.split('=')
                    if key in exp_param_dict_column:
                        exp_dict[key] = eval(val)

            manager.print_given_dict(exp_dict)
        elif choice[0] in ('pa',):
            manager.print_all()
        elif choice[0] in ('w',):
            if len(choice) > 1:
                manager.write_pending_experiments_request(EXPERIMENTS_PATH, int(choice[1]), int(choice[2]))
            else:
                manager.write_pending_experiments_request(EXPERIMENTS_PATH)
        elif choice[0] in ('eat',):
            count = manager.eat_pickles(EXPERIMENTS_PATH)
            if count > 0:
                with open(manager_data_filename, 'wb') as f:
                    pickle.dump(manager, f)
                print('Saved manager')
        elif choice[0] in ('pr',):
            manager.print_pending_experiments()
        elif choice[0] in ('reset',):
            if len(choice) == 2:
                start = end = int(choice[1])
            else:
                start = int(choice[1])
                end = int(choice[2])
            response = input("Going to RESET experiments from index {:d} to {:d}, type 'reset' to proceed: ".format(start, end))
            if response == 'reset':
                target_runs = int(input("Enter target runs: "))
                manager.reset_experiments_between_indices(start, end, target_runs)
        elif choice[0] in ('cancel',):
            if len(choice) == 2:
                start = end = int(choice[1])
            else:
                start = int(choice[1])
                end = int(choice[2])
            manager.cancel_experiments_between_indices(start, end)
        elif choice[0] in ('pp',):
            if len(choice) == 2:
                index = int(choice[1])
                manager.print_results_table_given_index(index)
        elif choice[0] in ('remove',):
            if len(choice) == 2:
                start = end = int(choice[1])
            else:
                start = int(choice[1])
                end = int(choice[2])
            response = input("Going to REMOVE experiments from index {:d} to {:d}, type 'remove' to proceed: ".format(start, end))
            if response == 'remove':
                manager.remove_experiments_between_indices(start, end)
        else:
            print("Unrecognized command")

    print("Bye")
