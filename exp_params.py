class ExpParams:

    def __init__(self, exp_params_dict=None):
        self.gen_params = {}
        self.att_params = {}
        self.def_params = {}
        if exp_params_dict is not None:
            # General parameters
            tuples = exp_params_dict['gen_p']
            gen_p_nodash = {tuples[i][1:]: tuples[i + 1] for i in range(0, len(tuples), 2)}
            self.set_general_params(dataset=exp_params_dict['dataset'], nkw=exp_params_dict['nkw'], **gen_p_nodash)
            # Defense parameters
            tuples = exp_params_dict['def_p']
            def_p_nodash = {tuples[i][1:]: tuples[i + 1] for i in range(0, len(tuples), 2)}
            if len(def_p_nodash) > 0:
                self.set_defense_params(name=exp_params_dict['def'], **def_p_nodash)
            else:
                self.set_defense_params(name=exp_params_dict['def'])
            # Attack parameters
            tuples = exp_params_dict['att_p']
            att_p_nodash = {tuples[i][1:]: tuples[i + 1] for i in range(0, len(tuples), 2)}
            self.set_attack_params(name=exp_params_dict['att'], **att_p_nodash)

    def set_general_params(self, **kwargs):
        TEMPLATE = {'dataset': 'enron',
                    'nkw': 100,
                    'nqr': 100,
                    'ndoc': 'full',
                    'freq': 'zipf',
                    'mode_ds': 'same',
                    'mode_kw': 'top',
                    'mode_fs': 'same',
                    'mode_query': 'iid',
                    'known_queries': 0}
        self.gen_params = {}
        for key, default_value in TEMPLATE.items():
            if key in kwargs:
                self.gen_params[key] = kwargs[key]
            else:
                self.gen_params[key] = default_value
        assert isinstance(self.gen_params['nkw'], int)
        assert isinstance(self.gen_params['nqr'], int)
        assert isinstance(self.gen_params['ndoc'], int) or self.gen_params['ndoc'] == 'full'
        assert self.gen_params['freq'] in ('file', 'zipf', 'none') or self.gen_params['freq'].startswith('zipfs')
        assert self.gen_params['mode_ds'].startswith(('same', 'common', 'split'))
        assert self.gen_params['mode_kw'] in ('top', 'rand')
        assert self.gen_params['mode_fs'] in ('same', 'same1', 'past', 'past1')
        assert self.gen_params['mode_query'] in ('iid', 'markov', 'each')
        return

    def set_attack_params(self, name, **kwargs):
        TEMPLATE = {'freq': {},
                    'ikk': {'naive': False, 'unique': True, 'cooling': 0.9999},
                    'graphm': {'naive': False, 'alpha': 0.5},
                    'sap': {'naive': False, 'alpha': 0.5},
                    'umemaya': {'naive': False},
                    'ihop': {'naive': False, 'mode': 'Vol', 'pfree': 0.25, 'niters': 1000},
                    'fastpfp': {'naive': False, 'alpha': 0.5},
                    }
        self.att_params = {'name': name}
        for key, default_value in TEMPLATE[name].items():
            if key in kwargs:
                self.att_params[key] = kwargs[key]
            else:
                self.att_params[key] = default_value
        # ('Vol', 'Vol_freq', 'Vol_Freq', 'Freq')
        return

    def set_defense_params(self, name, **kwargs):
        TEMPLATE = {'none': {},
                    'clrz': {'tpr': 1.0, 'fpr': 0.0},
                    'osse': {'tpr': 1.0, 'fpr': 0.0},
                    'pancake': {}
                    }
        assert name in TEMPLATE.keys()
        self.def_params = {'name': name}
        for key, default_value in TEMPLATE[name].items():
            if key in kwargs:
                self.def_params[key] = kwargs[key]
            else:
                self.def_params[key] = default_value
        return

    def return_as_dict(self):
        exp_params_dict = {}
        exp_params_dict['dataset'] = self.gen_params['dataset']
        exp_params_dict['nkw'] = self.gen_params['nkw']
        exp_params_dict['gen_p'] = tuple(x for key, val in self.gen_params.items() for x in ['-' + key, val] if key not in ('nkw', 'dataset'))
        exp_params_dict['def'] = self.def_params['name']
        exp_params_dict['def_p'] = tuple(x for key, val in self.def_params.items() for x in ['-' + key, val] if key not in ('name',))
        exp_params_dict['att'] = self.att_params['name']
        exp_params_dict['att_p'] = tuple(x for key, val in self.att_params.items() for x in ['-' + key, val] if key not in ('name',))
        return exp_params_dict

    def get_dataset_name(self):
        return self.gen_params['dataset']

    def get_defense_name(self):
        return self.def_params['name']

    def get_attack_name(self):
        return self.att_params['name']

    def __str__(self):
        output = ""
        if len(self.gen_params) > 0:
            output += "Gen params: {:s}\n".format(str(self.gen_params))
        if len(self.att_params) > 0:
            output += "Att params: {:s}\n".format(str(self.att_params))
        if len(self.def_params) > 0:
            output += "Def params: {:s}\n".format(str(self.def_params))
        return output
