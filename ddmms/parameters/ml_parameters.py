import sys
import ddmms.misc.ml_misc as ml_misc
import math

import itertools
from SALib.sample import saltelli


class HyperParameters:
    """
  this is a class to perform hyper-parameter search for ML
  """

    def __init__(self, config, debug):
        self.config = config
        self.debug = debug
        self.total_model_numbers = 1
        self.implemented_keys = ['hiddenlayernumber', 'nodeslist', 'learningrate', 'activation']
        self.studying_parameters = []
        self.all_parameters = [[], []]
        self.para_id = -1
        self.para_str = 'model-' + 'H' + str(len(ml_misc.getlist_int(self.config['MODEL']['NodesList']))) \
                                 + 'N' + str(ml_misc.getlist_int(self.config['MODEL']['NodesList'])[0]) \
                                 + 'A' + ml_misc.getlist_str(self.config['MODEL']['Activation'])[0] \
                                 + 'L' + self.config['MODEL']['LearningRate'] \
                                 + 'O' + self.config['MODEL']['Optimizer']
        self.sobol_parameters = ''
        if (self.config['MODEL']['HyperParameterSearch'].lower() == 'y'):
            self.form_parameter_list()

    def form_parameter_list(self):
        # print(hl_num, nd_num, lr_num, act_fcn)

        lists = []
        for k1 in self.config['HYPERPARAMETERS'].keys():
            if k1 in self.implemented_keys:
                if (k1 == 'hiddenlayernumber'):
                    self.update_hidden_layer_number()
                    self.mixing_parameters(self.hl_num_list)
                    self.studying_parameters.append(k1)
                    lists.append(self.hl_num_list)
                if (k1 == 'nodeslist'):
                    self.update_node_list_number()
                    self.mixing_parameters(self.nd_num_list)
                    self.studying_parameters.append(k1)
                    lists.append(self.nd_num_list)
                if (k1 == 'learningrate'):
                    self.update_learning_rate_number()
                    self.mixing_parameters(self.lr_num_list)
                    self.studying_parameters.append(k1)
                    lists.append(self.lr_num_list)
                if (k1 == 'activation'):
                    self.update_activation_number()
                    self.mixing_parameters(self.act_fcn)
                    self.studying_parameters.append(k1)
                    lists.append(self.act_fcn)
            else:
                raise ValueError('The hyper-parameter search key: ', k1,
                                 ' is not implemented in HyperParameters class: ', self.implemented_keys)

        # prepare for sobol list study
        self.sobol_parameters = {
            'num_vars': len(self.studying_parameters),
            'names': self.studying_parameters,
            'bounds': [],
            'lists': lists
        }
        if (int(self.config['MODEL']['SensitivityAnalysisNumber']) > 0):
            self.form_sobol_parameter_list()

    def update_model(self):
        one_para = self.parameter_studies[0]
        if (len(self.studying_parameters) != len(one_para)):
            raise ValueError(
                'something is wrong in hyperparameter next_model()!!! Parameter names != parameter values: ',
                self.studying_parameters, one_para)

        del self.parameter_studies[0]
        # print("update config to get new model info", one_para, self.studying_parameters)

        hl_num = 0
        if 'hiddenlayernumber' in self.studying_parameters:
            hl_num = one_para[self.studying_parameters.index('hiddenlayernumber')]
        else:
            hl_num = len(ml_misc.getlist_int(self.config['MODEL']['NodesList']))

        para_str = ''
        for i0 in range(0, len(self.studying_parameters)):
            para_name = self.studying_parameters[i0]
            para_str += para_name[0].upper()
            para_val = str(one_para[i0])
            # print('para_val', para_val)
            para_str += para_val + '-'
            if (para_name == 'nodeslist' or para_name == 'activation'):
                para_val = ','.join([para_val] * hl_num)
            self.config['MODEL'][para_name] = para_val
            if (self.debug):
                print('para_name: ', para_name, para_val)
        self.para_str = para_str[0:-1] + 'O' + self.config['MODEL']['Optimizer']
        if (self.debug):
            print('para_str: ', self.para_str)

    def form_sobol_parameter_list(self):
        """ call sobol  """
        NUM_OF_SAMPLE = int(self.config['MODEL']['SensitivityAnalysisNumber'])
        print('parameters: ', self.sobol_parameters, NUM_OF_SAMPLE)

        new_bounds = []
        new_names = []
        other_names = []
        other_lists = []
        for i in range(0, self.sobol_parameters['num_vars']):
            if (self.sobol_parameters['names'][i] != 'activation'):
                if (len(self.sobol_parameters['lists'][i]) >= 2):
                    if (self.sobol_parameters['names'][i] != 'learningrate'):
                        new_bounds.append(
                            [min(self.sobol_parameters['lists'][i]),
                             max(self.sobol_parameters['lists'][i])])
                        new_names.append(self.sobol_parameters['names'][i])
                    else:
                        new_bounds.append([
                            math.log10(min(self.sobol_parameters['lists'][i])),
                            math.log10(max(self.sobol_parameters['lists'][i]))
                        ])
                        new_names.append(self.sobol_parameters['names'][i])
                else:    # only one parameters
                    other_names.append(self.sobol_parameters['names'][i])
                    other_lists.append(self.sobol_parameters['lists'][i])
            else:    # activation will be deal differently
                other_names.append(self.sobol_parameters['names'][i])
                other_lists.append(self.sobol_parameters['lists'][i])

        print(new_bounds, new_names, other_names, other_lists)

        self.sobol_parameters = {
            'num_vars': len(new_bounds),
            'names': new_names,
            'bounds': new_bounds,
        }

        para_list = saltelli.sample(self.sobol_parameters, NUM_OF_SAMPLE)
        new_para_list = []

        # make others to be int
        for i in range(0, len(para_list)):
            l0 = para_list[i]
            l1 = []
            for j in range(0, len(l0)):
                if (new_names[j] != 'learningrate'):
                    l1.append(int(round(l0[j])))
                else:
                    l1.append(math.pow(10, l0[j]))

            # print('l0: ', l0)
            new_para_list.append(list(l1))

        # print ('new_para_list: ', new_para_list)
        self.studying_parameters = new_names
        self.all_parameters[0] = new_para_list
        for i in range(0, len(other_names)):
            self.studying_parameters.append(other_names[i])
            self.mixing_parameters(other_lists[i])

        # print ('para_list: ', para_list)
        # print(self.studying_parameters)
        # print(self.parameter_studies)
        # bounds.append()
        # bounds.append([min(self.nd_num_list), max(self.nd_num_list)])
        # bounds.append([min(self.lr_num_list), max(self.lr_num_list)])
        # bounds.append([0, 0])

    def mixing_parameters(self, list0):
        if (len(self.all_parameters[0]) == 0):
            self.all_parameters[0] = list0
            self.parameter_studies = self.all_parameters[0]
        else:
            if (len(self.all_parameters[1]) == 0):
                self.all_parameters[1] = list0
                tmp_studies = list(itertools.product(*self.all_parameters))

                # flatten the irregular nested tuples
                for i0 in range(0, len(tmp_studies)):
                    t1 = tmp_studies[i0]
                    t1 = list(t1)
                    t2 = []
                    if type(t1[0]) == type([0, 1]):
                        t2.extend(list(t1[0]))
                        t2.append(t1[1])
                    else:
                        t2 = t1
                    tmp_studies[i0] = t2

                self.parameter_studies = tmp_studies
                self.all_parameters[0] = self.parameter_studies
                self.all_parameters[1] = []
                # print ("..1..", self.all_parameters)
        # print ("..studies..", self.parameter_studies, len(self.parameter_studies))
        self.total_model_numbers = len(self.parameter_studies)

    def update_hidden_layer_number(self):
        hl_num = ml_misc.getlist_int(self.config['HYPERPARAMETERS']['HiddenLayerNumber'])
        if (len(hl_num) == 3):
            if ((hl_num[2] < hl_num[1]) and (hl_num[1] > hl_num[2] + hl_num[0])):
                self.hl_num_list = [i for i in range(hl_num[0], hl_num[1] + 1, hl_num[2])]
            else:
                self.hl_num_list = hl_num
        else:
            self.hl_num_list = hl_num
        if (self.debug):
            print('hl_num_list: ', self.hl_num_list)

    def update_node_list_number(self):
        nd_num = ml_misc.getlist_int(self.config['HYPERPARAMETERS']['NodesList'])
        if (len(nd_num) == 3):
            if ((nd_num[2] < nd_num[1]) and (nd_num[1] > nd_num[2] + nd_num[0])):
                nd_num_list = [i for i in range(nd_num[0], nd_num[1] + 1, nd_num[2])]
                # at least have a ratio of 1.25
                self.nd_num_list = [nd_num_list[0]]
                for i in range(1, len(nd_num_list)):
                    if (float(nd_num_list[i]) / self.nd_num_list[-1] >= 1.2):
                        self.nd_num_list.append(nd_num_list[i])
            else:
                self.nd_num_list = nd_num
        else:
            self.nd_num_list = nd_num

        if (self.debug):
            print('nd_num_list: ', self.nd_num_list)

    def update_learning_rate_number(self):
        lr_num = ml_misc.getlist_float(self.config['HYPERPARAMETERS']['LearningRate'])
        # if(len(lr_num) == 3):
        # # print(lr_num) # [0.001, 0.01, 0.1] Bugs are here. Avoid auto generation. Manually provide the list
        # if(  (lr_num[2] * lr_num[0] < lr_num[1])):
        # self.lr_num_list = [lr_num[0]]
        # while (self.lr_num_list[-1] * lr_num[2] <= lr_num[1]):
        # self.lr_num_list.append(self.lr_num_list[-1] * lr_num[2])
        # else:
        # self.lr_num_list = lr_num
        # else:
        # self.lr_num_list = lr_num
        self.lr_num_list = lr_num
        if (self.debug):
            print('lr_num_list: ', self.lr_num_list)

    def update_activation_number(self):
        act_fcn = ml_misc.getlist_str(self.config['HYPERPARAMETERS']['Activation'])
        self.act_fcn = act_fcn
        if (self.debug):
            print('act_fcn: ', self.act_fcn)

    def get_model_numbers(self):
        return self.total_model_numbers

    def next_model(self):
        if (self.config['MODEL']['HyperParameterSearch'].lower() == 'y'):
            self.update_model()
        self.para_id += 1
        return self.para_id, self.para_str, self.sobol_parameters


if __name__ == '__main__':
    import sys
    sys.path.append('../preprocess')
    import ml_preprocess

    sys.path.append('../help')
    import ml_help

    print("...testing....")

    args = ml_help.parse_sys_args()
    config = ml_preprocess.read_config_file(args.configfile, args.debug)
    parameter = HyperParameters(config, args.debug)
    print('total model numbers: ', parameter.get_model_numbers())
    for i0 in range(0, parameter.get_model_numbers()):
        para_id, para_str, sobol_parameters = parameter.next_model()
        print('model: ', i0, para_id)
