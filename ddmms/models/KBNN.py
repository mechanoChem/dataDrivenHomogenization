import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import ddmms.misc.ml_misc as ml_misc
import ddmms.preprocess.ml_preprocess as ml_preprocess
import ddmms.models.ml_models as ml_models
import ddmms.specials.ml_specials as ml_specials
import ddmms.parameters.ml_parameters as ml_parameters
import ddmms.models.DNN_user as DNN_user
from tensorflow.keras import regularizers
import numpy as np


def shift_labels(config, dataset, dataset_index, dataset_frame, data_file):
    print("---!!!!--- reach shift_labels!!!!")
    print(
        "---!!!!--- Remember to modify old 'std', old 'mean' for DNN based KBNN"
    )
    trained_model_lists = ml_misc.getlist_str(
        config['KBNN']['LabelShiftingModels'])
    if len(trained_model_lists) > 0:
        # all_fields = ml_misc.getlist_str(config['TEST']['AllFields'])
        label_fields = ml_misc.getlist_str(config['TEST']['LabelFields'])

        if len(label_fields) > 1:
            # raise ValueError(
                # 'Shift labels is not working for two labels shifting yet!')
            print(
                'Shift labels is not working for two labels shifting yet!')

        # if label_fields[0] != all_fields[-1]:
        # raise ValueError('the single label for KBNN should put at the end of all label fields!')

        print("---!!!!---  load trained model!!!!")
        old_models = load_trained_model(trained_model_lists)
        print("---!!!!---  after load trained model!!!!")
        key0 = label_fields[0]
        old_label_scale = ml_misc.getlist_float(
            config['KBNN']['OldShiftLabelScale'])

        print('old shift features: ', config['KBNN']['OldShiftFeatures'])

        # to switch between vtk and other features
        if (config['KBNN']['OldShiftFeatures'].find('.vtk') >= 0):
            """ """
            print("--- here: vtk for label shift")
            # index should not be used anymore.
            # dataset_old = ml_preprocess.load_data_from_vtk_for_label_shift(config, dataset_index, normalization_flag = True, verbose = 0 )
            # use base frame info to do the base free energy shifting
            dataset_old = ml_preprocess.load_data_from_vtk_for_label_shift_frame(
                config, dataset_frame, normalization_flag=True, verbose=0)
        elif (config['KBNN']['OldShiftFeatures'].find('.npy') >= 0):
            """ """
            print("--- here: npy for label shift")
            # index should not be used anymore.
            # dataset_old = ml_preprocess.load_data_from_vtk_for_label_shift(config, dataset_index, normalization_flag = True, verbose = 0 )
            # use base frame info to do the base free energy shifting
            dataset_old = ml_preprocess.load_data_from_npy_for_label_shift_frame(
                config, dataset_frame, normalization_flag=True, verbose=0)
        else:
            old_feature_fields = ml_misc.getlist_str(
                config['KBNN']['OldShiftFeatures'])
            raw_dataset_old = ml_preprocess.read_csv_fields(
                data_file, old_feature_fields)
            dataset_old = raw_dataset_old.copy()

            if len(old_feature_fields) > 0:
                old_mean = ml_misc.getlist_float(config['KBNN']['OldShiftMean'])
                old_std = ml_misc.getlist_float(config['KBNN']['OldShiftStd'])
                old_data_norm = int(config['KBNN']['OldShiftDataNormOption'])
                if (old_data_norm != 2):
                    dataset_old = (dataset_old - old_mean) / old_std
                else:
                    dataset_old = (dataset_old - old_mean) / old_std + 0.5
                    raise ValueError(
                        "This part is not carefully checked. Please check it before you disable it."
                    )

        if (len(old_models) > 0):
            # convert dataset_old to numpy data array in case it is not
            try:
                dataset_old = dataset_old.to_numpy()
            except:
                try:
                    dataset_old = dataset_old.numpy()
                except:
                    pass
                pass

            for model0 in old_models:
                label_shift_amount = []
                batch_size = int(config['MODEL']['BatchSize'])
                print('run...', model0)

                # use model.predict() will run it in the eager mode and evaluate the tensor properly.
                for i0 in range(0, len(dataset_old), batch_size):
                    tmp_shift = model0.predict(
                        special_input_case(dataset_old[i0:i0 + batch_size])
                    ) / old_label_scale  # numpy type
                    label_shift_amount.extend(tmp_shift)

        for i0 in range(0, len(dataset[key0])):
            a = dataset[key0][i0] - label_shift_amount[i0]
            # tf1.13
            if(i0%200==0):
                print('--i0--',i0, 'DNS:', dataset[key0][i0],'\t', 'NN:',label_shift_amount[i0], ' key0 = ', key0, ' dataset size = ', len(dataset[key0]), ' label shift size = ',  len(label_shift_amount))
            # for tf2.0
            # print('--i0--',i0, 'DNS:', dataset[key0][i0],'\t', 'NN:',label_shift_amount[i0].numpy()[0], '\t', a.numpy()[0], '\t', abs(a.numpy()[0]/dataset[key0][i0])*100, 'new label', new_label[key0][i0])
            dataset[key0][i0] = dataset[key0][i0] - label_shift_amount[i0]


def generate_dummy_dataset(old_config):
    """ based on the label list, generate dummy dataset """
    all_fields = ml_misc.getlist_str(old_config['TEST']['AllFields'])
    label_fields = ml_misc.getlist_str(old_config['TEST']['LabelFields'])
    train_dataset = ml_specials.get_dummy_data(
        len(all_fields) - len(label_fields))
    train_label = ml_specials.get_dummy_data(len(label_fields))
    return train_dataset, train_label


def load_one_model(old_config_file):
    """ Based on the config file name to load pre-saved model info"""
    print('old_config_file:', old_config_file)
    old_config = ml_preprocess.read_config_file(old_config_file, False)
    old_data_file = old_config['TEST']['DataFile']
    if old_data_file.find('.csv') >= 0:
        dummy_train_dataset, dummy_train_labels = generate_dummy_dataset(
            old_config)
        print('dummy_train_dataset:', dummy_train_dataset, 'dummy label: ', dummy_train_labels)
    elif old_data_file.find('.vtk') >= 0:
        dummy_train_dataset, dummy_train_labels, _, _, _, _, _, _ = ml_preprocess.load_data_from_vtk_database(
            old_config, normalization_flag=True, verbose=0)
    else:
        print(
            '***WARNING***: unknown datafile in old_config file for KBNN, old_datafile = ',
            old_data_file)
        print('               could potentially lead to errors!!!!')

    old_model = ml_models.build_model(
        old_config,
        dummy_train_dataset,
        dummy_train_labels,
        set_non_trainable=True)
    parameter = ml_parameters.HyperParameters(old_config, False)
    total_model_numbers = parameter.get_model_numbers()
    if (total_model_numbers > 1):
        raise ValueError(
            'You are loading a single model, but the config file generate multiple parameter sets. It is impossible to identify which saved model to load in configfile = ',
            old_config, parameter)
    para_id, para_str, _ = parameter.next_model()
    if old_config['MODEL']['ParameterID'] != '':
        model_name_id = old_config['MODEL']['ParameterID']
    else:
        model_name_id = str(para_id) + '-' + para_str
    checkpoint_dir = old_config['RESTART']['CheckPointDir'] + model_name_id

    if (old_config['RESTART']['SavedCheckPoint'] != ''):
        saved_check_point = old_config_file[0:old_config_file.rfind(
            '/')] + '/' + old_config['RESTART']['SavedCheckPoint']
        print('saved_check_point: ', saved_check_point)
        old_model.load_weights(saved_check_point)
        print("...loading saved model ...:",
              old_config['RESTART']['SavedCheckPoint'],
              old_config_file.rfind('/'), saved_check_point)
        return old_model

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(' ... checkpoint_dir in KBNN for trained model: ', checkpoint_dir,
          ' latest model: ', latest)
    if (latest != None):
        old_model.load_weights(latest)
        print("Successfully load weight: ", latest)
    else:
        raise ValueError(
            "No saved weights, You suppose to load some thing to the old model!"
        )
    # print (checkpoint_dir)
    # print(old_config['MODEL']['NodesList'], dummy_train_dataset, dummy_train_labels)
    return old_model


def load_trained_model(trained_model_lists):
    old_models = []
    if (len(trained_model_lists) > 0):
        for m0 in trained_model_lists:
            model0 = load_one_model(m0)
            print('Pre-trained Model summary: (before) ', m0, model0)
            model0.summary()
            print('Pre-trained Model summary: (after) ', m0)
            old_models.append(model0)
    return old_models


def special_input_case(inputs, input_case=''):
    """ deal with special case for old inputs:
      for example, another DNN is for the frame 800, which has input only F11, F12, F21, F22, 
      whereas, the current model might have additional microstructure features, thus, the input
      needs to be sliced to fit for the old DNN. 
  """
    if input_case != '':

        input_ind = ml_misc.getlist_int(input_case)
        if (len(input_ind) == 1):
            s_ind = 0
            e_ind = input_ind[0]
        elif (len(input_ind) == 2):
            s_ind = input_ind[0]
            e_ind = input_ind[1]
        else:
            s_ind = 0
            e_ind = -1
        return tf.slice(inputs, [0, s_ind], [-1, e_ind])
    else:
        return inputs


class KBNN_Model(tf.keras.Model):

    def __init__(self, input_shape, num_outputs, NodesList, Activation, config):
        super(KBNN_Model, self).__init__()

        print('In KBNN')
        print('embed mdoels: ', config['KBNN']['EmbeddingModels'])
        trained_model_lists = ml_misc.getlist_str(
            config['KBNN']['EmbeddingModels'])

        self.old_models = load_trained_model(trained_model_lists)

        self.old_scale = float(config['KBNN']['OldEmbedLabelScale'])
        self.new_scale = float(config['TEST']['LabelScale'])

        self.config = config
        self.input_layer = layers.Dense(
            NodesList[0],
            input_shape=input_shape,
            activation=Activation[0],
            name='KBNN-input',
            kernel_regularizer=regularizers.l2(0.001))
        self.hidden_layer = []
        for i0 in range(1, len(NodesList)):
            # self.hidden_layer.append(layers.GaussianNoise(0.001))
            self.hidden_layer.append(
                layers.Dense(
                    NodesList[i0],
                    activation=Activation[i0],
                    name='KBNN-hd' + str(i0),
                    kernel_regularizer=regularizers.l2(0.001)))
        self.output_layer = layers.Dense(num_outputs, name='KBNN-output')

    def call(self, inputs):
        y1 = self.input_layer(inputs)
        for hd in self.hidden_layer:
            y2 = hd(y1)
            y1 = y2
        f = self.output_layer(y2)
        # print('inputs', inputs, inputs[:,0])

        if (len(self.old_models) > 0):
            for model0 in self.old_models:
                f += model0(
                    special_input_case(
                        inputs, self.config['KBNN']['OldEmbedInputSelect'])
                ) / self.old_scale * self.new_scale
                # print('--model prediction--: ', model0(special_input_case(inputs, self.config['KBNN']['OldEmbedInputSelect'])))
        return f
