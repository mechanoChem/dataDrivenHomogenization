# load modules
# for legacy python compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

# # TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# disable GPU and only CPU and Warning etc
import os, sys

import datetime
from SALib.analyze import sobol

# from tensorflow import feature_column
# from sklearn.model_selection import train_test_split
# import pathlib
# import seaborn as sns

import ddmms.help.ml_help as ml_help

import ddmms.preprocess.ml_preprocess as ml_preprocess

import ddmms.parameters.ml_parameters as ml_parameters
import ddmms.parameters.ml_parameters_dnn as ml_parameters_dnn

import ddmms.models.ml_models as ml_models
import ddmms.models.ml_optimizer as ml_optimizer
import ddmms.models.ml_loss as ml_loss

import ddmms.postprocess.ml_postprocess as ml_postprocess
import ddmms.math.ml_math as ml_math
import ddmms.specials.ml_specials as ml_specials

import ddmms.misc.ml_misc as ml_misc
import ddmms.misc.ml_callbacks as ml_callbacks

import ddmms.train.ml_kfold as ml_kfold

print("TensorFlow version: ", tf.__version__)
print(os.getcwd())
args = ml_help.sys_args()

args.configfile = 'dnn-free-energy-1dns.config'
args.platform = 'gpu'
args.inspect = 0
args.debug = False
args.verbose = 1
args.show = 0

ml_help.notebook_args(args)
config = ml_preprocess.read_config_file(args.configfile, args.debug)
# train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative, train_stats = ml_preprocess.load_and_inspect_data(
    # config, args)
dataset, labels, derivative, train_stats = ml_preprocess.load_all_data(config, args)

str_form = config['FORMAT']['PrintStringForm']
epochs = int(config['MODEL']['Epochs'])
batch_size = int(config['MODEL']['BatchSize'])
verbose = int(config['MODEL']['Verbose'])
n_splits = int(config['MODEL']['KFoldTrain'])

parameter = ml_parameters_dnn.HyperParametersDNN(
    config,
    input_shape=len(dataset.keys()),
    output_shape=len(labels.keys()),
    uniform_sample_number=25,
    neighbor_sample_number=1,
    iteration_time=3,
    sample_ratio=0.3,
    best_model_number=20,
    max_total_parameter=680,
    repeat_train=n_splits,
    debug=args.debug)

the_kfolds = ml_kfold.MLKFold(n_splits, dataset)

#total_model_numbers = parameter.get_model_numbers()
print("...done with parameters")
model_summary_list = []
while True:
    para_id, para_str, train_flag = parameter.get_next_model()
    if (train_flag and the_kfolds.any_left_fold()):
        print(para_id, para_str)

        # print("")
        # print(str_form.format("Training model: "), i0, " out of ", len(index_list))
        # print("-----------------------------------------------------")

        model_name_id = str(para_id) + '-' + para_str
        print(str_form.format('Model: '), model_name_id)

        checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
        model_path = checkpoint_dir + '/' + 'model.h5'

        train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative = the_kfolds.get_next_fold(
            dataset, labels, derivative)

        model = ml_models.build_model(config, train_dataset, train_labels)
        # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

        metrics = ml_misc.getlist_str(config['MODEL']['Metrics'])
        optimizer = ml_optimizer.build_optimizer(config)
        loss = ml_loss.build_loss(config)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        callbacks = ml_callbacks.build_callbacks(config)
        # print(type(train_dataset))
        history = model.fit(
            train_dataset.to_numpy(),
            train_labels.to_numpy(),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_dataset.to_numpy(), val_labels.to_numpy()),    # or validation_split= 0.1,
            verbose=verbose,
            callbacks=callbacks)
        # print(tf.keras.backend.eval(optimizer.lr))
        # print(model.optimizer.lr.get_value())

        model.summary()
        # print(' optimizer parameters: ', optimizer.get_config())
        # print(' optimizer updates   : ', optimizer.updates)

        # print("history: " , history.history['loss'], history.history['val_loss'], history.history)
        parameter.update_model_info(para_id, history.history)
        the_kfolds.update_kfold_status()

