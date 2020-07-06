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
import ddmms.parameters.ml_parameters_cnn as ml_parameters_cnn

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

args.configfile = 'kbnn-load-dnn-1-frame.config'

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

the_kfolds = ml_kfold.MLKFold(n_splits, dataset)
train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative = the_kfolds.get_next_fold(
    dataset, labels, derivative, final_data=True)
# train_derivative, val_derivative, test_derivative = the_kfolds.get_current_fold_derivative_data()

#total_model_numbers = parameter.get_model_numbers()
print("...done with parameters")
model_summary_list = []

config['RESTART']['CheckPointDir'] = './saved_weight'
config['MODEL']['ParameterID'] = ''
checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
model_path = checkpoint_dir + '/' + 'model.h5'

model = ml_models.build_model(config, train_dataset, train_labels, train_stats=train_stats)
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

if (config['RESTART']['RestartWeight'].lower() == 'y'):
    print('checkpoint_dir for restart: ', checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print("latest checkpoint: ", latest)
    # latest="/opt/scratch/ml/cnn-hyperelasticity-bvp-2d-conv/restart/cp-2000.ckpt"
    if (latest != None):
        model.load_weights(latest)
        print("Successfully load weight: ", latest)
    else:
        print("No saved weights, start to train the model from the beginning!")
        pass

metrics = ml_misc.getlist_str(config['MODEL']['Metrics'])
optimizer = ml_optimizer.build_optimizer(config)
# loss = ml_loss.build_loss(config)
loss = ml_loss.my_mse_loss_with_grad(BetaP=0.0)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
label_scale = float(config['TEST']['LabelScale'])

callbacks = ml_callbacks.build_callbacks(config)
train_dataset = train_dataset.to_numpy()
train_labels = train_labels.to_numpy()
# print('before: ', train_labels, train_stats)
val_dataset = val_dataset.to_numpy()
val_labels = val_labels.to_numpy()
test_dataset = test_dataset.to_numpy()
test_labels = test_labels.to_numpy()

# make sure that the derivative data is scaled correctly

# The NN/DNS scaled derivative data should be: * label_scale * train_stats['std'] (has already multiplied by label_scale )

# Since the feature is scaled, and label psi is scaled, the S_NN will be scaled to: label_scale * train_stats['std']
# the model will scale S_NN back to no-scaled status.
# here we scale F, and P to no-scaled status
modified_label_scale = np.array([1.0, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale])
train_labels = train_labels*modified_label_scale
val_labels = val_labels*modified_label_scale
test_labels = test_labels*modified_label_scale
print('after: ', train_labels)
# print(type(train_dataset))
history = model.fit(
    train_dataset,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_dataset, val_labels),    # or validation_split= 0.1,
    verbose=verbose,
    callbacks=callbacks)

model.summary()
# print("history: " , history.history['loss'], history.history['val_loss'], history.history)

all_data = {'test_label': [], 'test_nn': [], 'val_label': [], 'val_nn': [], 'train_label': [], 'train_nn': []}

test_nn = model.predict(test_dataset, verbose=0, batch_size=batch_size)
val_nn = model.predict(val_dataset, verbose=0, batch_size=batch_size)
train_nn = model.predict(train_dataset, verbose=0, batch_size=batch_size)

for i in np.squeeze(test_nn):
    # print('test_nn:', i)
    all_data['test_nn'].append(i[0] / label_scale)
for i in np.squeeze(val_nn):
    all_data['val_nn'].append(i[0] / label_scale)
for i in np.squeeze(train_nn):
    all_data['train_nn'].append(i[0] / label_scale)

for i in test_labels:
    all_data['test_label'].append(i[0] / label_scale)
    # print('test_label: ', i)
for i in val_labels:
    all_data['val_label'].append(i[0] / label_scale)
for i in train_labels:
    all_data['train_label'].append(i[0] / label_scale)
# print('all_data: ', all_data)
print('test_nn shape: ', np.shape(np.squeeze(test_nn)))
print('test_labels shape: ', np.shape(test_labels))

import pickle
import time
now = time.strftime("%Y%m%d%H%M%S")
pickle_out = open('all_data_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()

pickle_out = open('history_' + now + '.pickle', "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

# all_data['P_DNS']= test_labels[:,1:4]/label_scale/train_stats['std'].to_numpy()[0:3]
# all_data['P_NN'] = test_nn[:,1:4]/label_scale/train_stats['std'].to_numpy()[0:3]
all_data['P_DNS']= test_labels[:,1:5]
all_data['P_NN'] = test_nn[:,1:5]

pickle_out = open('all_P_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()

print('save to: ', 'all_data_' + now + '.pickle', 'history_' + now + '.pickle', 'all_P_' + now + '.pickle')
print('the prediction of P and delta Psi_me is not the best model fit with lowest loss!')

