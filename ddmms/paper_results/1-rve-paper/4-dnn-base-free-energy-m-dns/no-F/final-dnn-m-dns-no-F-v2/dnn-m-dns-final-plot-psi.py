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

args.configfile = 'dnn-free-energy-m-dns-final.config'

args.platform = 'cpu'
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

#total_model_numbers = parameter.get_model_numbers()
print("...done with parameters")
model_summary_list = []

config['RESTART']['CheckPointDir'] = './saved_weight'
config['MODEL']['ParameterID'] = ''
checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
model_path = checkpoint_dir + '/' + 'model.h5'

model = ml_models.build_model(config, train_dataset, train_labels)
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

epochs = 0
config['RESTART']['RestartWeight'] = 'Y'

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
# print("history: " , history.history['loss'], history.history['val_loss'], history.history)


label_scale = float(config['TEST']['LabelScale'])
all_data = {'label_dns': [], 'label_nn': []}


config['TEST']['DataFile'] = '../../../data/dnn-m-dns-base-free-energy/_home_xiaoxuan_comet_quench-20191012-60x60-c_large_13data.csv'
config['TEST']['DataNormalization'] = '0'
DataNormOption = int(config['TEST']['DataNormOption'])
dataset, labels, derivative, _ = ml_preprocess.load_all_data(config, args)
normed_dataset = ml_preprocess.norm(dataset, train_stats, DataNormOption)
label_nn = model.predict(normed_dataset, verbose=0, batch_size=batch_size)
all_data['label_dns'].append(labels/label_scale)
all_data['label_nn'].append(np.squeeze(label_nn)/label_scale)

config['TEST']['DataFile'] = '../../../data/dnn-m-dns-base-free-energy/_home_xiaoxuan_comet_quench-20191012-60x60-c_large_2data.csv'
config['TEST']['DataNormalization'] = '0'
DataNormOption = int(config['TEST']['DataNormOption'])
dataset, labels, derivative, _ = ml_preprocess.load_all_data(config, args)
normed_dataset = ml_preprocess.norm(dataset, train_stats, DataNormOption)
label_nn = model.predict(normed_dataset, verbose=0, batch_size=batch_size)
all_data['label_dns'].append(labels/label_scale)
all_data['label_nn'].append(np.squeeze(label_nn)/label_scale)

config['TEST']['DataFile'] = '../../../data/dnn-m-dns-base-free-energy/_home_xiaoxuan_comet_quench-20191012-60x60-c_large_32data.csv'
config['TEST']['DataNormalization'] = '0'
DataNormOption = int(config['TEST']['DataNormOption'])
dataset, labels, derivative, _ = ml_preprocess.load_all_data(config, args)
normed_dataset = ml_preprocess.norm(dataset, train_stats, DataNormOption)
label_nn = model.predict(normed_dataset, verbose=0, batch_size=batch_size)
all_data['label_dns'].append(labels/label_scale)
all_data['label_nn'].append(np.squeeze(label_nn)/label_scale)

config['TEST']['DataFile'] = '../../../data/dnn-m-dns-base-free-energy/_home_xiaoxuan_comet_quench-20191012-60x60-c_large_46data.csv'
config['TEST']['DataNormalization'] = '0'
DataNormOption = int(config['TEST']['DataNormOption'])
dataset, labels, derivative, _ = ml_preprocess.load_all_data(config, args)
normed_dataset = ml_preprocess.norm(dataset, train_stats, DataNormOption)
label_nn = model.predict(normed_dataset, verbose=0, batch_size=batch_size)
all_data['label_dns'].append(labels/label_scale)
all_data['label_nn'].append(np.squeeze(label_nn)/label_scale)

config['TEST']['DataFile'] = '../../../data/dnn-m-dns-base-free-energy/_home_xiaoxuan_comet_quench-20191012-60x60-c_large_8data.csv'
config['TEST']['DataNormalization'] = '0'
DataNormOption = int(config['TEST']['DataNormOption'])
dataset, labels, derivative, _ = ml_preprocess.load_all_data(config, args)
normed_dataset = ml_preprocess.norm(dataset, train_stats, DataNormOption)
label_nn = model.predict(normed_dataset, verbose=0, batch_size=batch_size)
all_data['label_dns'].append(labels/label_scale)
all_data['label_nn'].append(np.squeeze(label_nn)/label_scale)

for i in range(len(all_data['label_dns'])):
    plt.plot(all_data['label_dns'][i], 'k')
    plt.plot(all_data['label_nn'][i], 'r')
plt.show()

import pickle
import time
now = sys.argv[1]
pickle_out = open('all_prediction_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()
