#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import socket

print("TensorFlow version: ", tf.__version__)
print(os.getcwd())
args = ml_help.sys_args()

args.configfile = 'cnn-free-energy-m-dns-final.config'

if (socket.gethostname() == 'pc256g'):
    args.platform = 'cpu'
else:
    args.platform = 'gpu'

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

config['RESTART']['RestartWeight'] = 'Y'
epochs = 0

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


# In[2]:


metrics = ml_misc.getlist_str(config['MODEL']['Metrics'])
optimizer = ml_optimizer.build_optimizer(config)
loss = ml_loss.build_loss(config)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

callbacks = ml_callbacks.build_callbacks(config)
# print(type(train_dataset))
history = model.fit(
    train_dataset,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_dataset, val_labels),    # or validation_split= 0.1,
    verbose=verbose,
    callbacks=callbacks)
# print(tf.keras.backend.eval(optimizer.lr))
# print(model.optimizer.lr.get_value())

model.summary()


# In[3]:


all_weights = model.get_weights()
print(type(all_weights))
print(all_weights[0])


# In[4]:


print(type(all_weights[0]), tf.shape(all_weights[0]))


# In[5]:


for i in range(0,9):
    print(i)
    print(all_weights[0][:,:,0,i])


# In[16]:


filter4 = all_weights[0][:,:,0:1,3:4]
print(filter4)


# In[22]:


oneM = np.load('m1.npy')
print(np.shape(oneM))

output = tf.nn.conv2d(
    oneM,
    filters=filter4,
    strides=(1,1),
    padding='SAME',
    name='a',
)
output = tf.nn.relu(output)
print(tf.shape(output))
out1 = output[0,:,:,0]


# In[26]:


print(out1)
plt.imshow(out1)
plt.gray()
plt.show()

output = tf.nn.conv2d(
    oneM,
    filters=all_weights[0][:,:,:,:],
    strides=(1,1),
    padding='SAME',
    name='a',
)
output = tf.nn.relu(output)

for i in range(0,9):
    print('i=', i, tf.shape(output))
    out1 = output[0,:,:,i]
    print(out1)
    plt.clf()
    plt.gray()
    plt.imshow(out1)
    plt.show()
