#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
# print("history: " , history.history['loss'], history.history['val_loss'], history.history)

# label_scale = float(config['TEST']['LabelScale'])
# all_data = {'test_label': [], 'test_nn': [], 'val_label': [], 'val_nn': [], 'train_label': [], 'train_nn': []}

# test_nn = model.predict(test_dataset, verbose=0, batch_size=batch_size)
# val_nn = model.predict(val_dataset, verbose=0, batch_size=batch_size)
# train_nn = model.predict(train_dataset, verbose=0, batch_size=batch_size)

# if epochs > 0 :
    # for i in np.squeeze(test_nn):
        # all_data['test_nn'].append(i / label_scale)
    # for i in np.squeeze(val_nn):
        # all_data['val_nn'].append(i / label_scale)
    # for i in np.squeeze(train_nn):
        # all_data['train_nn'].append(i / label_scale)
    
    # for i in test_labels:
        # all_data['test_label'].append(i / label_scale)
    # for i in val_labels:
        # all_data['val_label'].append(i / label_scale)
    # for i in train_labels:
        # all_data['train_label'].append(i / label_scale)
    # # print('all_data: ', all_data)
    
    # import pickle
    # import time
    # now = time.strftime("%Y%m%d%H%M%S")
    # pickle_out = open('all_data_' + now + '.pickle', "wb")
    # pickle.dump(all_data, pickle_out)
    # pickle_out.close()
    
    # pickle_out = open('history_' + now + '.pickle', "wb")
    # pickle.dump(history.history, pickle_out)
    # pickle_out.close()
    # print('save to: ', 'all_data_' + now + '.pickle', 'history_' + now + '.pickle')


# In[ ]:


#ml_postprocess.inspect_cnn_features(model, config, test_dataset, savefig=True)


# In[ ]:


#ml_postprocess.inspect_cnn_features(model, config, test_dataset[20:50], savefig=True)


# In[2]:


#ml_postprocess.inspect_cnn_features(model, config, test_dataset[300:400], savefig=True)


# In[3]:


def inspect_cnn_features_output_each_filter(model, config, test_dataset, savefig=False, name=''):
    num_images = int(config['OUTPUT']['NumImages'])
    inspect_layers = ml_misc.getlist_int(config['OUTPUT']['InspectLayers'])

    total_images = 0
    for l0 in inspect_layers:
        out1 = model.check_layer(test_dataset[0:1], l0)
        total_images += tf.shape(out1[0]).numpy()[2]
        print('total_images:', total_images)

    if (int(np.sqrt(total_images)) * int(np.sqrt(total_images)) >=
            total_images):
        num_col = int(np.sqrt(total_images))
    else:
        num_col = int(np.sqrt(total_images)) + 1
    num_row = num_col

    for i0 in range(0, num_images):
        plt.figure()
        count = 0
        for l0 in inspect_layers:
            out1 = model.check_layer(test_dataset[i0:i0 + 1], l0)
            img0 = out1[0]
            shape0 = tf.shape(img0).numpy()
            print('shape0: ', shape0)
            for i in range(1,shape0[2] + 1):  # 2nd index is the feature numbers
                count += 1
                plt.clf()
                ax = plt.gca()
                plt.imshow(out1[0, :, :, i - 1])  # tensor
                print(i, out1[0, :, :, i - 1])
                
                import scipy.io as sio
                a = {}
                m = np.copy(out1[0, :, :, i - 1])
                a['m'] = m
                sio.savemat(str(i)+'.mat', a)
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                if savefig:
                    plt.savefig(name+str(l0)+'-'+str(i)+'.pdf', bbox_inches='tight', format='pdf')
                plt.show()


# In[4]:


def inspect_cnn_features(model, config, test_dataset, savefig=False, name=''):
    num_images = int(config['OUTPUT']['NumImages'])
    inspect_layers = ml_misc.getlist_int(config['OUTPUT']['InspectLayers'])

    total_images = 0
    for l0 in inspect_layers:
        out1 = model.check_layer(test_dataset[0:1], l0)
        total_images += tf.shape(out1[0]).numpy()[2]
        print('total_images:', total_images)

    if (int(np.sqrt(total_images)) * int(np.sqrt(total_images)) >=
            total_images):
        num_col = int(np.sqrt(total_images))
    else:
        num_col = int(np.sqrt(total_images)) + 1
    num_row = num_col

    for i0 in range(0, num_images):
        plt.figure()
        count = 0
        for l0 in inspect_layers:
            out1 = model.check_layer(test_dataset[i0:i0 + 1], l0)
            #print(out1)
            img0 = out1[0]
            shape0 = tf.shape(img0).numpy()
            print('shape0: ', shape0)
            for i in range(1,shape0[2] + 1):  # 2nd index is the feature numbers
                count += 1
                ax = plt.subplot(num_col, num_row, count)
                plt.imshow(out1[0, :, :, i - 1])  # tensor
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if savefig:
            plt.savefig(name+str(l0)+'.png', bbox_inches='tight', format='png')
        plt.show()


# In[5]:


config['OUTPUT']['NumImages']='1'
config['OUTPUT']['InspectLayers'] = '0'
inspect_cnn_features_output_each_filter(model, config, test_dataset[300:400], savefig=True, name='m-dns-image-l')
config['OUTPUT']['InspectLayers'] = '1'
inspect_cnn_features_output_each_filter(model, config, test_dataset[300:400], savefig=True, name='m-dns-image-l')
#config['OUTPUT']['InspectLayers'] = '3'
#inspect_cnn_features(model, config, test_dataset[300:400], savefig=True, name='m-dns-image-l')


# # In[ ]:



# config['OUTPUT']['NumImages']='1'
# config['OUTPUT']['InspectLayers'] = '0'
# inspect_cnn_features(model, config, test_dataset[350:400], savefig=True, name='m-dns-image-2-l')
# config['OUTPUT']['InspectLayers'] = '1'
# inspect_cnn_features(model, config, test_dataset[350:400], savefig=True, name='m-dns-image-2-l')
# config['OUTPUT']['InspectLayers'] = '3'
# inspect_cnn_features(model, config, test_dataset[350:400], savefig=True, name='m-dns-image-2-l')


# # In[ ]:





# # In[ ]:


# config['OUTPUT']['NumImages']='1'
# config['OUTPUT']['InspectLayers'] = '0'
# inspect_cnn_features_output_each_filter(model, config, test_dataset[1000:1100], savefig=True, name='m-dns-image-3-l')
# config['OUTPUT']['InspectLayers'] = '1'
# inspect_cnn_features_output_each_filter(model, config, test_dataset[1000:1100], savefig=True, name='m-dns-image-3-l')
# #config['OUTPUT']['InspectLayers'] = '3'
# #inspect_cnn_features(model, config, test_dataset[1000:1100], savefig=True, name='m-dns-image-3-l')


# # In[ ]:





# # In[ ]:


# m1 = np.load('m1.npy')
# config['OUTPUT']['NumImages']='1'
# config['OUTPUT']['InspectLayers'] = '0'
# inspect_cnn_features(model, config, m1, savefig=True, name='m-dns-image-4-l')
# config['OUTPUT']['InspectLayers'] = '1'
# inspect_cnn_features(model, config, m1, savefig=True, name='m-dns-image-4-l')
# config['OUTPUT']['InspectLayers'] = '3'
# inspect_cnn_features(model, config, m1, savefig=True, name='m-dns-image-4-l')


# # In[ ]:


# config['OUTPUT']['NumImages']='1'
# config['OUTPUT']['InspectLayers'] = '0'
# inspect_cnn_features(model, config, test_dataset[0:1100], savefig=True, name='m-dns-image-5-l')
# config['OUTPUT']['InspectLayers'] = '1'
# inspect_cnn_features(model, config, test_dataset[0:1100], savefig=True, name='m-dns-image-5-l')
# config['OUTPUT']['InspectLayers'] = '3'
# inspect_cnn_features(model, config, test_dataset[0:1100], savefig=True, name='m-dns-image-5-l')


# # In[ ]:


# config['OUTPUT']['NumImages']='1'
# config['OUTPUT']['InspectLayers'] = '0'
# inspect_cnn_features(model, config, test_dataset[20:1100], savefig=True, name='m-dns-image-6-l')
# config['OUTPUT']['InspectLayers'] = '1'
# inspect_cnn_features(model, config, test_dataset[20:1100], savefig=True, name='m-dns-image-6-l')
# config['OUTPUT']['InspectLayers'] = '3'
# inspect_cnn_features(model, config, test_dataset[20:1100], savefig=True, name='m-dns-image-6-l')


# # In[ ]:


# #config['OUTPUT']['NumImages']='1'
# #config['OUTPUT']['InspectLayers'] = '1'
# #for i in range(0, 500):
# #    inspect_cnn_features(model, config, test_dataset[i:i+2], savefig=True, name='m-dns-image-a'+str(i)+'-l')


# # In[ ]:




