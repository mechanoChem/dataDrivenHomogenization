# load modules
# for legacy python compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

# # TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

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

import tensorflow.keras.backend as K

def merge_two_tensor(a):
    return K.concatenate([a[0], a[1]], axis=1)

class user_KBNN_with_grad(tf.keras.Model):

    def __init__(self,label_scale, train_stats):
        super(user_KBNN_with_grad, self).__init__()

        self.label_scale = label_scale
        self.train_stats_std = train_stats['std'].to_numpy()[0:3] # E11, E12, E22
        print('--------------:', 1/label_scale/self.train_stats_std)
        # exit(0)

        self.cnn_layers = []
        self.cnn_layers.append(layers.Conv2D(8, (3,3), input_shape=[61,61,1], strides=(2,2), activation='relu',padding='same'))
        self.cnn_layers.append(layers.MaxPooling2D((2,2), padding='same'))
        self.cnn_layers.append(layers.Conv2D(16, (3,3), strides=(2,2), activation='relu',padding='same'))
        self.cnn_layers.append(layers.MaxPooling2D((2,2), padding='same'))
        self.cnn_layers.append(layers.Conv2D(24, (3,3), strides=(2,2), activation='relu',padding='same'))
        self.cnn_layers.append(layers.MaxPooling2D((2,2), padding='same'))
        self.cnn_layers.append(layers.Flatten())
        self.cnn_layers.append(layers.Dense(8, activation="relu"))


        self.dnn_layers = []
        self.dnn_layers.append(layers.Lambda(merge_two_tensor))
        self.dnn_layers.append(layers.Dense(48, activation="softplus"))
        self.dnn_layers.append(layers.Dense(48, activation="softplus"))
        self.dnn_layers.append(layers.Dense(48, activation="softplus"))
        self.dnn_layers.append(layers.Dense(1, activation="linear"))

    @tf.function(autograph=False)
    def call(self, inputs):
        inp0 = inputs[0]
        inp1 = inputs[1]
        y1 = self.cnn_layers[0](inp0)  #,
        for hd in self.cnn_layers[1:]:
            y2 = hd(y1)
            y1 = y2

        with tf.GradientTape() as g:
            g.watch(inp1)
            y1 = self.dnn_layers[0]([inp1, y2])  #,
            for hd in self.dnn_layers[1:]:
                y2 = hd(y1)
                y1 = y2
        dy_dx = g.gradient(y2, inp1)/self.label_scale/self.train_stats_std
        # dy_dx = g.gradient(y2, inp1)

        # for penalize P
        return tf.concat([y2, dy_dx[:, 0:1], dy_dx[:, 1:2], dy_dx[:, 1:2], dy_dx[:, 2:3]], 1)

        # for penalize S
        # return tf.concat([y2, dy_dx[:, 0:3]], 1)



print("TensorFlow version: ", tf.__version__)
print(os.getcwd())
args = ml_help.sys_args()

args.configfile = 'kbnn-load-dnn-many-frames-penalize-P.config'

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

vtk_dataset = np.load('../../../data/kbnn-m-dns-perturbed-free-energy-E/kbnn_all_vtk_data.npy')
#vtk_dataset = np.load('../../../data/kbnn-m-dns-perturbed-free-energy-E/kbnn_all_perturb_vtk_data.npy')


vtk_dataset = vtk_dataset.astype(np.float32)

dummy_labels = labels.to_numpy()

# if (tf.__version__[0:1] == '2'):
    # vtk_dataset = tf.convert_to_tensor(vtk_dataset, dtype=tf.float32)
    # dummy_labels = tf.convert_to_tensor(dummy_labels, dtype=tf.float32)
    # epochs = 0
    # config['RESTART']['RestartWeight'] = 'Y'

print('all data : ', tf.shape(vtk_dataset), type(vtk_dataset))
vtk_train_dataset, vtk_train_labels, vtk_val_dataset, vtk_val_labels, vtk_test_dataset, vtk_test_labels, vtk_test_derivative = the_kfolds.get_next_fold(
    vtk_dataset, dummy_labels, derivative, final_data=True)

print('all train data : ', tf.shape(vtk_train_dataset))

# print(type(vtk_train_dataset))
# print(np.shape(vtk_train_dataset))

#total_model_numbers = parameter.get_model_numbers()
print("...done with parameters")
model_summary_list = []

config['RESTART']['CheckPointDir'] = './saved_weight'
config['MODEL']['ParameterID'] = ''
checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
model_path = checkpoint_dir + '/' + 'model.h5'

# # define two sets of inputs
# inp0 = Input(shape=(61,61,1))
# x = layers.Conv2D(8, (3,3), strides=(2,2), activation='relu',padding='same')(inp0)
# x = layers.MaxPooling2D((2,2), padding='same')(x)
# x = layers.Conv2D(16, (3,3), strides=(2,2), activation='relu',padding='same')(x)
# x = layers.MaxPooling2D((2,2), padding='same')(x)
# x = layers.Conv2D(24, (3,3), strides=(2,2), activation='relu',padding='same')(x)
# x = layers.MaxPooling2D((2,2), padding='same')(x)
# x = layers.Flatten()(x)
# x = layers.Dense(8, activation="relu")(x)
# x = Model(inputs=inp0, outputs=x)

# inp1 = Input(shape=(4,))

# combined = layers.concatenate([x.output, inp1])

# # apply a FC layer and then a regression prediction on the
# # combined outputs
# z = layers.Dense(48, activation="softplus")(combined)
# z = layers.Dense(48, activation="softplus")(z)
# z = layers.Dense(48, activation="softplus")(z)
# z = layers.Dense(1, activation="linear")(z)
# model = Model(inputs=[x.input, inp1], outputs=z)

train_dataset = train_dataset.to_numpy()
val_dataset = val_dataset.to_numpy()
test_dataset = test_dataset.to_numpy()
train_dataset = tf.convert_to_tensor(train_dataset, dtype=tf.float32)
val_dataset = tf.convert_to_tensor(val_dataset, dtype=tf.float32)
test_dataset = tf.convert_to_tensor(test_dataset, dtype=tf.float32)

train_labels = train_labels.to_numpy()
val_labels = val_labels.to_numpy()
test_labels = test_labels.to_numpy()
print('before train label: ', train_labels[0:5,:])

label_scale = float(config['TEST']['LabelScale'])

# make sure that the derivative data is scaled correctly

# The NN/DNS scaled derivative data should be: * label_scale * train_stats['std'] (has already multiplied by label_scale )

# Since the feature is scaled, and label psi is scaled, the S_NN will be scaled to: label_scale * train_stats['std']
# the model will scale S_NN back to no-scaled status.
# here we scale F, and P to no-scaled status
# modified_label_scale = np.array([1.0, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale])
modified_label_scale = np.array([1.0, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale])
# modified_label_scale = np.array([1, train_stats['std'].to_numpy()[0], train_stats['std'].to_numpy()[1], train_stats['std'].to_numpy()[2], 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale, 1.0/label_scale])

train_labels = train_labels*modified_label_scale
val_labels = val_labels*modified_label_scale
test_labels = test_labels*modified_label_scale
print('after train label: ', train_labels[0:5,:])

# P_DNS = train_labels[:,1:5]
# P_DNS = tf.reshape(P_DNS, [-1,2,2])

# F_DNS = train_labels[:,5:9]
# F_DNS = tf.reshape(F_DNS, [-1,2,2])

# A_NN = tf.linalg.matmul(F_DNS, P_DNS)
# print(P_DNS)
# print(F_DNS)
# print(A_NN)

# exit(0)

train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)


model = user_KBNN_with_grad(label_scale, train_stats)



if (config['RESTART']['RestartWeight'].lower() == 'y'):
    print('checkpoint_dir for restart: ', checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    timemark = sys.argv[1]
    history_file = 'history_' + timemark + '.pickle'
    import pickle
    history = pickle.load(open(history_file, "rb"))
    # print(history['val_loss'])
    val_loss = []
    for i in range(0, len(history['val_loss'])):
        if (i+1) % 100 == 0:
            val_loss.append(history['val_loss'][i])
    val_loss = np.array(val_loss)
    # print(val_loss)
    # print(len(history['val_loss']), min(val_loss))
    best_ind0 = str(np.argmin(val_loss) * 100)
    latest = latest.replace('10000', best_ind0)
    print("latest checkpoint: ", latest)

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
#loss = ml_loss.build_loss(config)
loss = ml_loss.my_mse_loss_with_grad(BetaP=1000.0)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

callbacks = ml_callbacks.build_callbacks(config)
print('after call_back: ', tf.shape(vtk_train_dataset), tf.shape(train_dataset))
# print(type(train_dataset))
history = model.fit(
    [vtk_train_dataset, train_dataset],
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([vtk_val_dataset, val_dataset], val_labels),    # or validation_split= 0.1,
    verbose=verbose,
    callbacks=callbacks)

model.summary()
# print("history: " , history.history['loss'], history.history['val_loss'], history.history)

label_scale = float(config['TEST']['LabelScale'])
all_data = {'test_label': [], 'test_nn': [], 'val_label': [], 'val_nn': [], 'train_label': [], 'train_nn': []}

test_nn = model.predict([vtk_test_dataset,test_dataset], verbose=0, batch_size=batch_size)
val_nn = model.predict([vtk_val_dataset,val_dataset], verbose=0, batch_size=batch_size)
train_nn = model.predict([vtk_train_dataset,train_dataset], verbose=0, batch_size=batch_size)

for i in np.squeeze(test_nn):
    # print('test_nn:', i)
    all_data['test_nn'].append(i[0] / label_scale)
for i in np.squeeze(val_nn):
    all_data['val_nn'].append(i[0] / label_scale)
for i in np.squeeze(train_nn):
    all_data['train_nn'].append(i[0] / label_scale)

test_labels = test_labels.numpy()
val_labels = val_labels.numpy()
train_labels = train_labels.numpy()

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

all_data['P_DNS']= test_labels[:,1:5]
all_data['P_NN'] = test_nn[:,1:5]

pickle_out = open('all_P_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()

print('save to: ', 'all_data_' + now + '.pickle', 'history_' + now + '.pickle', 'all_P_' + now + '.pickle')
print('the prediction of P and delta Psi_me is not the best model fit with lowest loss!')

exit(0)


from myplot import *
# plt_zxx(True)

# #--------------------------tf2.0----------------------------------------------
def plot_labels(plt, P_true, P_pred):
    plt.clf()
    plt.plot(P_true, P_pred, 'k.')
    amax = max(np.amax(P_pred), np.amax(P_true))
    amin = min(np.amin(P_pred), np.amin(P_true))
    plt.plot([amin, amax], [amin, amax], 'k-', lw=1.0)
    plt.axis('equal')
    plt.axis('square')
    # plt.gca().set_title('true vs predicted values')
    plt.xlabel('$\Delta\Psi_{\mathrm{mech,DNS}}$')
    plt.ylabel('$\Delta\Psi_{\mathrm{mech,KBNN}}$')
    plt.xlim([amin, amax])
    plt.ylim([amin, amax])
    plt.savefig('kbnn-dnn-m-dns-test.pdf', bbox_inches='tight', format='pdf')

# labels_nn = model.predict(dataset, verbose=0, batch_size=batch_size)
plot_labels(plt, test_labels.numpy()/label_scale, test_nn/label_scale)
plt.show()
# plt.clf()


# # var_np = test_dataset.to_numpy()

# test_derivative = derivative
# var_np = dataset.to_numpy()

# var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
# with tf.GradientTape() as g:
    # g.watch(var_tf)
    # y = model(var_tf)
# dy_dx = g.gradient(y, var_tf)
# dy_dx = dy_dx.numpy()[:,0:4]

# # print(str_form.format("dy_dx(NN-scaled): "), dy_dx[:, 0:4])
# print(str_form.format("dy_dx(NN-scaled): "), dy_dx)
print('scaled_derivative: ', test_derivative, np.shape(test_derivative))

# print(str_form.format("dy_dx(NN no scaled): "), dy_dx/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])])
# print('no_scale_derivative: ', test_derivative/ train_stats['std'].to_numpy()[0:len(test_derivative[0])] / label_scale)

# var_np = dataset.to_numpy()
tensor0 = tf.convert_to_tensor(vtk_test_dataset, dtype=tf.float32)
tensor1 = tf.convert_to_tensor(test_dataset.to_numpy(), dtype=tf.float32)
var_tf = [tensor0, tensor1]
with tf.GradientTape() as g:
    g.watch(var_tf)
    y = model(var_tf)
dy_dx = g.gradient(y, var_tf)
# print('dy_dx: 0', dy_dx[0])
print('dy_dx: 1', dy_dx[1])
# dy_dx = dy_dx.numpy()[:,0:4]
dy_dx = dy_dx[1]


def plot_P_one_field(plt, P_true, P_pred, ind0):
    plt.clf()
    plt.plot(P_true[:,ind0], P_pred[:,ind0],'k.')
    amax = max(np.amax(P_pred[:,ind0]), np.amax(P_true[:,ind0]))
    amin = min(np.amin(P_pred[:,ind0]), np.amin(P_true[:,ind0]))
    if(ind0 == 0):
        pre_fix = 'P11'
    elif(ind0 == 1):
        pre_fix = 'P12'
    elif(ind0 == 2):
        pre_fix = 'P21'
    elif(ind0 == 3):
        pre_fix = 'P22'
    else:
        raise ValueError('Unknown value for ind0')
    # plt.gca().set_title(pre_fix + 'true vs predicted values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.axis('equal')
    plt.axis('square')
    _ = plt.plot([amin, amax], [amin, amax], 'k-', lw=1.0)
    plt.xlim([amin, amax])
    plt.ylim([amin, amax])
    plt.savefig('kbnn-dnn-m-dns-'+pre_fix+'.pdf', bbox_inches='tight', format='pdf')

# plt.subplot(2,2,1)

# plot_P_one_field(plt, test_derivative, dy_dx, 0)
plot_P_one_field(plt, test_derivative/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], dy_dx/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], 0)

# plt.subplot(2,2,2)
# plot_P_one_field(plt, test_derivative, dy_dx, 1)
plot_P_one_field(plt, test_derivative/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], dy_dx/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], 1)

# plt.subplot(2,2,3)
# plot_P_one_field(plt, test_derivative, dy_dx, 2)
plot_P_one_field(plt, test_derivative/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], dy_dx/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], 2)

# plt.subplot(2,2,4)
# plot_P_one_field(plt, test_derivative, dy_dx, 3)
plot_P_one_field(plt, test_derivative/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], dy_dx/label_scale/train_stats['std'].to_numpy()[0:len(test_derivative[0])], 3)
print('...mean:', train_stats['mean'])
print('...std:', train_stats['std'])

# plt.show()


