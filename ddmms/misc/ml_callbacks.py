import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys, os, datetime
import ddmms.misc.ml_misc as ml_misc


############################# call back ####################################
# Display training progress by printing a single dot for each completed epoch
class callback_PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):
        # print(logs)
        # try:
        # logs.keys().index['val_loss']
        # if epoch % 100 == 0: print('epoch: ', epoch, 'loss: ', logs['loss'], 'val_loss: ', logs['val_loss'])
        # except:
        # if epoch % 100 == 0: print('epoch: ', epoch, 'loss: ', logs['loss'])
        # pass
        if epoch % 100 == 0:
            print('epoch: ', epoch, 'loss: ', logs['loss'], 'val_loss: ',
                  logs['val_loss'])
        # print('.', end='')


# The patience parameter is the amount of epochs to check for improvement
def early_stop_callback(config):
    patience = int(config['MODEL']['EarlyStopPatience'])
    # lost_tol = float(config['MODEL']['TolLoss'])
    # print('lost_tol', lost_tol)

    es_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=patience
    )  #, ,  , baseline=lost_tol , restore_best_weights=True
    # print (es_callback)
    return es_callback


def check_point_callback(config):
    checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
    checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"  # it's difficult to include time info, as we need to restart simulation
    # checkpoint_dir = os.path.dirname(checkpoint_path)
    # print('checkpoint path & dir: ', checkpoint_path, checkpoint_dir)
    period = int(config['RESTART']['CheckPointPeriod'])
    verbose = int(config['MODEL']['Verbose'])
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        verbose=verbose,
        save_weights_only=True,
        # Save weights, every 5-epochs.
        period=period)
    return cp_callback


def tensor_board_callback(config):
    tensorboard_dir = config['OUTPUT']['TensorBoardDir'] + config['MODEL']['ParameterID']
    log_dir = tensorboard_dir + '/' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    if( ml_misc.get_package_version(tf.__version__)[0] == 1):
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)
    elif( ml_misc.get_package_version(tf.__version__)[0] == 2):
        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, profile_batch = 2)
    else:
        raise ValueError("unknown tf version for tensor board callback support")
    # print (tb_callback)
    return tb_callback


def terminate_on_nan_callback(config):
    nan_callback = keras.callbacks.TerminateOnNaN()
    return nan_callback


def callback_ReduceLrPlateau(config):
    # if use val_loss, it seems val_loss is not output in the tensorboard afterward
    # do not performance very well
    lr_0 = float(config['MODEL']['LearningRate'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, min_lr=1e-3 * lr_0, mode='min')
    return reduce_lr


#  current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#  train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
#  test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
#  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#  test_summary_writer = tf.summary.create_file_writer(test_log_dir)
#
#  print('-----before save weight-----')
#  model.save_weights(checkpoint_path.format(epoch=0))
#  print('---- after save weight-----')
#
#  [cp_callback, tensorboard_callback, early_stop, PrintDot()]


def build_callbacks(config):
    callbacks = []
    callback_names = ml_misc.getlist_str(config['MODEL']['CallBacks'])
    # print(callback_names, callback_names.index('checkpoint'), callback_names.index('tensorboard'), callback_names.index('earlystop'))

    if 'checkpoint' in callback_names:
        callbacks.append(check_point_callback(config))
    # print('---1---', callbacks)

    if 'tensorboard' in callback_names:
        callbacks.append(tensor_board_callback(config))
    # print('---2---', callbacks)

    if 'earlystop' in callback_names:
        callbacks.append(early_stop_callback(config))
    # print('---3---', callbacks)

    if 'printdot' in callback_names:
        callbacks.append(callback_PrintDot())

    if 'nan' in callback_names:
        callbacks.append(terminate_on_nan_callback(config))

    if 'reducelrplateau' in callback_names:
        callbacks.append(callback_ReduceLrPlateau(config))

    # print('---4---', callbacks, len(callbacks))
    return callbacks


###############################################################################
