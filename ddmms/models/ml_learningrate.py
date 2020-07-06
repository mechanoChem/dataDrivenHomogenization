import tensorflow as tf
from tensorflow import keras
import ddmms.misc.ml_misc as ml_misc


############################### learning rate #####################################
def build_learningrate(config):
    LR = ml_misc.getlist_str(config['MODEL']['LearningRate'])
    # print('LR str: ', LR)
    if (len(LR) == 1):
        LearningRate = float(LR[0])
        print('--Decay in mono rate: rate = ', LearningRate)
    elif (len(LR) > 1):
        LR_type = LR[0]
        if (LR_type == 'mono'):
            LearningRate = float(LR[1])
        elif (LR_type == 'decay_exp'):
            initial_learning_rate = 0.001
            decay_steps = 1000
            decay_rate = 0.96
            initial_learning_rate = float(LR[1])
            if len(LR) > 2:
                decay_steps = int(LR[2])
            if len(LR) > 3:
                decay_rate = float(LR[3])
            print('--Decay in exponential rate: initial rate = ',
                  initial_learning_rate, ', decay steps = ', decay_steps,
                  ', decay_rate = ', decay_rate)

            # matlab script to check
            # learning_rate = 0.001;
            # decay_rate = 0.1;
            # decay_steps = 1000;
            # global_step = 1:1:2000;
            # decayed_learning_rate = learning_rate *  decay_rate .^ (global_step / decay_steps);
            # plot(global_step, decayed_learning_rate)

            if (ml_misc.get_package_version(tf.__version__)[0] == 1 and ml_misc.get_package_version(tf.__version__)[1] <= 13):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                global_step = tf.train.get_global_step()
                LearningRate = tf.train.exponential_decay(
                    initial_learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True)
            else:
                print("!!!! Caution: use Learning rate with care, there were occasions that tf1.13 should better performance on training. !!!")
                global_step = tf.Variable(0, name='global_step', trainable=False)
                # global_step = tf.train.get_global_step()

                LearningRate = tf.compat.v1.train.exponential_decay(
                    initial_learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=decay_rate,
                    staircase=True)
                # the following does not work very well for tf2.0/tf2.1 as comparing with tf1.13
                # LearningRate = tf.keras.optimizers.schedules.ExponentialDecay(
                    # initial_learning_rate,
                    # decay_steps=decay_steps,
                    # decay_rate=decay_rate,
                    # staircase=True)
        else:
            raise ValueError(
                'unknown choice for learning rate (mono, decay_exp): ', LR)

    return LearningRate


###################################################################################
