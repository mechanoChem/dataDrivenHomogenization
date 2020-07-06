import tensorflow as tf
from tensorflow.keras import layers
import ddmms.models.ml_learningrate as ml_learningrate


############################### optimizer #########################################
"""
    todo: ( not urgent, and will be easy to change)
    1. Define a class object: 
      input: (i) the optimizer name, (ii) learning rate
"""
def build_optimizer(config):

    ModelOptimizer = config['MODEL']['Optimizer']
    LearningRate = ml_learningrate.build_learningrate(config)

    print('Avail Optimizer: ',
          ['adam', 'sgd', 'adadelta', 'gradientdescentoptimizer'])

    if (ModelOptimizer.lower() == "adam".lower()):
        optimizer = tf.keras.optimizers.Adam(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "sgd".lower()):
        optimizer = tf.keras.optimizers.SGD(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "adadelta".lower()):
        optimizer = tf.keras.optimizers.Adadelta(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "gradientdescentoptimizer".lower()):
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(LearningRate)
        return optimizer
    elif (ModelOptimizer.lower() == "user".lower()):
        raise ValueError('Model optimizer = ', ModelOptimizer,
                         ' is chosen, but is not implemented!')
    else:
        raise ValueError('Model optimizer = ', ModelOptimizer,
                         ' is chosen, but is not implemented!')


###################################################################################
