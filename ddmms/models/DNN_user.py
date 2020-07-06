import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class user_DNN_with_grad(tf.keras.Model):

    def __init__(self, input_shape, num_outputs, NodesList, Activation):
        super(user_DNN_with_grad, self).__init__()

        self.the_inputs = input_shape
        self.num_outputs = num_outputs

        self.the_layers = []
        self.the_layers.append(layers.Dense(NodesList[0], input_shape=input_shape, activation=Activation[0]))

        for i0 in range(1, len(NodesList)):
            self.the_layers.append(layers.Dense(NodesList[i0], activation=Activation[0]))

        # only 1 output is allowed, the rest are derivative data
        self.the_layers.append(layers.Dense(1))

    @tf.function(autograph=False)
    def call(self, inputs):

        with tf.GradientTape() as g:
            g.watch(inputs)
            y1 = self.the_layers[0](inputs)  #,
            for hd in self.the_layers[1:]:
                y2 = hd(y1)
                y1 = y2
        
        dy_dx = g.gradient(y2, inputs)
        # print(y2, dy_dx)
        return tf.concat([y2, dy_dx], 1)


class user_DNN_kregl1l2_gauss_grad(tf.keras.Model):

    def __init__(self, config, NodesList, Activation, train_dataset, train_labels, label_scale, train_stats):
        super(user_DNN_kregl1l2_gauss_grad, self).__init__()
        self.label_scale = label_scale
        self.train_stats_std = train_stats['std'].to_numpy()[0:3] # E11, E12, E22

        kreg_l1 = 0.0
        kreg_l2 = 0.0
        gauss_noise = 0.0

        try:
            kreg_l2 = float(config['MODEL']['KRegL2'])
            print('l2 regularize could potential cause the loss != mse')
        except:
            pass

        try:
            kreg_l1 = float(config['MODEL']['KRegL1'])
            print('l1 regularize could potential cause the loss != mse')
        except:
            pass

        try:
            gauss_noise = float(config['MODEL']['GaussNoise'])
            print('gauss noise could potential cause the loss != mse')
        except:
            pass

        if (kreg_l1 < 0 or kreg_l2 < 0 or gauss_noise < 0):
            raise ValueError(
                '***ERR***: regularizer or guass noise are < 0! They should be > 0. kreg_l1 = ',
                kreg_l1, 'kreg_l2 = ', kreg_l2, 'gauss_noise = ', gauss_noise)

        self.all_layers = []

        if kreg_l2 > 0 and kreg_l1 == 0:
            self.all_layers.append(
                layers.Dense(
                    NodesList[0],
                    activation=Activation[0],
                    input_shape=[len(train_dataset.keys())],
                    name='input',
                    kernel_regularizer=regularizers.l2(kreg_l2),
                    kernel_initializer='random_uniform'))
        elif kreg_l1 > 0 and kreg_l2 == 0:
            self.all_layers.append(
                layers.Dense(
                    NodesList[0],
                    activation=Activation[0],
                    input_shape=[len(train_dataset.keys())],
                    name='input',
                    kernel_regularizer=regularizers.l1(kreg_l1),
                    kernel_initializer='random_uniform'))
        elif kreg_l1 > 0 and kreg_l2 > 0:
            raise ValueError(
                'you can not use both l1 and l2 kernel regularizer: try just use l2'
            )
        else:
            self.all_layers.append(
                layers.Dense(
                    NodesList[0],
                    activation=Activation[0],
                    input_shape=[len(train_dataset.keys())],
                    name='input',
                    kernel_initializer='random_uniform'))

        # first hidden layer
        if gauss_noise > 0.0:
            self.all_layers.append(layers.GaussianNoise(gauss_noise))

        # remaining hidden layer
        for i0 in range(1, len(NodesList)):
            name_str = 'dense-' + str(i0)
            self.all_layers.append(
                layers.Dense(
                    NodesList[i0],
                    activation=Activation[i0],
                    name=name_str,
                    kernel_initializer='random_uniform'))

        # output layer
        self.all_layers.append(layers.Dense(1, name='output'))

    @tf.function(autograph=False)
    def call(self, inputs):
        with tf.GradientTape() as g:
            g.watch(inputs)
            y1 = self.all_layers[0](inputs)  #,
            for hd in self.all_layers[1:]:
                y2 = hd(y1)
                y1 = y2
        dy_dx = g.gradient(y2, inputs)/self.label_scale/self.train_stats_std


        # for penalize P
        return tf.concat([y2, dy_dx[:, 0:1], dy_dx[:, 1:2], dy_dx[:, 1:2], dy_dx[:, 2:3]], 1)
        # # for penalize S
        # return tf.concat([y2, dy_dx[:, 0:3]], 1)

