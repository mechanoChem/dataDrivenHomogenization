# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import numpy as np

import sys, os, datetime
import ddmms.misc.ml_misc as ml_misc
import ddmms.models.ml_layers as ml_layers


##################################### model ##########################################

def pure_DNN(config, train_dataset, train_labels, NodesList, Activation):
    model = keras.Sequential()
    # activity_regularizer
    # bias_regularizer
    # kernel_regularizer

    # first hidden layer
    # model.add(layers.Dense(NodesList[0], activation=Activation[0], input_shape=[len(train_dataset.keys())], name='input', kernel_initializer='random_uniform',bias_initializer='random_uniform'))
    model.add(
        layers.Dense(
            NodesList[0],
            activation=Activation[0],
            input_shape=[len(train_dataset.keys())],
            name='input',
            kernel_regularizer=regularizers.l2(0.001)))
    # model.add(layers.Dense(NodesList[0], activation=Activation[0], input_shape=[len(train_dataset.keys())], name='input') )
    # model.add(layers.Dense(NodesList[0], activation=Activation[0], input_shape=[len(train_dataset.keys())], name='input', activity_regularizer=regularizers.l2(0.001) ) )
    model.add(layers.GaussianNoise(0.001))

    # remaining hidden layer
    for i0 in range(1, len(NodesList)):
        name_str = 'dense-' + str(i0)
        # model.add(layers.Dense(NodesList[i0], activation=Activation[i0], name=name_str,kernel_initializer='random_uniform',bias_initializer='random_uniform'))
        # model.add(layers.Dense(NodesList[i0], activation=Activation[i0], name=name_str))
        model.add(
            layers.Dense(
                NodesList[i0], activation=Activation[i0], name=name_str))
        # model.add(layers.Dropout(0.1)) # test: not working well, too slow convergence, discarded
        # it seems the gradient will also decrease 10% if rate=0.1, and 50% if rate=0.5
        model.add(layers.GaussianNoise(0.001))

    # output layer
    model.add(layers.Dense(len(train_labels.keys()), name='output'))

    return model

def user_DNN_kregl1l2_gauss_grad(config, train_dataset, train_labels, NodesList,
                       Activation, train_stats):
    import ddmms.models.DNN_user as DNN_user
    print(
        'Warning Msg: user_DNN_kregl1l2_gauss_grad is hard coded with fixed label numbers = 1!'
    )

    label_scale = float(config['TEST']['LabelScale'])
    model = DNN_user.user_DNN_kregl1l2_gauss_grad(config, NodesList, Activation, train_dataset,train_labels, label_scale, train_stats)
    return model

def DNN_kregl1l2_gauss(config, train_dataset, train_labels, NodesList,
                       Activation):
    model = keras.Sequential()
    # activity_regularizer
    # bias_regularizer
    # kernel_regularizer
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

    if kreg_l2 > 0 and kreg_l1 == 0:
        model.add(
            layers.Dense(
                NodesList[0],
                activation=Activation[0],
                input_shape=[len(train_dataset.keys())],
                name='input',
                kernel_regularizer=regularizers.l2(kreg_l2),
                kernel_initializer='random_uniform'))
    elif kreg_l1 > 0 and kreg_l2 == 0:
        model.add(
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
        model.add(
            layers.Dense(
                NodesList[0],
                activation=Activation[0],
                input_shape=[len(train_dataset.keys())],
                name='input',
                kernel_initializer='random_uniform'))

    # first hidden layer
    if gauss_noise > 0.0:
        model.add(layers.GaussianNoise(gauss_noise))

    # remaining hidden layer
    for i0 in range(1, len(NodesList)):
        name_str = 'dense-' + str(i0)

        model.add(
            layers.Dense(
                NodesList[i0],
                activation=Activation[i0],
                name=name_str,
                kernel_initializer='random_uniform'))

    # output layer
    model.add(layers.Dense(len(train_labels.keys()), name='output'))

    return model


def add_input_layer(config,
                    model,
                    train_dataset,
                    Name,
                    Node,
                    Act,
                    padding='valid'):
    # [1000, 28, 28, 1] -> [28, 28, 1]
    # print(train_dataset)
    # print(tf.shape(train_dataset.to_numpy()))

    if (tf.__version__[0:1] == '1'):
        # input_shape = train_dataset.get_shape().as_list()[1:]
        input_shape = train_dataset.shape[1:]
    elif (tf.__version__[0:1] == '2'):
        try:
            input_shape = tf.shape(train_dataset).numpy()[1:]
        except:
            input_shape = tf.shape(train_dataset.to_numpy()).numpy()[1:]
    else:
        raise ValueError("Unknown tensorflow version: ", tf.__version__)

    print('input_shape:', input_shape)

    if (Name.lower().find('conv2d') >= 0):
        # _3_3 -> [3,3]
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1]
        if len(kernel) == 4:
            strides = kernel[2:4]
        model.add(
            layers.Conv2D(
                Node,
                kernel[0:2],
                strides=strides,
                activation=Act,
                input_shape=input_shape,
                padding=padding,
                name='input'))
    elif (Name.lower().find('conv3d') >= 0):
        # _3_3 -> [3,3]
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1, 1]
        if len(kernel) == 6:
            strides = kernel[3:6]
        model.add(
            layers.Conv3D(
                Node,
                kernel[0:3],
                strides=strides,
                activation=Act,
                input_shape=input_shape,
                padding=padding,
                name='input'))

    elif (Name.lower().find('random') >= 0):
        model.add(ml_layers.LayerFillRandomNumber(name='input'))
        # model.add(layers.Dense(Node, activation=Act, input_shape=input_shape, name='input'))
    elif (Name.lower().find('dense') >= 0):
        model.add(
            layers.Dense(
                Node, activation=Act, input_shape=input_shape, name='input'))
    elif (Name.lower().find('lstm') >= 0):
        print('!!!!input_shape:', input_shape, ' should be [1, 1]!!!!!')
        model.add(layers.LSTM(Node, input_shape=(1, 1), return_sequences=True))
    elif (Name.lower().find('gru') >= 0):
        print('!!!!input_shape:', input_shape, ' should be [1, 1]!!!!!')
        model.add(layers.GRU(Node, input_shape=(1, 1), return_sequences=True))
    else:
        raise ValueError('The first layer can only be conv2d, your input is: ',
                         Name)


def add_one_layer(config, model, Name, Node, Act, padding='valid', tf_name=''):
    if (Act == 'None'):
        Act = None
    # print('Name:', tf_name)
    if (Name.lower().find('maxpooling2d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        print('kernel:', kernel, 'padding:', padding)
        model.add(layers.MaxPooling2D(kernel, padding=padding, name=tf_name))
    elif (Name.lower().find('upsampling2d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        model.add(layers.UpSampling2D(kernel, name=tf_name))
        # print('kernel:', kernel)
    elif (Name.lower().find('conv2d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1]
        if len(kernel) == 4:
            strides = kernel[2:4]
        model.add(
            layers.Conv2D(
                Node,
                kernel[0:2],
                strides=strides,
                activation=Act,
                padding=padding,
                name=tf_name))
        print('conv2d: ', Node, kernel, strides, Act, padding, tf_name)
    elif (Name.lower().find('conv3d') >= 0):
        kernel = Name.split('_')[1:]
        kernel = [int(x) for x in kernel]
        strides = [1, 1, 1]
        if len(kernel) == 6:
            strides = kernel[3:6]
        model.add(
            layers.Conv3D(
                Node,
                kernel[0:3],
                strides=strides,
                activation=Act,
                padding=padding,
                name=tf_name))
        # print('kernel:', kernel)
    elif (Name.lower().find('flatten') >= 0):
        model.add(layers.Flatten(name=tf_name))
    elif (Name.lower().find('dense') >= 0):
        model.add(layers.Dense(Node, activation=Act, name=tf_name))
    elif (Name.lower().find('lstm') >= 0):
        # model.add(layers.LSTM(Node,return_sequences = True))
        model.add(layers.LSTM(Node))
    elif (Name.lower().find('gru') >= 0):
        model.add(layers.GRU(Node))
    elif (Name.lower().find('reshape') >= 0):
        shapes = Name.split('_')[1:]
        shapes = [int(x) for x in shapes]
        model.add(ml_layers.LayerReshape(shapes, name='reshape'))
    else:
        raise ValueError('The layer name: ', Name, ' is not programmed!')


def add_output_layer(config, model, train_labels):
    print('****activation:---')
    Act = None
    try:
        Act = config['MODEL']['OutputLayerActivation']
        print('****activation:---', Act)
        if (Act == ''):
            Act = None
    except:
        pass


    num_output = 1
    try:
        num_output = len(train_labels.keys())
    except:
        try:
            num_output = len(train_labels[0])
        except:
            pass
        pass
    print('----activation:---', Act)

    output_layer = 'Dense'
    try:
        output_layer = config['MODEL']['OutputLayer']
    except:
        pass
    if (output_layer == 'Dense'):
        model.add(layers.Dense(num_output, activation=Act, name='output'))
    elif (output_layer == 'No'):
        return
    elif (output_layer == 'Conv3D'):
        """ not checked """
        # model.add(layers.Conv3D(Node, kernel[0:3], strides=strides, activation=Act, input_shape=input_shape, padding=padding,  name='input'))


def CNN_autoencoder(config, train_dataset, train_labels, NodesList, Activation,
                    LayerName, Padding):
    model = keras.Sequential()

    add_input_layer(config, model, train_dataset, LayerName[0], NodesList[0],
                    Activation[0], Padding[0])
    for i0 in range(1, len(NodesList)):
        add_one_layer(
            config,
            model,
            LayerName[i0],
            NodesList[i0],
            Activation[i0],
            Padding[i0],
            tf_name=LayerName[i0] + '-' + str(i0))
    return model


def pure_CNN(config, train_dataset, train_labels, NodesList, Activation,
             LayerName, Padding):
    model = keras.Sequential()

    add_input_layer(config, model, train_dataset, LayerName[0], NodesList[0],
                    Activation[0], Padding[0])
    for i0 in range(1, len(NodesList)):
        add_one_layer(
            config,
            model,
            LayerName[i0],
            NodesList[i0],
            Activation[i0],
            Padding[i0],
            tf_name=LayerName[i0] + '-' + str(i0))
    # no output layer, as a softmax is needed to convert the labels
    ##########add_output_layer(config, model, train_labels)
    #Convert array of indices to 1-hot encoded numpy array
    return model


def CNN_supervise(config, train_dataset, train_labels, NodesList, Activation,
                  LayerName, Padding):
    model = keras.Sequential()

    add_input_layer(config, model, train_dataset, LayerName[0], NodesList[0],
                    Activation[0], Padding[0])
    for i0 in range(1, len(NodesList)):
        add_one_layer(
            config,
            model,
            LayerName[i0],
            NodesList[i0],
            Activation[i0],
            Padding[i0],
            tf_name=LayerName[i0] + '-' + str(i0))
    add_output_layer(config, model, train_labels)
    print('!!!supervise!!!')

    return model


def user_DNN_with_grad(config, train_dataset, train_labels, NodesList, Activation):
    import ddmms.models.DNN_user as DNN_user
    print(
        'Warning Msg: user_DNN_with_grad is hard coded with fixed activation softplus! For testing purpose only, use with caution!'
    )
    # print('features:',train_dataset)
    # print('labels:',train_labels)
    model = DNN_user.user_DNN_with_grad([len(train_dataset.keys())],
                             len(train_labels.keys()), NodesList, Activation)
    # cannot build model either with model.build or model.layers[1].build(), will move model summary() after fitting
    return model


def KBNN_user_test(config, train_dataset, train_labels, NodesList, Activation):
    import ddmms.models.KBNN as KBNN
    print(
        'Warning Msg: KBNN_user_test ... ! For testing purpose only, use with caution!'
    )
    model = KBNN.KBNN_Model([len(train_dataset.keys())], len(
        train_labels.keys()), NodesList, Activation, config)
    return model


def build_model(config, train_dataset, train_labels, set_non_trainable=False, train_stats=None):
    ModelArchitect = config['MODEL']['ModelArchitect']

    # print('hl: ',config['MODEL']['hiddenlayernumber'],
    # 'nl:', config['MODEL']['nodeslist'],
    # 'lr: ', config['MODEL']['learningrate'],
    # 'act: ', config['MODEL']['activation'])

    NodesList = ml_misc.getlist_int(config['MODEL']['NodesList'])
    Activation = ml_misc.getlist_str(config['MODEL']['Activation'])

    if (len(NodesList) != len(Activation)):
        raise ValueError(
            'In the config file, number of NodesList != Activation list with NodesList = ',
            NodesList, ' and Activation = ', Activation)

    if (ModelArchitect.lower() == "pure_DNN".lower()):
        model = pure_DNN(config, train_dataset, train_labels, NodesList,
                         Activation)
    elif (ModelArchitect.lower() == "dnn_kregl1l2_gauss".lower()):
        model = DNN_kregl1l2_gauss(config, train_dataset, train_labels,
                                   NodesList, Activation)
    elif (ModelArchitect.lower() == "user_dnn_kregl1l2_gauss_grad".lower()):
        model = user_DNN_kregl1l2_gauss_grad(config, train_dataset, train_labels,
                                   NodesList, Activation, train_stats)
    elif (ModelArchitect.lower().find('cnn') >= 0):
        LayerName = ml_misc.getlist_str(config['MODEL']['LayerName'])
        Padding = ml_misc.getlist_str(config['MODEL']['Padding'])
        if (len(NodesList) != len(LayerName) or len(NodesList) != len(Padding)):
            raise ValueError(
                'In the config file, number of NodesList != LayerName with NodesList = ',
                NodesList, len(NodesList), ' and LayerName = ', LayerName,
                len(LayerName), 'and Padding = ', Padding, len(Padding))
        if (ModelArchitect.lower() == "pure_CNN".lower()):
            model = pure_CNN(config, train_dataset, train_labels, NodesList,
                             Activation, LayerName, Padding)
        elif (ModelArchitect.lower() == "CNN_user_supervise".lower()):
            model = CNN_user_supervise(config, train_dataset, train_labels,
                                       NodesList, Activation, LayerName,
                                       Padding)
        elif (ModelArchitect.lower() == "CNN_supervise".lower()):
            model = CNN_supervise(config, train_dataset, train_labels,
                                  NodesList, Activation, LayerName, Padding)
        else:
            raise ValueError('Model architect = ', ModelArchitect,
                             ' is chosen, but is not implemented!')
    elif (ModelArchitect.lower() == "pure_DNN_user".lower()):
        model = pure_DNN_user(config, train_dataset, train_labels, NodesList,
                              Activation)
    elif (ModelArchitect.lower() == "user_DNN_with_grad".lower()):
        model = user_DNN_with_grad(config, train_dataset, train_labels, NodesList,
                              Activation)
    elif (ModelArchitect.lower() == "KBNN_user_test".lower()):
        model = KBNN_user_test(config, train_dataset, train_labels, NodesList,
                               Activation)
    else:
        raise ValueError('Model architect = ', ModelArchitect,
                         ' is chosen, but is not implemented!')

    # print(type(train_dataset))

    if set_non_trainable:
        model.trainable = False

    return model
