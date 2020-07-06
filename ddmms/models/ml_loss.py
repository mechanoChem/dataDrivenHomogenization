import tensorflow as tf
import ddmms.models.ml_layers as ml_layers

################################## loss ############################################


def my_mse_combined_loss(loss_model):
    # Create a mse loss function
    def loss(y_true, y_pred):
        # raise ValueError(tf.shape(y_true), tf.shape(y_pred))
        a = loss_model(y_pred)
        b = loss_model(y_true)
        # print('y_pred: ', tf.shape(y_pred))
        # print('a: ', tf.shape(a))
        # print('b: ', tf.shape(b))
        # raise ValueError(tf.shape(y_true), tf.shape(y_pred), tf.shape(a), tf.shape(b))
        return tf.reduce_mean(tf.square(y_pred - y_true)) + tf.reduce_mean(
            tf.square(a - b))
        # return tf.reduce_mean(tf.square(y_pred - y_true))

    # Return a function
    return loss


def my_mse_loss():
    # Create a mse loss function
    def loss(y_true, y_pred):
        # raise ValueError(tf.shape(y_true), tf.shape(y_pred))
        return tf.reduce_mean(tf.square(y_pred - y_true))

    # Return a function
    return loss

def my_mse_loss_with_grad(BetaP=1000.0):

    def loss(y_true, y_pred):
        # raise ValueError(tf.shape(y_true), tf.shape(y_pred))
        # return tf.reduce_mean(tf.square(y_pred - y_true))

        S_NN = y_pred[:,1:5]
        S_NN = tf.reshape(S_NN,[-1,2,2])

        P_DNS = y_true[:,1:5]
        P_DNS = tf.reshape(P_DNS, [-1,2,2])

        F_DNS = y_true[:,5:9]
        F_DNS = tf.reshape(F_DNS, [-1,2,2])

        P_NN = tf.linalg.matmul(F_DNS, S_NN)
        # # raise ValueError(type(F_DNS), tf.shape(y_true), tf.shape(y_pred), tf.shape(F_DNS), tf.shape(S_NN), tf.shape(P_NN))
        # S_DNS = y_true[:,1:5]

        P_NN = tf.reshape(P_NN,[-1,4])
        P_DNS = tf.reshape(P_DNS,[-1,4])

        # return tf.reduce_mean(tf.square(y_pred[:,0] - y_true[:,0])) + 10000.0 * tf.reduce_mean(tf.square(y_pred[:,1:4] - y_true[:,1:4]))
        # return tf.reduce_mean(tf.square(y_pred[:,0] - y_true[:,0]))  + 10000.0 * tf.reduce_mean(tf.square(P_NN[:,0:1] - P_DNS[:,0:1]))  + 100000.0 * tf.reduce_mean(tf.square(P_NN[:,1:3] - P_DNS[:,1:3]))  + 10000.0 * tf.reduce_mean(tf.square(P_NN[:,3:4] - P_DNS[:,3:4]))
        return tf.reduce_mean(tf.square(y_pred[:,0] - y_true[:,0]))  + BetaP * tf.reduce_mean(tf.square(P_NN - P_DNS))  

        # return tf.reduce_mean(tf.square(y_pred[:,0] - y_true[:,0])) + 1.0 * tf.reduce_mean(tf.square(S_NN[:,1:5] - S_DNS[:,1:5]))

    # Return a function
    return loss



def build_loss(config, loss_model=None):

    ModelLoss = config['MODEL']['Loss']

    print('Avail Loss: ', [
        'mse', 'mae', 'sparse_categorical_crossentropy', 'binary_crossentropy'
    ])
    if (ModelLoss.lower() == "mse".lower()):
        if (tf.__version__[0:1] == '1'):
            loss = tf.keras.losses.MSE()
        else:
            loss = tf.keras.losses.MeanSquaredError()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss".lower()):
        loss = my_mse_loss()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_with_grad".lower()):
        loss = my_mse_loss_with_grad()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_exclude_margin".lower()):
        loss = my_mse_loss_exclude_margin()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_exclude_margin_with_laplacian_c".lower()):
        loss = my_mse_loss_exclude_margin_with_laplacian_c()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_exclude_margin_with_weak_laplacian_c".lower()):
        loss = my_mse_loss_exclude_margin_with_weak_laplacian_c()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_exclude_margin_with_div_p".lower()):
        loss = my_mse_loss_exclude_margin_with_Div_P()
        return loss
    elif (ModelLoss.lower() == "my_mse_loss_exclude_margin_with_weak_div_p".lower()):
        loss = my_mse_loss_exclude_margin_with_weak_Div_P()
        return loss
    elif (ModelLoss.lower() == "my_mse_strain_cnn3d_loss".lower()):
        loss = my_mse_strain_cnn3d_loss(loss_model)
        return loss
    elif (ModelLoss.lower() == "my_mse_combined_loss".lower()):
        loss = my_mse_combined_loss(loss_model)
        return loss
    elif (ModelLoss.lower() == "mae".lower()):
        loss = tf.keras.losses.MeanAbsoluteError()
        return loss
    elif (ModelLoss.lower() == "sparse_categorical_crossentropy".lower()):
        # loss = tf.losses.sparse_categorical_crossentropy(y_true, y_predict)
        loss = 'sparse_categorical_crossentropy'
        return loss
    elif (ModelLoss.lower() == "binary_crossentropy".lower()):
        loss = 'binary_crossentropy'
        return loss
    elif (ModelLoss.lower() == "user".lower()):
        raise ValueError('Model loss = ', ModelLoss,
                         ' is chosen, but is not implemented!')
    else:
        raise ValueError('Model loss = ', ModelLoss,
                         ' is chosen, but is not implemented!')


###################################################################################
