import ddmms.misc.ml_misc as ml_misc
import ddmms.math.ml_math as ml_math
import ddmms.preprocess.ml_preprocess as ml_preprocess
import tensorflow as tf
import numpy as np


def get_dummy_data(num):
    """ get dummy_data for num of fields """
    one_I = [1] * (num + 1)
    data2D = [one_I, one_I]
    I = csvDf(data2D)
    return I


def get_I(test_dataset):
    """ get I for different cases """
    one_data = test_dataset[0:1].to_numpy()
    if (len(one_data[0]) == 1):
        data2D = [[0, 1], [1, 1]]  # just 1
    elif (len(one_data[0]) == 2):
        data2D = [[0, 1, 2], [1, 1, 1]]  # just 1 1
    elif (len(one_data[0]) == 3):
        data2D = [[0, 1, 2, 3], [1, 1, 1, 0]]  # 11, 22, 12: voigt notation
    elif (len(one_data[0]) == 4):
        data2D = [[0, 1, 2, 3, 4], [1, 1, 0, 0, 1]]  # 11, 12, 21, 22
    elif (len(one_data[0]) == 6):
        data2D = [[0, 1, 2, 3, 4, 5, 6], [1, 1, 1, 1, 0, 0,
                                          0]]  # 11, 22, 33, 12, 23, 13
    elif (len(one_data[0]) == 9):
        data2D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                  [1, 1, 0, 0, 0, 1, 0, 0, 0,
                   1]]  # 11, 12, 13, 21, 22, 23, 31, 32, 33
    else:
        one_I = [1] * (len(one_data[0]) + 1)
        data2D = [one_I, one_I]
        print('return a crazy 1 in get_I():', data2D)
        # raise ValueError('Do not know which I to return, check it!')
    I = csvDf(data2D)
    return I


def get_zeros(test_dataset):
    """ get zero for different cases """
    zeros = test_dataset[0:1] * 0.0
    return zeros


def get_X(config, test_dataset, str_form):
    """ get x for different cases """
    one_data = test_dataset[0:1].to_numpy()
    X = ml_misc.getlist_float(config['OUTPUT']['SpecialX'])
    if len(X) == 0:
        return []
    else:
        print(str_form.format("(no-scale) x: "), X)
        if (len(X) != len(one_data[0])):
            raise ValueError(
                'The special X given in the input has different length from the features of NN!'
            )
        else:
            data2D = [[], []]
            data2D[0] = [x for x in range(0, len(one_data[0]) + 1)]
            data2D[1] = [1]
            data2D[1].extend(X)

        X = csvDf(data2D)
        return X


def special_operations(config, model, test_dataset, test_labels,
                       predicted_labels, test_derivative, str_form,
                       train_stats):
    """ special operation for the code """
    special_output_list = ml_misc.getlist_str(
        config['OUTPUT']['SpecialOperation'])
    if len(special_output_list) == 0:
        return

    if (int(config['TEST']['DataNormalization']) == 0):
        normalization_flag = False
    else:
        normalization_flag = True

    I = get_I(test_dataset)
    zeros = get_zeros(test_dataset)
    X = get_X(config, test_dataset, str_form)

    if (normalization_flag):
        std = train_stats['std'].to_numpy()  # pay attention to the types.
        mean = train_stats['mean'].to_numpy()  # pay attention to the types.
        # print( 'X before norm', X)
        if (len(X) > 0):
            X = (X - mean) / std
        I = (I - mean) / std
        zeros = (zeros - mean) / std
        print('X after norm', X)
        print('--- mean:', mean)
        print('---  std:', std)

    for s1 in special_output_list:

        dy_dx = 0
        d2y_dx2 = 0
        if s1.lower() == 'eval_of_zeros':
            y = eval_of_X(model, zeros, str_form, 'zeros')
        elif s1.lower() == 'grad1_of_zeros':
            dy_dx = gradient_of_X(model, zeros, str_form, 'zeros')
        elif s1.lower() == 'grad2_of_zeros':
            d2y_dx2 = gradient2_of_X(model, zeros, str_form, 'zeros')
        elif s1.lower() == 'jacobian_of_zeros':
            dy_dx = jacobian_of_X(model, zeros, str_form, 'zeros')

        elif s1.lower() == 'eval_of_i':
            y = eval_of_X(model, zeros, str_form, 'I')
        elif s1.lower() == 'grad1_of_i':
            dy_dx = gradient_of_X(model, zeros, str_form, 'I')
        elif s1.lower() == 'grad2_of_i':
            d2y_dx2 = gradient2_of_X(model, zeros, str_form, 'I')
        elif s1.lower() == 'jacobian_of_i':
            dy_dx = jacobian_of_X(model, zeros, str_form, 'I')

        elif s1.lower() == 'eval_of_x':
            y = eval_of_X(model, X, str_form, 'x')
        elif s1.lower() == 'grad1_of_x':
            dy_dx = gradient_of_X(model, X, str_form, 'x')
        elif s1.lower() == 'grad2_of_x':
            d2y_dx2 = gradient2_of_X(model, zeros, str_form, 'x')
        elif s1.lower() == 'jacobian_of_x':
            dy_dx = jacobian_of_X(model, zeros, str_form, 'x')

        else:
            raise ValueError("The special operation: ", s1,
                             " is not implemented!!!")

        if (normalization_flag and dy_dx != 0):
            dy_dx = dy_dx / std / float(config['TEST']['LabelScale'])
            print(str_form.format("dy_dx(NN-no-scale s): "), dy_dx)

        if (normalization_flag and d2y_dx2 != 0):
            d2y_dx2 = tf.squeeze(d2y_dx2)
            # print(type(d2y_dx2), std, type(std))
            d2y_dx2 = d2y_dx2 / std / float(config['TEST']['LabelScale'])
            std_transpose = np.reshape(std, (len(std), 1))
            # print(std_transpose)
            d2y_dx2 = d2y_dx2 / std_transpose
            print(str_form.format("d2y_dx2(NN-no-scale s): "), d2y_dx2)


def eval_of_X(model, var_int, str_form, str0=''):
    """ eval of X """
    print(str_form.format("eval of " + str0 + ' : '), var_int.to_numpy())
    y = model.predict(var_int)
    print(str_form.format("y(NN): "), y)
    return y


def gradient_of_X(model, var_int, str_form, str0=''):
    """ gradient of X """
    print(str_form.format("grad1 of " + str0 + " : "))
    dy_dx = ml_math.compute_grad1(model, var_int, 1, str_form)
    return dy_dx


def gradient2_of_X(model, var_int, str_form, str0=''):
    """ gradient 2 of X """
    print(str_form.format("grad2 of " + str0 + " : "))
    d2y_dx2 = ml_math.compute_grad2(model, var_int, 1, str_form)
    return d2y_dx2


def jacobian_of_X(model, var_int, str_form, str0=''):
    """ jacobian of X """
    print(str_form.format("jacobian of " + str0 + " : "))
    dy_dx = ml_math.compute_jacobian1(model, var_int, 1, str_form)
    return dy_dx


def test_csvDF():
    data = [['', 'a', 'b', 'c'], ['row1', 'row1cola', 'row1colb', 'row1colc'],
            ['row2', 'row2cola', 'row2colb',
             'row2colc'], ['row3', 'row3cola', 'row3colb', 'row3colc']]
    print(csvDF(data))
    data2D = [[0, 0, 1], [0, 1.0, 0.0], [1, 0.0, 1.0]]
    print(csvDF(data2D))
    data3D = [[0, 0, 1, 2], [0, 1.0, 0.0, 0.0], [1, 0.0, 1.0, 0.0],
              [2, 0.0, 0.0, 1.0]]
    print(csvDF(data3D))


def csvDf(dat, **kwargs):
    import pandas as pd
    from numpy import array
    data = array(dat)
    if data is None or len(data) == 0 or len(data[0]) == 0:
        return None
    else:
        return pd.DataFrame(
            data[1:, 1:], index=data[1:, 0], columns=data[0, 1:], **kwargs)
