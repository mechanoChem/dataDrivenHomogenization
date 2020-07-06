import tensorflow as tf
import numpy as np
import ddmms.misc.ml_misc as ml_misc
import ddmms.postprocess.ml_postprocess as ml_postprocess


def analytical_integration(var_in, int_y):
    z = []
    x = [s[0] for s in var_in]
    x = np.array(x)
    if (len(var_in[0]) == 1):
        z = x + x * x
    elif (len(var_in[0]) == 2):
        y = [s[1] for s in var_in]
        y = np.array(y)
        z = 2 * x + 3 * y + 4 * x * y + 5 * x * x + 6 * y * y
        # print('z=',z)
        if (len(int_y[0]) == 2):
            z1 = 1 + 2 * x + 3 * y + 4 * x * y
            z2 = 2 + 3 * x * y + 4 * x * x * y * y
            z = zip(z1, z2)
            z = [list(s) for s in z]
            z = np.array(z)
    return z


def output_gradients(config, model, test_dataset, test_labels, predicted_labels,
                     test_derivative, str_form, train_stats):
    """ top level of interface to output gradients """
    num_grad_output = int(config['OUTPUT']['NumGradientCalc'])

    if (int(config['TEST']['DataNormalization']) == 0):
        normalization_flag = False
    else:
        normalization_flag = True
        label_scale = float(config['TEST']['LabelScale'])

    grad_output_list = ml_misc.getlist_str(config['OUTPUT']['GradientOutput'])
    for o1 in grad_output_list:
        if o1.lower() == 'grad0':
            compute_grad0(test_dataset, test_labels, predicted_labels,
                          num_grad_output, str_form)
        elif o1.lower() == 'grad1':
            if (len(test_derivative) > 0):
                output_grad1(test_derivative, num_grad_output, str_form)

            dy_dx = compute_grad1(model, test_dataset, num_grad_output,
                                  str_form)
            if normalization_flag:
                dy_dx_no_scale = dy_dx / label_scale / train_stats[
                    'std'].to_numpy()
                print(str_form.format("dy_dx(NN-no-scale): "),
                      dy_dx_no_scale[:, 0:4])
                # print('---dy_dx:', dy_dx)
                # print('---label_scale:', label_scale)
                # print('---std:', train_stats['std'].to_numpy())

            if len(predicted_labels[0]) > 1:
                compute_jacobian1(model, test_dataset, num_grad_output,
                                  str_form)
        elif o1.lower() == 'grad2':
            compute_grad2(model, test_dataset, num_grad_output, str_form)
            ##compute_jacobian2(model, test_dataset, num_grad_output, str_form) # not working
        elif o1.lower() == 'grad3':
            compute_grad3(model, test_dataset, num_grad_output, str_form)
        elif o1.lower() == 'int1':
            compute_int1(model, test_dataset, num_grad_output, str_form)
        elif o1.lower() == 'plot_int1':
            plot_int1(model, test_dataset, num_grad_output, str_form)
        else:
            raise ValueError('unknown gradient output name in: ',
                             grad_output_list)

    # # print('-------------------------', dy_dx[:,0], test_derivative[:,0])
    # ml_postprocess.plot_scatter_two_fields(dy_dx[:,0], test_derivative[:,0], savefig=True, filename='P11.png')
    # ml_postprocess.plot_scatter_two_fields(dy_dx[:,1], test_derivative[:,1], savefig=True, filename='P12.png')
    # ml_postprocess.plot_scatter_two_fields(dy_dx[:,2], test_derivative[:,2], savefig=True, filename='P21.png')
    # ml_postprocess.plot_scatter_two_fields(dy_dx[:,3], test_derivative[:,3], savefig=True, filename='P22.png')


def compute_grad0(var_in, var_out, var_pre, num, str_form="{}"):
    """ print x, y, and y prediction """
    # , ' prediction ', predicted_labels[0], ' test_labels ', test_labels['Psi'][0]
    new_var = var_in[0:num]
    try:
        var_np = new_var.to_numpy()
    except:
        var_np = new_var.numpy()

    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    # print(str_form.format("feature name: "), var_in.index()) # not working
    print(str_form.format("x(scaled): "), var_tf)

    new_var = var_out[0:num]

    try:
        var_np = new_var.to_numpy()
    except:
        var_np = new_var.numpy()

    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    print(str_form.format("y(DNS-scaled): "), var_tf)

    var_np = var_pre[0:num]
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    print(str_form.format("y(NN-scaled): "), var_tf)


def output_grad1(dydx, num, str_form="{}"):
    """ output the gradient of one set of data """
    if (len(dydx) > 0):
        var_np = dydx[0:num]
        var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
        print(str_form.format("dy_dx(DNS-scaled): "), var_tf)
    else:
        print(
            "No derivative information from input data file. Nothing to print for output_grad1()!!!"
        )


def compute_grad1(fcn, var, num, str_form="{}"):
    """ compute the gradient of one set of data """
    # assuming var is in pandas framwork data type
    # this fcn could be loss, could be model, could be anything
    new_var = var[0:num]
    var_np = new_var.to_numpy()
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(var_tf)
        y = fcn(var_tf)

    dy_dx = g.gradient(y, var_tf)
    # print(str_form.format("x: "), var_tf)
    print(str_form.format("dy_dx(NN-scaled): "), dy_dx[:, 0:4])
    return dy_dx
    # exit(0)


def compute_grad2(fcn, var, num, str_form="{}"):
    """ compute the grad.grad of one set of data """
    # assuming var is in pandas framwork data type
    # this fcn could be loss, could be model, could be anything
    new_var = var[0:num]
    var_np = new_var.to_numpy()
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(var_tf)
        with tf.GradientTape() as gg:
            gg.watch(var_tf)
            y = fcn(var_tf)
        dy_dx = gg.gradient(y, var_tf)
    # d2y_dx2 = g.gradient(dy_dx, var_tf)
    d2y_dx2 = g.jacobian(dy_dx, var_tf)
    print(str_form.format("d2y_dx2(NN-scaled): "), d2y_dx2)
    return d2y_dx2


def compute_grad3(fcn, var, num, str_form="{}"):
    """ compute the grad^3 of one set of data """
    # assuming var is in pandas framwork data type
    # this fcn could be loss, could be model, could be anything
    new_var = var[0:num]
    var_np = new_var.to_numpy()
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(var_tf)
        with tf.GradientTape() as gg:
            gg.watch(var_tf)
            with tf.GradientTape() as ggg:
                ggg.watch(var_tf)
                y = fcn(var_tf)
            dy_dx = ggg.gradient(y, var_tf)
        d2y_dx2 = gg.gradient(dy_dx, var_tf)
    d3y_dx3 = g.gradient(d2y_dx2, var_tf)
    print(str_form.format("d3y_dx3(NN-scaled): "), d3y_dx3)


def compute_jacobian1(fcn, var, num, str_form="{}"):
    """ compute jacobian of one set of data """
    # assuming var is in pandas framwork data type
    # this fcn could be loss, could be model, could be anything
    new_var = var[0:num]
    var_np = new_var.to_numpy()
    # var_np = np.array([[0,0,0,0]])
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(var_tf)
        y = fcn(var_tf)

    dy_dx = g.jacobian(y, var_tf)
    # print(str_form.format("x: "), var_tf)
    # print(str_form.format("y: "), y)
    print(str_form.format("dy_dx (jacobian-NN-scaled): "), dy_dx)
    return dy_dx
    # exit(0)


def compute_jacobian2(fcn, var, num, str_form="{}"):
    """ compute jacobian2 of one set of data """

    new_var = var[0:num]
    var_np = new_var.to_numpy()
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(var_tf)
        with tf.GradientTape() as gg:
            gg.watch(var_tf)
            y = fcn(var_tf)
        dy_dx = gg.jacobian(y, var_tf)
    d2y_dx2 = g.jacobian(dy_dx, var_tf)
    print(str_form.format("d2y_dx2(jacobian-NN-scaled): "), d2y_dx2)
    return d2y_dx2
    # exit(0)


def compute_int1(fcn, var, num, str_form="{}"):
    """ compute the integration of one set of data """
    # assuming var is in pandas framwork data type
    # this fcn could be loss, could be model, could be anything
    new_var = var[0:num]
    var_np = new_var.to_numpy()
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)

    int_y = fcn.output_integration(var_tf)
    print(str_form.format("int(y)(NN): "), int_y)
    return int_y


def plot_int1(fcn, var, num, str_form="{}"):
    """ plot the integration of one set of data """
    import matplotlib.pyplot as plt
    # assuming var is in pandas framwork data type
    # this fcn could be loss, could be model, could be anything
    var_np = var.to_numpy()
    var_tf = tf.convert_to_tensor(var_np, dtype=tf.float32)
    int_y = fcn.output_integration(
        var_tf)  # tensor with shape of (num, 1), (num, 2), (num, 3), etc.

    # print('x in plot_int1: ', len(var_np), len(var_np[0]), var_np)
    # print('int_y in plot_int1: ', len(int_y), len(int_y[0]), int_y)
    if (len(var_np[0]) == 1):
        plt.scatter(var_np, int_y, label='NN')
    elif (len(var_np[0]) == 2):
        x = [s[0] for s in var_np]
        y = [s[1] for s in var_np]
        if (len(int_y[0]) == 1):
            plot_3d(x, y, int_y, marker='o', title='Integration')
        elif (len(int_y[0]) == 2):
            # plot_3d(x, y, int_y.numpy()[0], marker='o')
            # plot_3d(x, y, int_y.numpy()[1], marker='o')
            plot_3d(x, y, int_y[:, 0], marker='o', title='Integration 1')
            plot_3d(x, y, int_y[:, 1], marker='o', title='Integration 2')
        else:
            raise ValueError('plot_int1 can not handle len(int_y) >2 case!!')
    else:
        raise ValueError('plot_int1 can not handle >=4D case!!')

    ana_int_y = analytical_integration(var_np, int_y)  # 1D array or 2D array
    # print('int_y in plot_int1: ', len(ana_int_y), len(ana_int_y[0]), ana_int_y)
    if (len(ana_int_y) == len(var_np)):
        if (len(var_np[0]) == 1):
            plt.scatter(var_np, ana_int_y, label='analytical')
            plt.scatter(
                var_np,
                ana_int_y[:] - int_y[:, 0],
                label='analytical - integration')
        elif (len(var_np[0]) == 2):
            if (len(int_y[0]) == 1):
                plot_3d(x, y, ana_int_y, marker='+', title='Analytical')
                plot_3d(
                    x,
                    y,
                    ana_int_y[:] - int_y[:, 0],
                    marker='+',
                    title='analytical - integration')
            elif (len(int_y[0]) == 2):
                plot_3d(x, y, ana_int_y[:, 0], marker='o', title='Analytical 1')
                plot_3d(x, y, ana_int_y[:, 1], marker='o', title='Analytical 2')
                plot_3d(
                    x,
                    y,
                    ana_int_y[:, 0] - int_y[:, 0],
                    marker='o',
                    title='analytical - integration 1')
                plot_3d(
                    x,
                    y,
                    ana_int_y[:, 1] - int_y[:, 1],
                    marker='o',
                    title='analytical - integration 2')
            else:
                raise ValueError(
                    'plot_int1 can not handle len(int_y) >2 case!!')
        else:
            raise ValueError('plot_int1 can not handle >=4D case!!')
    else:
        print(
            'You can provide the analytical solution in analytical_integration() in ml_math.py'
        )
    plt.legend()
    plt.show()


def plot_3d(x, y, z, marker='o', title=''):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, marker=marker)
    ax.set_title(title)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [5, 6, 2, 3, 13, 4, 1, 2, 4, 8]
    z = [2, 3, 3, 3, 5, 7, 9, 11, 9, 10]
    plot_3d(x, y, z, marker='o')
    plt.show()

# def compute_integration(fcn, var, num):
# numerical integration or

#-----------------------test gradient-------------------------------------------
#  # def loss(model, x, y):
#    # y_ = model(x)
#    # print(x,y)
#    # return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)
#  # features, labels = next(iter(train_dataset))
#  # l = loss(model, features, labels)
#  # print("Loss test: {}".format(l))
#
#  # def grad(model, inputs, targets):
#    # with tf.GradientTape() as tape:
#      # loss_value = loss(model, inputs, targets)
#    # return loss_value, tape.gradient(loss_value, model.trainable_variables)
#
##  #By default, the resources held by a GradientTape are released as soon as GradientTape.gradient() method is called
# print('model variables: ', len(model.variables))

# x  = np.array([1.0, 2.0])
# print('value_0: ', value_0)
# tape.watch(test_dataset)
# print('test_dataset', test_dataset)
# print('test_dataset index', test_dataset.index)
# x = test_dataset.index(0)
# tape.watch(x)
# predicted_labels = model.predict(x)
# predicted_labels = model.predict(test_dataset)

# print('predicted_labels: ', predicted_labels)
# print('test_labels: ', test_labels)
# grads = tape.gradient(loss, model.trainable_variables)
#grads = tape.gradient(predicted_labels, model.input)
# exit(0)
#------------------------------------------------------------------------
# print('data type: ', type(train_dataset))
# print(train_dataset)
# # value_0 = train_dataset.values
# value_0 = train_dataset.to_numpy()
# print(type(value_0))
# print(value_0)
# def my_func(arg):
# arg = tf.convert_to_tensor(arg, dtype=tf.float32)
# return arg
# value_1 = my_func(train_dataset.values)
# print(type(value_1))
# exit(0)
