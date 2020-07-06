import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

import sys
import ddmms.misc.ml_misc as ml_misc
import datetime


def plot_scatter_two_fields(predicted_labels,
                            test_labels,
                            savefig=False,
                            filename='scatter2.png'):
    # print(predicted_labels)
    # print(test_labels[0:len(predicted_labels)])
    plt.clf()
    plt.scatter(test_labels[0:len(predicted_labels)], predicted_labels)
    amax = max(np.amax(test_labels), np.amax(predicted_labels))
    amin = min(np.amin(test_labels), np.amin(predicted_labels))

    plt.gca().set_title('true vs predicted values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.axis('equal')
    plt.axis('square')
    _ = plt.plot([amin, amax], [amin, amax])
    print('amax: = ', amax, plt.xlim()[1], plt.ylim()[1])
    plt.xlim([amin, amax])
    plt.ylim([amin, amax])

    if savefig:
        # plt.savefig(filename,format='eps', bbox_inches='tight')
        plt.savefig(filename, format='png')


def plot_scatter(histories, test_labels, savefig=False):
    if (savefig):
        plt.clf()
    for m_id, name, history, predicted_labels in histories:
        # print(test_labels)
        # print(predicted_labels)
        plt.scatter(test_labels, predicted_labels
                    #, label=name.title()+' TT' # hide labels
                   )
        try:
            amax = max(np.amax(test_labels.values), np.amax(predicted_labels))
            amin = min(np.amin(test_labels.values), np.amin(predicted_labels))
        except:
            amax = max(np.amax(test_labels.numpy()), np.amax(predicted_labels))
            amin = min(np.amin(test_labels.numpy()), np.amin(predicted_labels))

        # if (m_id == 0):
        # plt.plot(-1000, -1000 ,'o', color = 'k', label='Test')

    plt.gca().set_title('true vs predicted values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.axis('equal')
    plt.axis('square')
    # _ = plt.plot([-amax, amax], [-amax, amax])
    _ = plt.plot([amin, amax], [amin, amax])
    print('amax: = ', amax, plt.xlim()[1], plt.ylim()[1])
    # plt.xlim([0,plt.xlim()[1]])
    # plt.ylim([0,plt.ylim()[1]])
    plt.xlim([amin, amax])
    plt.ylim([amin, amax])
    # plot a straight diagonal line
    # plt.legend()

    if savefig:
        # plt.savefig('scatter.eps',format='eps', bbox_inches='tight')
        plt.savefig('scatter.png', format='png')


def plot_error(histories, test_labels, r_error=False, show_bin=False):
    for m_id, name, history, predicted_labels in histories:
        if (r_error):
            error = (predicted_labels - test_labels) / test_labels
        else:
            error = predicted_labels - test_labels

        # do not want to plot the bins
        y, binEdges = np.histogram(error, bins=15)
        bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
        val = plt.plot(bincenters, y, '-'
                       # , label=name.title()+' TT' # hide label
                      )
        if (m_id == 0):
            plt.plot(0, 0, '-', color='k', label='Test')

        if (show_bin):
            plt.hist(error.values, bins=15, color=val[0].get_color())

    if (r_error):
        plt.xlabel("Rel. Error of Prediction ")
        plt.gca().set_title('histogram of relative prediction error')
    else:
        plt.xlabel("Error of Prediction")
        plt.gca().set_title('histogram of prediction error')
    plt.ylabel("Count")
    plt.legend()


def plot_test_loss(histories, test_loss, config):
    markers = ['o', '+', 'x', '*', '.', 'v', '^', '<', '>']
    for m_id, name, history, predicted_labels in histories:
        label = name.title() + ' TT'
        x = m_id
        loss = test_loss[0]
        val = plt.plot(x, loss, markers[0]
                       #, label=label + '-loss' # hide label
                      )

        metrics = ml_misc.getlist_str(config['MODEL']['Metrics'])
        for i0 in range(1, len(test_loss)):
            other = test_loss[i0]
            plt.plot(
                x,
                other,
                markers[i0],
                color=val[0].get_color()
                #, label=label + '-' + metrics[i0-2] # hide label
            )

        if (m_id == 0):
            plt.plot(-1, 0, markers[0], color='k', label='loss')
            for i0 in range(1, len(test_loss)):
                plt.plot(-1, 0, markers[i0], color='k', label=metrics[i0 - 2])

    # do not want to plot the bins
    # print("test loss: ", loss, " mae: ", mae, " mse: ", mse)

    plt.gca().set_title('test loss and other quantities')
    plt.xlim([-0.5, m_id + 1])
    plt.xlabel("Model #")
    plt.ylabel("Loss")
    plt.legend()


def plot_label(histories):
    for m_id, name, history, predicted_labels in histories:
        label = name.title()
    plt.plot([-1.0], [-1.0], '-', label=label)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc=10, ncol=3)
    plt.axis('off')


def plot_default_results(config, histories, test_labels, test_loss):
    plotting_fields = ml_misc.getlist_str(config['OUTPUT']['PlotFields'])

    # make sure we have the correct total images
    num_images = len(plotting_fields)
    num_rows = int(math.sqrt(num_images))
    num_cols = int(math.sqrt(num_images)) + 1
    if (num_rows * num_cols < num_images):
        num_rows += 1

    # print (num_rows, num_cols, num_rows*num_cols, num_images)
    img_id = 0
    for i in range(0, num_images):
        key = plotting_fields[i]
        history = histories[0][2]  #
        if (key in history.history.keys()):
            # print ('key=',key)
            img_id += 1
            plt.subplot(num_rows, num_cols, img_id)
            for m_id, name, history, predicted_labels in histories:
                try:
                    history.history.keys().index['val_loss']
                    val = plt.plot(
                        history.epoch,
                        history.history['val_' + key],  #semilogy
                        '--'
                        # , label=name.title()+' VD' # hide label
                    )
                    # make sure the colors are the same
                    plt.plot(
                        history.epoch,
                        history.history[key],
                        '-',
                        color=val[0].get_color()  #semilogy
                        # , label=name.title()+' TR' # hide label
                    )
                except:
                    plt.plot(
                        history.epoch,
                        history.history[key],
                        '-'  #semilogy
                        # , label=name.title()+' TR' # hide label
                    )
                    pass
                # print(history.history[key])

                if (m_id == 0):
                    plt.plot(0, 0, '--', color='k', label='Validation')
                    plt.plot(0, 0, '-', color='k', label='Train')

            plt.gca().set_title('train/validation loss vs epochs')
            plt.xlabel('Epochs')
            plt.ylabel(key.replace('_', ' ').title())
            plt.legend()

            plt.xlim([0, max(history.epoch)])
            # plt.show()

    if 'scatter' in plotting_fields:
        img_id += 1
        plt.subplot(num_rows, num_cols, img_id)
        plot_scatter(histories, test_labels)

    if 'error' in plotting_fields:
        img_id += 1
        plt.subplot(num_rows, num_cols, img_id)
        plot_error(histories, test_labels)

    if 'r_error' in plotting_fields:
        img_id += 1
        plt.subplot(num_rows, num_cols, img_id)
        plot_error(histories, test_labels, True, False)

    if 't_loss' in plotting_fields:
        img_id += 1
        plt.subplot(num_rows, num_cols, img_id)
        plot_test_loss(histories, test_loss, config)

    if 'label' in plotting_fields:
        img_id += 1
        plt.subplot(num_rows, num_cols, img_id)
        plot_label(histories)

    if (img_id < num_images):
        print(
            "***Warning***: some fields specified in the config file is not plotted!!!"
        )


def plot_final_results(config, histories, test_labels, test_loss):
    plot_option = config['PLOT']['Option'].lower()
    if (plot_option == 'default'):
        plot_default_results(config, histories, test_labels, test_loss)
    elif (plot_option == ''):
        return
    else:
        raise ValueError("Given plot option = ", plot_option,
                         ' is not implemented!!!')


def plot_images(img_test, img_predict, encoded_imgs=[], n=10):
    plt.figure(figsize=(20, 4))
    shape0 = tf.shape(encoded_imgs[0]).numpy()
    img0 = img_test[0]
    for i in range(1, n + 1):
        # display original
        ax = plt.subplot(3, n, i)
        # plt.imshow(img_test[i].reshape(28, 28)) # np array
        plt.imshow(tf.reshape(img_test[i],
                              tf.shape(img0).numpy()[:-1]))  # tensor
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + n)
        # plt.imshow(img_predict[i].reshape(28, 28)) # np array
        plt.imshow(tf.reshape(img_predict[i],
                              tf.shape(img0).numpy()[:-1]))  # tensor
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded img
        if (len(encoded_imgs) > 0):
            ax = plt.subplot(3, n, i + n + n)
            plt.imshow(
                tf.transpose(
                    tf.reshape(encoded_imgs[i],
                               (shape0[0], shape0[1] * shape0[2]))))  # tensor
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()


def plot_encoded_images(encoded_imgs, n=10):
    plt.figure(figsize=(20, 8))
    img0 = encoded_imgs[0]
    # print(tf.shape(img0))
    shape0 = tf.shape(img0).numpy()
    for i in range(1, n + 1):
        ax = plt.subplot(1, n, i)
        # plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T)
        plt.imshow(
            tf.transpose(
                tf.reshape(encoded_imgs[i],
                           (shape0[0], shape0[1] * shape0[2]))))  # tensor
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def plot_encoded_images_features(encoded_imgs, n=10):
    plt.figure(figsize=(20, 8))
    img0 = encoded_imgs[0]
    print(tf.shape(img0))
    shape0 = tf.shape(img0).numpy()

    num_col = int(np.sqrt(shape0[2])) + 1
    num_row = num_col

    for i0 in range(0, n):
        for i in range(1, shape0[2] + 1):  # 2nd index is the feature numbers
            ax = plt.subplot(num_col, num_row, i)
            # print(i,tf.slice(encoded_imgs[0], [0,0,i], [shape0[0], shape0[1], 1]))
            plt.imshow(encoded_imgs[i0, :, :, i - 1])  # tensor
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


def inspect_cnn_features(model, config, test_dataset, savefig=False):
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
            for i in range(1,
                           shape0[2] + 1):  # 2nd index is the feature numbers
                count += 1
                ax = plt.subplot(num_col, num_row, count)
                plt.imshow(out1[0, :, :, i - 1])  # tensor
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        if savefig:
            plt.savefig(str(i0)+'.pdf', bbox_inches='tight', format='pdf')
        plt.show()


def inspect_cnn_last_layer(model, config, test_dataset, filename):
    num_images = int(config['OUTPUT']['NumImages'])
    inspect_layers = [len(model.layers) - 1]

    l0 = len(model.layers) - 1
    out1 = model.check_layer(test_dataset, l0)
    l0 = len(model.layers)
    out2 = model.check_layer(test_dataset, l0)

    # print(out1)
    # print(out2)
    # # out = tf.stack([out2, out1], axis=0)  # not working, has to be the same shape
    out = tf.concat([out2, out1], 1)
    # tf.io.write_file('test.csv', out)
    np.savetxt(filename[:-7] + '.csv', out, delimiter=',')


############ save data to file ##############
def save_list_to_file(file_path, list0):
    """ write list to file """
    f = open(file_path, "w")
    for l0 in list0:
        l1 = [str(x) for x in l0]
        one_line = ',\t'.join(l1)
        one_line += '\n'
        f.write(one_line)
    f.close()


def plot_images3D(img_test, img_predict, img_input, n=1, mark=False):
    # plt.figure(figsize=(8, 8))
    img0 = img_test[0]
    # print(type(img_test), type(img_predict), type(img_input))
    # print(tf.shape(img0))
    if (tf.__version__[0:1] == '1'):
        sess = tf.Session()
        with sess.as_default():
            input_shapes = tf.shape(img0).eval()
    elif (tf.__version__[0:1] == '2'):
        input_shapes = tf.shape(img0).numpy()

    for i in range(1, n + 1):
        if (len(input_shapes) == 4):
            the_img_test = tf.reshape(img_test[i], input_shapes[:-1])
            the_img_pre = tf.reshape(img_predict[i], input_shapes[:-1])
            the_img_input = tf.reshape(img_input[i], input_shapes[:-1])
        elif (len(input_shapes) == 3):
            the_img_test = img_test[i]
            the_img_pre = img_predict[i]
            the_img_input = img_input[i]
        if (mark):
            the_img_pre = np.ma.masked_where(the_img_test < -0.9, the_img_pre)

            tmp_img_test = np.concatenate((the_img_test, the_img_test), axis=2)
            print(np.shape(tmp_img_test))
            the_img_input = np.ma.masked_where(tmp_img_test < -0.9,
                                               the_img_input)
            the_img_input = np.ma.masked_where(the_img_input <= 0,
                                               the_img_input)

            the_img_test = np.ma.masked_where(the_img_test < -0.9, the_img_test)
        the_img_mark = 1.0e-10 * np.ones(np.shape(the_img_test))

        # display reconstruction
        ax = plt.subplot(2, 5, 1)
        c_img = plt.imshow(the_img_input[:, :, 0])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('input: ux')

        # display reconstruction
        ax = plt.subplot(2, 5, 2)
        c_img = plt.imshow(the_img_input[:, :, 2])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('input: Tx')


        # display original
        ax = plt.subplot(2, 5, 3)
        c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('label: ux')

        # display reconstruction
        ax = plt.subplot(2, 5, 4)
        c_img = plt.imshow(the_img_pre[:, :, 0])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('prediction: ux')

        # display error
        ax = plt.subplot(2, 5, 5)
        c_img = plt.imshow((the_img_test[:, :, 0] - the_img_pre[:, :, 0]) / (the_img_test[:, :, 0] + the_img_mark[:, :, 0]))  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('error: uy')

        # display reconstruction
        ax = plt.subplot(2, 5, 6)
        c_img = plt.imshow(the_img_input[:, :, 1])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('input: uy')

        # display reconstruction
        ax = plt.subplot(2, 5, 7)
        c_img = plt.imshow(the_img_input[:, :, 3])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('input: Ty')

        # display original
        ax = plt.subplot(2, 5, 8)
        c_img = plt.imshow(the_img_test[:, :, 1])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('label: uy')

        # display reconstruction
        ax = plt.subplot(2, 5, 9)
        c_img = plt.imshow(the_img_pre[:, :, 1])  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('prediction: uy')

        # display error
        ax = plt.subplot(2, 5, 10)
        c_img = plt.imshow((the_img_test[:, :, 1] - the_img_pre[:, :, 1]) / (the_img_test[:, :, 1] + the_img_mark[:, :, 1]))  # tensor
        # plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('error: uy')


    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig("prediction" + now_str + ".png", format='png')
    plt.show()


def plot_strain_images_2D3D(img_test, dim=2):
    img0 = img_test[0]
    print('img0 shape: ', tf.shape(img0).numpy(), 'total_shape:',
          tf.shape(img_test))

    for i in range(0, 1):
        if (dim == 2):
            total_image = 4
            the_img_test = 16.0 * 2.0 * img_test[
                i]  #tf.reshape(img_test[i], tf.shape(img0).numpy()[:-1])
        elif (dim == 3):
            the_img_test = img_test[i]
            total_image = 9

        ax = plt.subplot(2, 4, 1)
        c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('eps_xx')

        # display reconstruction
        ax = plt.subplot(2, 4, 2)
        c_img = plt.imshow(the_img_test[:, :, 1])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('eps_xy')

        # display reconstruction
        ax = plt.subplot(2, 4, 3)
        c_img = plt.imshow(the_img_test[:, :, 2])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('eps_yx')

        # display reconstruction
        ax = plt.subplot(2, 4, 4)
        c_img = plt.imshow(the_img_test[:, :, 3])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('eps_yy')

        # # display original
        # ax = plt.subplot(2, 4, 5)
        # c_img = plt.imshow(the_img_test[:,:,1]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('label: uy')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 6)
        # c_img = plt.imshow(the_img_pre[:,:,1]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('prediction: uy')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 7)
        # c_img = plt.imshow(the_img_input[:,:,1]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('input: uy')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 8)
        # c_img = plt.imshow(the_img_input[:,:,3]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('input: Ty')

    plt.show()


def plot_stress_images_2D3D(img_test, dim=2):
    img0 = img_test[0]
    print('img0 shape: ', tf.shape(img0).numpy(), 'total_shape:',
          tf.shape(img_test))

    for i in range(0, 1):
        if (dim == 2):
            total_image = 4
            the_img_test = img_test[
                i]  #tf.reshape(img_test[i], tf.shape(img0).numpy()[:-1])
        elif (dim == 3):
            the_img_test = img_test[i]
            total_image = 9

        ax = plt.subplot(2, 4, 1)
        c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('sig_xx')

        # display reconstruction
        ax = plt.subplot(2, 4, 2)
        c_img = plt.imshow(the_img_test[:, :, 1])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('sig_xy')

        # display reconstruction
        ax = plt.subplot(2, 4, 3)
        c_img = plt.imshow(the_img_test[:, :, 2])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('sig_yx')

        # display reconstruction
        ax = plt.subplot(2, 4, 4)
        c_img = plt.imshow(the_img_test[:, :, 3])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('sig_yy')
    plt.show()


def plot_residual_images_2D3D(img_test, dim=2):
    img0 = img_test[0]

    print('img0 shape: ', tf.shape(img0).numpy(), 'total_shape:',
          tf.shape(img_test))

    for i in range(0, 1):
        if (dim == 2):
            total_image = 4
            the_img_test = img_test[
                i]  #tf.reshape(img_test[i], tf.shape(img0).numpy()[:-1])
        elif (dim == 3):
            the_img_test = img_test[i]
            total_image = 9

        ax = plt.subplot(2, 4, 1)
        c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('r_x')

        # display reconstruction
        ax = plt.subplot(2, 4, 2)
        c_img = plt.imshow(the_img_test[:, :, 1])  # tensor
        plt.gray()
        plt.colorbar(c_img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('r_y')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 3)
        # c_img = plt.imshow(the_img_test[:,:,2]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('sig_yx')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 4)
        # c_img = plt.imshow(the_img_test[:,:,3]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('sig_yy')

        # # display original
        # ax = plt.subplot(2, 4, 5)
        # c_img = plt.imshow(the_img_test[:,:,1]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('label: uy')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 6)
        # c_img = plt.imshow(the_img_pre[:,:,1]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('prediction: uy')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 7)
        # c_img = plt.imshow(the_img_input[:,:,1]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('input: uy')

        # # display reconstruction
        # ax = plt.subplot(2, 4, 8)
        # c_img = plt.imshow(the_img_input[:,:,3]) # tensor
        # plt.gray()
        # plt.colorbar(c_img)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # plt.title('input: Ty')

    plt.show()




def plot_images3D_c_with_error(img_test, img_predict, img_input, mask=True):
    img0 = img_test[0]

    # get the input shape
    if (tf.__version__[0:1] == '1'):
        sess = tf.Session()
        with sess.as_default():
            input_shapes = tf.shape(img0).eval()
    elif (tf.__version__[0:1] == '2'):
        input_shapes = tf.shape(img0).numpy()

    i = 1
    print('len of inputs: ', len(input_shapes))
    if (len(input_shapes) == 4):
        the_img_test = tf.reshape(img_test[i], input_shapes[:-1])
        the_img_pre = tf.reshape(img_predict[i], input_shapes[:-1])
        the_img_input = tf.reshape(img_input[i], input_shapes[:-1])
    elif (len(input_shapes) == 3):
        the_img_test = img_test[i]
        the_img_pre = img_predict[i]
        the_img_input = img_input[i]

    if (mask):
        the_img_pre = np.ma.masked_where(the_img_test < -0.9, the_img_pre)

        tmp_img_test = np.concatenate((the_img_test, the_img_test), axis=2)
        print(np.shape(tmp_img_test))
        the_img_input = np.ma.masked_where(tmp_img_test < -0.9,
                                           the_img_input)
        the_img_input = np.ma.masked_where(the_img_input <= 0,
                                           the_img_input)

        the_img_test = np.ma.masked_where(the_img_test < -0.9, the_img_test)

    the_img_mark = 1.0e-10 * np.ones(np.shape(the_img_test))

    # display reconstruction
    ax = plt.subplot(2, 3, 1)
    c_img = plt.imshow(the_img_input[:, :, 0])  # tensor
    # plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: uc')

    # display reconstruction
    ax = plt.subplot(2, 3, 4)
    c_img = plt.imshow(the_img_input[:, :, 1])  # tensor
    # plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('input: hc')

    # display original
    ax = plt.subplot(2, 3, 2)
    c_img = plt.imshow(the_img_test[:, :, 0])  # tensor
    # plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('label: uc')

    # display reconstruction
    ax = plt.subplot(2, 3, 5)
    c_img = plt.imshow(the_img_pre[:, :, 0])  # tensor
    # plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('prediction: uc')

    # display error
    ax = plt.subplot(2, 3, 6)
    c_img = plt.imshow((the_img_test[:, :, 0] - the_img_pre[:, :, 0]) / (the_img_test[:, :, 0] + the_img_mark[:, :, 0]))  # tensor
    # plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title('error: uc')


    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    plt.savefig("prediction" + now_str + ".png", format='png')
    plt.show()


def plot_one_2D_image(img, title='img'):
    # display reconstruction
    ax = plt.subplot(1, 1, 1)
    c_img = plt.imshow(img)  
    # plt.gray()
    plt.colorbar(c_img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title(title)
    plt.show()

