def get_package_version(tf_version):
    """ get the major and minor version of tensor flow """
    versions = tf_version.split('.')[0:2]
    versions = [int(x) for x in versions]
    # print(versions)
    return versions


def getlist_str(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = [(chunk.strip(chars)) for chunk in option.split(sep)]
    list0 = [x for x in list0 if x]
    return list0


def getlist_int(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = option.split(sep)
    list0 = [x for x in list0 if x]
    if (len(list0)) > 0:
        return [int(chunk.strip(chars)) for chunk in list0]
    else:
        return []


def getlist_float(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = option.split(sep)
    list0 = [x for x in list0 if x]
    if (len(list0)) > 0:
        return [float(chunk.strip(chars)) for chunk in list0]
    else:
        return []


def get_now():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def exe_cmd(cmd):
    """ execute shell cmd """
    import os
    output_info = os.popen(cmd).read()


def merge_two_pandas(p1, p2, opt='LR'):
    """ merge two pandas, LR or TB """
    import pandas as pd

    if opt == 'TB':
        frames = [p1, p2]
        result = pd.concat(frames)
        return result
    elif opt == 'LR':
        result = pd.concat([p1, p2], axis=1, sort=False)
        return result
    else:
        raise ValueError('Option: ', opt,
                         'is not known for merge_two_pandas() function!')


def save_label_comparison(label_true, label_predicted):
    """ save the true label and predicted label to a csv file """
    # print("in save_label_comparison")
    # print(type(label_true), type(label_predicted))
    # print(len(label_true), len(label_predicted))
    import numpy as np
    filename = 'label.csv'
    file1 = open(filename, 'w')
    if (len(label_true.keys()) > 1):
        raise ValueError('save label currently only support one output!')
    labelname = label_true.keys()[0]

    new_label_true = label_true.to_numpy()
    new_label_true = np.squeeze(new_label_true)
    label_predicted = np.squeeze(label_predicted)

    # print(label_predicted)
    # print(new_label_true)
    file1.writelines('true_' + labelname + ',predict_' + labelname + '\n')
    for i0 in range(0, len(label_true)):
        line0 = str(new_label_true[i0]) + ',' + str(label_predicted[i0]) + '\n'
        file1.writelines(line0)


def save_data_comparison(config,
                         data_true,
                         data_predicted,
                         train_stats,
                         filename='tmp_data_compare.csv'):
    """ save the true value and predicted value to a csv file """
    # print("in save_label_comparison")
    # print(type(data_true), type(data_predicted))
    # print(len(data_true), len(data_predicted))
    import numpy as np
    file1 = open(filename, 'w')

    label_scale = float(config['TEST']['LabelScale'])

    label_flag = False
    # get the keys, if not, read from config
    try:
        # label data
        keys = data_true.keys()
        label_flag = True
    except:
        # derivative data
        keys = getlist_str(config['TEST']['DerivativeFields'])
        label_flag = False
        pass

    vtk_flag = False
    if config['TEST']['DataFile'].find('*.vtk') >= 0:
        label_flag = True
        vtk_flag = True
        keys = ['label']

    # scale everything back to the none scaled state
    if (label_scale):
        data_true = data_true / label_scale
        data_predicted = data_predicted / label_scale
    else:
        std = train_stats['std'].to_numpy()
        data_predicted = data_predicted / label_scale / std

    # header for csv file
    true_keys = ['t_' + s for s in keys]
    predicted_keys = ['p_' + s for s in keys]

    all_keys = true_keys + predicted_keys

    # try to convert everything to numpy array
    try:
        data_true_new = data_true.to_numpy()
    except:
        try:
            data_true_new = data_true.numpy()
        except:
            data_true_new = data_true
            pass
        pass

    # try to convert everything to numpy array
    try:
        data_predicted_new = data_predicted.to_numpy()
    except:
        try:
            data_predicted_new = data_predicted.numpy()
        except:
            data_predicted_new = data_predicted
            pass
        pass

    #print(all_keys)
    #print(data_true_new)

    # write header
    if (not vtk_flag):
        file1.writelines(','.join(all_keys) + '\n')
        for i0 in range(0, len(data_true)):
            one_data = []
            for j0 in range(0, len(keys)):
                one_key = keys[j0]
                #print(i0, one_key)
                #print(data_true_new[i0][j0])
                one_data.append(data_true_new[i0][j0])
            for j0 in range(0, len(keys)):
                one_data.append(data_predicted_new[i0][j0])
            str_one_data = [str(s) for s in one_data]
            line0 = ','.join(str_one_data) + '\n'
            #print (line0)
            # write each line of data
            file1.writelines(line0)
    else:
        # print ('data_true_new: ', data_true_new)
        # print ('data_predicted_new: ', data_predicted_new)
        file1.writelines(','.join(all_keys) + '\n')
        for i0 in range(0, len(data_true)):
            one_data = []
            for j0 in range(0, len(keys)):
                one_key = keys[j0]
                one_data.append(
                    data_true_new[i0]
                )  # true value is a list of scalar, thus no second index as not (vtk_flag) part
            for j0 in range(0, len(keys)):
                one_data.append(data_predicted_new[i0][j0])
            str_one_data = [str(s) for s in one_data]
            line0 = ','.join(str_one_data) + '\n'
            #print (line0)
            # write each line of data
            file1.writelines(line0)
