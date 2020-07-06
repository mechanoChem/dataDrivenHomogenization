def split_data(datax, datay, split_ratio=['0.6', '0.25', '0.15']):
    tr_ratio = float(split_ratio[0])
    cv_ratio = float(split_ratio[1])
    tt_ratio = float(split_ratio[2])

    number_examples = datax.shape[0]
    idx = np.arange(0, number_examples)
    np.random.shuffle(idx)
    datax = [datax[i] for i in idx]  # get list of `num` random samples
    datay = [datay[i] for i in idx]  # get list of `num` random samples

    start = 0
    end_tr = int(tr_ratio * number_examples)
    end_cv = int((tr_ratio + cv_ratio) * number_examples)
    end_tt = number_examples
    tr_datax = np.array(datax[start:end_tr])
    tr_datay = np.array(datay[start:end_tr])
    cv_datax = np.array(datax[end_tr:end_cv])
    cv_datay = np.array(datay[end_tr:end_cv])
    tt_datax = np.array(datax[end_cv:end_tt])
    tt_datay = np.array(datay[end_cv:end_tt])

    return tr_datax, tr_datay, cv_datax, cv_datay, tt_datax, tt_datay


def split_data1(datax, split_ratio=['0.6', '0.25', '0.15']):
    tr_ratio = float(split_ratio[0])
    cv_ratio = float(split_ratio[1])
    tt_ratio = float(split_ratio[2])
    number_examples = datax.shape[0]
    idx = np.arange(0, number_examples)
    np.random.shuffle(idx)
    datax = [datax[i] for i in idx]  # get list of `num` random samples

    start = 0
    end_tr = int(tr_ratio * number_examples)
    end_cv = int((tr_ratio + cv_ratio) * number_examples)
    end_tt = number_examples
    tr_datax = np.array(datax[start:end_tr])
    cv_datax = np.array(datax[end_tr:end_cv])
    tt_datax = np.array(datax[end_cv:end_tt])

    return tr_datax, cv_datax, tt_datax


# def k_fold_cv(three-ways):

# def normalize_data():

# import numpy as np
# def next_batch(num, data, labels):
# '''
# Return a total of `num` random samples and labels.
# '''
# # only use it when you have large enough data samples
# # from: https://stackoverflow.com/questions/40994583/how-to-implement-tensorflows-next-batch-for-own-data

# # Questions:
# #   will it revisit all the data?
# # Answer:
# #   yes, it should.
# # So, the following implementation is not correct.
# idx = np.arange(0 , len(data))
# np.random.shuffle(idx) # This function only shuffles the array along the first axis of a multi-dimensional array. The order of sub-arrays is changed but their contents remains the same.

# idx = idx[:num]
# data_shuffle = [data[ i] for i in idx]
# labels_shuffle = [labels[ i] for i in idx]

# print("ATT: this fcn is not implemented correctly, as there is no guarantee that all the data will be visited during each epoch.!!! Use with caution!")

# return np.asarray(data_shuffle), np.asarray(labels_shuffle)

import numpy as np


class Dataset1D:
    """
  to obtain a customized next-batch option for 1D data
  """

    def __init__(self, data):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._num_examples = data.shape[0]
        pass

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe
            self._data = [self._data[i] for i in idx
                         ]  # get list of `num` random samples

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            self._data = [self._data[i] for i in idx0
                         ]  # get list of `num` random samples

            start = 0
            self._index_in_epoch = batch_size - rest_num_examples  #avoid the case where the #sample != integar times of batch_size
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


import numpy as np


class Dataset2D:
    """
  to obtain a customized next-batch option for 2D data
  """

    def __init__(self, datax, datay):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._datax = datax
        self._datay = datay
        self._num_examples = datax.shape[0]
        pass

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx)  # shuffle indexe

            self._datax = [self._datax[i] for i in idx]
            self._datay = [self._datay[i] for i in idx]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start

            datax_rest_part = self._datax[start:self._num_examples]
            datay_rest_part = self._datay[start:self._num_examples]

            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes

            self._datax = [self._datax[i] for i in idx0]
            self._datay = [self._datay[i] for i in idx0]

            start = 0

            self._index_in_epoch = batch_size - rest_num_examples  #avoid the case where the #sample != integar times of batch_size

            end = self._index_in_epoch
            datax_new_part = self._datax[start:end]
            datay_new_part = self._datay[start:end]
            return np.concatenate(
                (datax_rest_part, datax_new_part), axis=0), np.concatenate(
                    (datay_rest_part, datay_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._datax[start:end], self._datay[start:end]
