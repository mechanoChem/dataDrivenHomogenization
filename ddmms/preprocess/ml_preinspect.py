
# # Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import ddmms.misc.ml_misc as ml_misc

# import ddmms.math.ml_math as ml_math

# import ddmms.main as main


def plot_scatter(train_dataset, train_labels, config):
    dataset = ml_misc.merge_two_pandas(train_labels, train_dataset)
    print('... in dataset ...')
    # print('dataset keys:', dataset.keys())
    # print('train labels:', train_labels.keys(), ' always come first!')

    all_fields = ml_misc.getlist_str(config['TEST']['AllFields'])
    NumberOfInspect = int(config['TEST']['NumberOfInspect'])

    show_key0_vs_key0 = True
    # show the key itself
    if show_key0_vs_key0:
        num_rows = min(NumberOfInspect, len(dataset.keys()))
        start_ind = 0
        end_ind = num_rows
    else:
        num_rows = min(NumberOfInspect - 1, len(dataset.keys()) - 1)
        start_ind = 1
        end_ind = num_rows + 1
    print('Visualizing keys:', all_fields[0:num_rows])

    num_cols = num_rows
    img_id = 0
    for i0 in range(start_ind, end_ind):
        key0 = all_fields[i0]
        print('Key: = ', key0, 'Min: ', np.min(dataset[key0].values), 'Max: ',
              np.max(dataset[key0].values))
        # print (i0, key0)
        for i1 in range(0, num_cols):
            key1 = all_fields[i1]
            # print (i1, key1)
            img_id += 1
            plt.subplot(num_rows, num_cols, img_id)
            if (i1 <= i0):
                # if(i1 < i0): # for not show_key0_vs_key0
                plt.scatter(
                    dataset[key0],
                    dataset[key1],
                    marker='.',
                    label=key0 + '-' + key1)
                # plt.xlabel(key0)
                # plt.ylabel(key1)
                plt.xlim([
                    np.min(dataset[key0].values),
                    np.max(dataset[key0].values)
                ])
                plt.ylim([
                    np.min(dataset[key1].values),
                    np.max(dataset[key1].values)
                ])
                # plt.axis('equal')
                # plt.axis('square')
                plt.legend()
            else:
                plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5, forward=True)
    # plt.savefig('data.eps',format='eps', bbox_inches='tight')
    plt.savefig('data.png', format='png', bbox_inches='tight')
    plt.show()


