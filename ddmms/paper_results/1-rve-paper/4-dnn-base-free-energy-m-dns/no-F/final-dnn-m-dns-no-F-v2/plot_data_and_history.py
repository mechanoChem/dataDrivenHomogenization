import sys
import pandas as pd
import matplotlib.pyplot as plt
from myplot import *
import pickle

if __name__ == '__main__':
    plt_zxx(True)

    timemark = '20200111222226'
    try:
        timemark = sys.argv[1]
    except:
        pass
    history_file = 'history_' + timemark + '.pickle'
    all_data_file = 'all_data_' + timemark + '.pickle'
    print('loading data:', history_file, all_data_file)

    all_data = pickle.load(open(all_data_file, "rb"))
    history = pickle.load(open(history_file, "rb"))

    epoches = range(0, len(history['loss']))

    #----------------------plot 1---------------------------------------
    plt.clf()
    plt.semilogy(epoches, history['loss'], 'b', lw=1.0, label='Training')
    plt.semilogy(epoches, history['val_loss'], 'k', lw=1.0, label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.axis('equal')
    plt.savefig('m-dns-free-energy-learning-dnn.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    #----------------------plot 2---------------------------------------
    plt.clf()
    plt.plot(all_data['test_label'], all_data['test_nn'], 'k.')
    xmin = min(min(all_data['test_label']), min(all_data['test_nn']))
    xmax = max(max(all_data['test_label']), max(all_data['test_nn']))
    plt.plot([xmin, xmax], [xmin, xmax], 'k-', lw=1.0)

    plt.axes().set_aspect('equal', 'box')
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel('$\Psi^0_{\mathrm{mech,DNS}}$')
    plt.ylabel('$\Psi^0_{\mathrm{mech,DNN}}$')
    plt.savefig('m-dns-free-energy-test-dnn.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    # #----------------------plot 3---------------------------------------
    all_prediction_file = 'all_prediction_' + timemark + '.pickle'
    all_prediction = pickle.load(open(all_prediction_file, "rb"))

    plt.clf()
    ind0 = range(51,901)

    plt.plot(ind0, all_prediction['label_dns'][0], 'k',  linewidth=4, label='DNS')
    plt.plot(ind0, all_prediction['label_nn'][0], 'r', label='DNN')
    for i in range(1, len(all_prediction['label_nn'])):
        plt.plot(ind0, all_prediction['label_dns'][i], 'k', linewidth=4)
        plt.plot(ind0, all_prediction['label_nn'][i], 'r')

    plt.xlabel('frame number')
    plt.ylabel('$\Psi^0_{\mathrm{mech}}$')
    plt.xlim([0, 900])
    # plt.axis('equal')
    plt.legend()
    plt.savefig('m-dns-free-energy-predict-all-dnn.pdf', bbox_inches='tight', format='pdf')
    plt.show()
