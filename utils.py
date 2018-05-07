import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt


def load_data():
    np.random.seed(1990)
    print("Loading MNIST data .....")

    # Load the MNIST dataset
    with gzip.open('Data/mnist.pkl.gz', 'r') as f:
        # u = pickle._Unpickler(f)
        # u.encoding = 'latin1'
        # train_set, valid_set, test_set = u.load()
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        train_set = [train_set[0].tolist(), [[1 if j == train_set[1][i] else 0 for j in range(10)] for i in np.arange(len(train_set[0]))]]
        valid_set = [valid_set[0].tolist(), [[1 if j == valid_set[1][i] else 0 for j in range(10)] for i in np.arange(len(valid_set[0]))]]
        test_set = [test_set[0].tolist(), [[1 if j == test_set[1][i] else 0 for j in range(10)] for i in np.arange(len(test_set[0]))]]
    print("Done.")
    return train_set, valid_set, test_set

   
def plot_curve(t,s,metric):
    plt.plot(t, s)
    plt.ylabel(metric) # or ERROR
    plt.xlabel('Epoch')
    plt.title('Learning Curve_'+str(metric))
    #curve_name=str(metric)+"LC.png"
    #plt.savefig(Figures/curve_name)
    plt.show()
    
def plot_train_val(t, st, sv, hn, lr):
    '''
    Print the plots for the accuracies 
    '''
    # set the figure size
    plt.figure(figsize=(10, 7))
    # plot values
    plt.plot(t, st, 'b', label='Accuracy on training set')
    plt.plot(t, sv, 'g', label='Accuracy on validation set')
    # labels
    plt.ylabel('Accuracy', size=16) # or ERROR
    plt.xlabel('Epoch', size=16)
    # title
    plt.title('Learning Curve: HN={}, LR={}'.format(hn, lr), size=20, y=1.03)
    # legend
    plt.legend(fontsize=15, loc=4)
    # print the figure
    plt.show()

