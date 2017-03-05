__author__ = 'aureliabustos'


import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np



def plot_learning_curve(title,ylim=None, train_sizes = None, logX = False, train_scores_mean = None,train_scores_std = None, test_scores_mean = None, test_scores_std = None  ):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()



    train_sizes  = np.array(train_sizes)
    if logX:
        fig, ax = plt.subplots()
        ax.set_xscale('log')


    if train_scores_mean:
        train_scores_mean = np.array(train_scores_mean)
        train_scores_std =  np.array(train_scores_std)
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    if test_scores_mean:
        test_scores_mean = np.array(test_scores_mean)
        test_scores_std =  np.array(test_scores_std)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")


    plt.legend(loc="best")
    plt.savefig(title + "_plot.png")


