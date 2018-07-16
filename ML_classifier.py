import numpy as np
np.random.seed(1337)
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import sys
import os
from fastText import train_unsupervised
from fastText import load_model
import multiprocessing



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug',action='store_true',help='Wait for remote debugger to attach')
parser.add_argument('--SVM',action='store_true',help='Applying SVM classifier instead of KNN (default)')
a = parser.parse_args()

if a.debug:
	import ptvsd
	print("Waiting for remote debugger...")
	# Allow other computers to attach to ptvsd at this IP address and port, using the secret
	ptvsd.enable_attach("", address = ('0.0.0.0', 3000))
	# Pause the program until a remote debugger is attached
	ptvsd.wait_for_attach()
    #print("Remote debugger connected: resuming execution.")

SVM = a.SVM
n_jobs = multiprocessing.cpu_count()
BASE_DIR = '.'
sys.path.append("../capstone")
os.chdir("../capstone")
VE_DIR = BASE_DIR + '/wordEmbeddings/'
VE_FNAME = '_vectorsFastText'
TEXT_DATA_DIR = BASE_DIR + '/textData/'
TEXT_DATA_FNAME = 'labeledEligibility.csv'
SAMPLE_TEXT_DATA_FNAME = 'labeledEligibilitySample'
RESULT_DIR = BASE_DIR + '/classifiers/'
RESULT_FNAME = 'KNN'
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 1000
import util  as u

#load pre-trained wordembeddings
# embeddings_index = {}
# f = open(os.path.join(VE_DIR, VE_FNAME),  encoding='utf8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
#
# print('Found %s word vectors.' % len(embeddings_index))

data =  "./textData/words_data.csv"
skipgram_fname = "./wordEmbeddings/_vectorsFastText"

#1 Stem: Save a we skipgram model (with same parameters used in the we used for the CNN) trained unsupervised on text
# Skipgram model
#model = train_unsupervised(
#    input=data,
#    model='skipgram',
#)
#model.save_model(skipgram_fname)
#model = fasttext.skipgram(data, skipgram_fname, ws = 5, minn = 3 ,maxn = 6, silent = 0, epoch = 10, bucket = 2000000)



#2 Step: Generate sentence vectors: KNN does not support input with dimensionality > 2 and flattening sequences of we
# per sentence would yield a 1D array of size 1000 x 100 per sentence which is not feasible for 6 M sentences. For this
# reason the average of word embeddings per sentence is used as sentence vectors.
# /home/auri/fastText print-sentence-vectors _vectorsFastText < ../textData/eligibility.csv
# paste ../textData/labelsOnly.csv sentenceEmbeddingsFastText.csv > labeledSentenceEmbeddingsFastText.csv

#3 Step: Load labels and sentence vectors and shuffle them.
VALIDATION_SPLIT = 0.2
TRAIN_MODEL = True
fname_data  =  "./textData/labeledEligibilityEmbeddingSample"
data_train = "./textData/labeledEligibilityEmbedding_train.csv"
data_val = "./textData/labeledEligibilityEmbedding_val.csv"
data_test =  "./textData/testClassifier.csv"
classifier_fname = "./classifiers/model_KNN_sample"
f = load_model(os.path.join(VE_DIR, VE_FNAME))  #fasttext skipgram trained model

def dataframe_to_2Darray(serie):
    _eligibility = serie.apply(lambda y: np.array([float(s) for s in y.strip().split(' ')])).values
    ta, tb = _eligibility.shape[0], _eligibility[0].shape[0]
    array_eligibility = np.empty((ta,tb))
    for a in range(ta):
        array_eligibility[a,:] = _eligibility[a]
    return array_eligibility

def hyper_parameter_search(set_size):
    from sklearn.model_selection import GridSearchCV
    #load data balanced by class labels limited to SET_SIZE
    fname = f'{fname_data}{set_size}.csv'
    path = os.path.join(os.getcwd(), fname)
    df = pd.read_csv(path, sep='\t', header=None, names = ["eligible", "eligibility_embeddings"])
    # split the data into a training set and a validation set
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SPLIT, random_state=0)
    X = df.eligibility_embeddings
    y = df.eligible
    for train_index, test_index in sss.split(X, y):
        df_val, df_train = df.iloc[test_index, :], df.iloc[train_index, :]
        train_eligibility = dataframe_to_2Darray(df_train.eligibility_embeddings)
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 
              'kernel':['linear’, ‘poly’, ‘rbf']}
        param_grid = {'kernel': {'linear': {'C': [0, 2]},
                                           'rbf': {'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 'C': [0, 10]},
                                           'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                           }}
        param_grid = [dict(kernel=['rbf', ], C=[1, 10], gamma=[0.1, 1]),dict(kernel=['poly', ], degree=[1, 2])]
        classifier = GridSearchCV(svm.SVC( class_weight='balanced'), param_grid)
        #classifier = svm.LinearSVC(random_state=0)
        classifier.fit(train_eligibility, df_train.eligible.values)  
        print("Best estimator found by grid search:")
        print(classifier.best_estimator_)
        #classification report
        from sklearn.metrics import classification_report ,f1_score
        target_names = ['Eligible', 'Not elegible']
        train_pred = classifier.predict(train_eligibility)
        val_eligibility = dataframe_to_2Darray(df_val.eligibility_embeddings)
        val_pred = classifier.predict(val_eligibility)
        report = classification_report(df_val.eligible.values, val_pred, target_names=target_names)
        print(report)
        f1Val =  f1_score(df_val.eligible.values, val_pred, pos_label=0, average='binary')
        print(f1Val)
        f1Train =  f1_score(df_train.eligible.values, train_pred, pos_label=0, average='binary')
        print(f1Train)  

#hyper_parameter_search(1000)

def run_classifier(set_size, generate_set = False):
    #load data balanced by class labels limited to SET_SIZE
    if set_size:
        fname = f'{fname_data}{set_size}.csv'
        if not generate_set:
            path = os.path.join(os.getcwd(), fname)
            df = pd.read_csv(path, sep='\t', header=None, names = ["eligible", "eligibility_embeddings"])
        else:
            df = u.generate_small_set_labeled_sentence_embeddings(set_size, fname_data)
    else: #load whole data limited by balanced undersampling
        fname = fname_data + '.csv'
        if not generate_set:
            path = os.path.join(os.getcwd(), fname)
            df = pd.read_csv(path, sep='\t', header=None, names = ["eligible", "eligibility_embeddings"])
        else:
            df = u.generate_small_set_labeled_sentence_embeddings(None, None)

    # split the data into a training set and a validation set
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=VALIDATION_SPLIT, random_state=0)
    X = df.eligibility_embeddings
    y = df.eligible

    scoresTrain = []
    scoresVal = []
    for train_index, test_index in sss.split(X, y):
        df_val, df_train = df.iloc[test_index, :], df.iloc[train_index, :]
        print("training sample after stratified sampling: ")
        print(df_train.describe() )
        print("validation sample after after stratified sampling: " )
        print(df_val.describe() )
        #df_train.to_csv(sep='\t', path_or_buf=data_train)
        #df_val.to_csv(sep='\t', path_or_buf=data_val)


        classifier = None
        if TRAIN_MODEL == False:
            print("starting to load model")
            classifier = None #TODO using pickle
        else:
            print("start to train classifier model")
            train_eligibility = dataframe_to_2Darray(df_train.eligibility_embeddings)
            if not SVM:
                classifier = KNeighborsClassifier(n_neighbors=3, n_jobs=32)
            else: 
                #classifier = svm.LinearSVC(random_state=0)
                classifier = svm.SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
            classifier.fit(train_eligibility, df_train.eligible.values)
            print("end")


        train_pred = classifier.predict(train_eligibility)
        val_eligibility = dataframe_to_2Darray(df_val.eligibility_embeddings)
        val_pred = classifier.predict(val_eligibility)
        from sklearn.metrics import average_precision_score
        average_precision = average_precision_score(df_val.eligible.values, val_pred)
        print('Average precision-recall validation score: {0:0.2f}'.format(average_precision))
        #print('P@1:', result.precision)
        #print('R@1:', result.recall)
        #print('Number of examples:', result.nexamples)

        #test sample
        print('Loading text test dataset')
        df_test = pd.read_csv(data_test, sep='\t', header=None, names = ["eligible", "eligibility"])
        print(df_test.describe())

        # predict with the probability
        words, frequency = f.get_words(include_freq=True)
        test_embeddings = [f.get_sentence_vector(t) for t in df_test.eligibility]
        test_pred = classifier.predict(test_embeddings)
        print(test_pred)
        df_test.eligible_digit = df_test.eligible.apply(lambda x: float(x.replace("__label__","")))
        average_precision = average_precision_score(df_test.eligible_digit, test_pred)
        print('Average precision-recall test score: {0:0.2f}'.format(average_precision))

        #print(result.precision) # Precision at one
        #print(result.recall)    # Recall at one
        #print(result.nexamples) # Number of test examples

        #k = 1
        # print(classifier.test(data_val, k))       # Test the classifier
        # print(classifier.predict(texts, k))       # Predict the most likely label
        #print(classifier.predict_proba(texts, k)) # Predict the most likely label include their probability

        #Confusion matrix
        #df_val = pd.read_csv(data_val, sep='\t', header=0, names = ["index", "y", "x"])


        #val_pred =  pd.Series(np.array(classifier.predict(df_val.eligibility)).flatten())
        #predictedTrain = pd.Series(np.array(classifier.predict(df_train.eligibility)).flatten())

        d = {"y_true" : df_val.eligible, "y_pred" : val_pred}
        df_confVal = pd.DataFrame(d)


        truePos =  df_confVal.loc[lambda df: (df.y_true == 0) & (df.y_true ==  df.y_pred), :]
        FalseNeg =  df_confVal.loc[lambda df: (df.y_true == 0) & (df.y_true !=  df.y_pred), :]
        trueNeg =  df_confVal.loc[lambda df: (df.y_true == 1) & (df.y_true ==  df.y_pred), :]
        FalsePos =  df_confVal.loc[lambda df: (df.y_true == 1) & (df.y_true !=  df.y_pred), :]

        confusion_table = pd.DataFrame({"True Positives": [truePos.y_true.size,FalseNeg.y_true.size], "True Negatives": [FalsePos.y_true.size, trueNeg.y_true.size]}, index=["Predicted Positives","Predicted Negatives"])
        print(confusion_table)

        #cohen's Kappa agreement
        from sklearn.metrics  import cohen_kappa_score
        kappa = cohen_kappa_score(df_confVal.y_true, df_confVal.y_pred)
        print("kappa =" + str(kappa) )



        #classification report
        from sklearn.metrics import classification_report ,f1_score
        target_names = ['Eligible', 'Not elegible']
        report = classification_report(df_confVal.y_true, df_confVal.y_pred, target_names=target_names)
        print(report)
        f1Val =  f1_score(df_val.eligible.values, val_pred, pos_label=0, average='binary')
        print(f1Val) 
        scoresVal.append(f1Val)
        f1Train =  f1_score(df_train.eligible.values, train_pred, pos_label=0, average='binary')
        print(f1Train) 
        scoresTrain.append(f1Train)


    scoresTrain = np.array(scoresTrain)
    scoresVal = np.array(scoresVal)
    print("Accuracy Train" + str(y.size) +": %0.2f (+/- %0.2f)" % (scoresTrain.mean(), scoresTrain.std() * 2))
    print("Accuracy Validation" + str(y.size) +": %0.2f (+/- %0.2f)" % (scoresVal.mean(), scoresVal.std() * 2))
    return scoresTrain,scoresVal

print("training fullset")
#scoresT, scoresV = run_classifier(None,True)
print("finished training fullset")
#run_classifier(1000, False)
train_sizes = [1000, 10000, 100000, 1000000]


train_scores_mean = []
train_scores_std = []
test_scores_mean = []
test_scores_std = []
for s in train_sizes:
    scoresT, scoresV = run_classifier(s, False)
    train_scores_mean.append(scoresT.mean())
    train_scores_std.append(scoresT.std())
    test_scores_mean.append(scoresV.mean())
    test_scores_std.append(scoresV.std())

import matplotlib
matplotlib.use('Agg')
import plot as pl

title = "Learning_Curves_KNN_Classifier"
if SVM:
    title = "Learning_Curves_SVM_Classifier" 
pl.plot_learning_curve(title, train_sizes = train_sizes, logX=True,
                       test_scores_mean = test_scores_mean,test_scores_std = test_scores_std,
                       train_scores_mean = train_scores_mean,train_scores_std = train_scores_std)