import os
import pandas as pd
import numpy as np
from fastText import train_supervised
from fastText import load_model
import sys
import os


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--debug',action='store_true',help='Wait for remote debugger to attach')
a = parser.parse_args()

if a.debug:
	import ptvsd
	print("Waiting for remote debugger...")
	# Allow other computers to attach to ptvsd at this IP address and port, using the secret
	ptvsd.enable_attach("", address = ('0.0.0.0', 3000))
	# Pause the program until a remote debugger is attached
	ptvsd.wait_for_attach()
    #print("Remote debugger connected: resuming execution.")

sys.path.append("../capstone")
os.chdir("../capstone")

VALIDATION_SPLIT = 0.2
TRAIN_MODEL = True

fname_data  =  "./textData/labeledEligibilitySample"
data_train = "./textData/labeledEligibilityFastText_train.csv"
data_val = "./textData/labeledEligibilityFastText_val.csv"
classifier_fname = "./classifiers/model_fasttext_sample"
data_test =  "./textData/testClassifier.csv"

def run_classifier(set_size):
    import util  as u

    #load data balanced by class labels limited to SET_SIZE
    if set_size:
        df = u.generate_small_set(set_size, fname_data)
    else: #load whole data limited by balanced undersampling
        df = u.generate_small_set(None, None)

    # split the data into a training set and a validation set
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=VALIDATION_SPLIT, random_state=0)
    X = df.eligibility
    y = df.eligible

    scoresTrain = []
    scoresVal = []
    for train_index, test_index in sss.split(X, y):
        df_val, df_train = df.iloc[test_index, :], df.iloc[train_index, :]
        print("training sample after stratified sampling: ")
        print(df_train.describe() )
        print("validation sample after after stratified sampling: " )
        print(df_val.describe() )
        df_train.to_csv(sep='\t', path_or_buf=data_train)
        df_val.to_csv(sep='\t', path_or_buf=data_val)


        classifier = None
        if TRAIN_MODEL == False:
            print("starting to load model")
            classifier = load_model(classifier_fname +'.bin')
        else:
            print("start to train classifier model")
            #classifier = fasttext.supervised(data_train, classifier_fname, pretrained_vectors = './wordEmbeddings/vectorsFastText.vec', epoch= 100)
            #classifier = fasttext.supervised(data_train, classifier_fname, epoch= 100, silent = 0, thread=4, pretrained_vectors = './wordEmbeddings/vectorsFastText_skipgram.vec',  )
            classifier = train_supervised(data_train, epoch= 100, verbose=1, thread=32 )
            classifier.save_model(classifier_fname +'.bin')
            print("end")


        
        result = classifier.test(data_val)
        print('P@1:', result[1])
        print('R@1:', result[2])
        print('Number of examples:', result[0])

        texts = [
            'neuropsychiatric history or altered mental status',
            'pembrolizumab and corticosteroids',
            'trastuzumab and breast cancer and heart insufficiency and dyspnea',
            'trastuzumab and breast cancer',
            'trastuzumab and breast cancer and invasive cancer' ,
            'nivolumab and hiv',
            'CAR and lymphoma',
            'TCR and breast cancer' ,
            'in situ breast cancer and pemetrexed',
            'bevacizumab and patients who has had any event of thrombosis',
            'capecitabine and breast cancer and brain metastasis',
            'capecitabine and colon cancer',
            'lapatinib and breast cancer and brain metastasis',
            'pertuzumab and breast cancer and brain metastasis'

        ]

        # predict with the probability
        #labels = classifier.predict_proba(texts)
        #print(labels)
        result = classifier.test(data_test)
        print('P@1:', result[1])
        print('R@1:', result[2])
        print('Number of examples:', result[0])

        #k = 1
        # print(classifier.labels)                  # List of labels
        # print(classifier.label_prefix)            # Prefix of the label
        # print(classifier.dim)                     # Size of word vector
        # print(classifier.ws)                      # Size of context window
        # print(classifier.epoch)                   # Number of epochs
        # print(classifier.min_count)               # Minimal number of word occurences
        # print(classifier.neg)                     # Number of negative sampled
        # print(classifier.word_ngrams)             # Max length of word ngram
        # print(classifier.loss_name)               # Loss function name
        # print(classifier.bucket)                  # Number of buckets
        # print(classifier.minn)                    # Min length of char ngram
        # print(classifier.maxn)                    # Max length of char ngram
        # print(classifier.lr_update_rate)          # Rate of updates for the learning rate
        # print(classifier.t)                       # Value of sampling threshold
        # print(classifier.encoding)                # Encoding that used by classifier
        # print(classifier.test(data_val, k))       # Test the classifier
        # print(classifier.predict(texts, k))       # Predict the most likely label
        #print(classifier.predict_proba(texts, k)) # Predict the most likely label include their probability

        #Confusion matrix
        classifier =  load_model(classifier_fname +'.bin')
        df_val = pd.read_csv(data_val, sep='\t', header=0, names = ["index", "y", "x"])

        predicted = classifier.predict(list(df_val.x.values))
        predictedTrain = classifier.predict(list(df_train.eligibility.values))
        predicted =  np.array(predicted[0]).flatten()
        predictedTrain = np.array(predictedTrain[0]).flatten()

        d = {"y_true" : df_val.y, "y_pred" : predicted}
        df_confVal = pd.DataFrame(d)
        print(df_confVal.head())





        truePos =  df_confVal.loc[lambda df: (df_confVal.y_true == "__label__0") & (df_confVal.y_true ==  df_confVal.y_pred), :]
        FalseNeg =  df_confVal.loc[lambda df: (df_confVal.y_true == "__label__0") & (df_confVal.y_true !=  df_confVal.y_pred), :]
        trueNeg =  df_confVal.loc[lambda df: (df_confVal.y_true == "__label__1") & (df_confVal.y_true ==  df_confVal.y_pred), :]
        FalsePos =  df_confVal.loc[lambda df: (df_confVal.y_true == "__label__1") & (df_confVal.y_true !=  df_confVal.y_pred), :]

        confusion_table = pd.DataFrame({"True Positives": [truePos.y_true.size,FalseNeg.y_true.size], "True Negatives": [FalsePos.y_true.size, trueNeg.y_true.size]}, index=["Predicted Positives","Predicted Negatives"])
        print(confusion_table)

        #cohen's Kappa agreement
        from sklearn.metrics  import cohen_kappa_score
        kappa = cohen_kappa_score(list(df_confVal.y_true.values), list(df_confVal.y_pred.values))
        print("kappa =" + str(kappa) )



        #classification report
        from sklearn.metrics import classification_report ,f1_score
        target_names = ['Eligible', 'Not elegible']
        report = classification_report(list(df_confVal.y_true.values), list(df_confVal.y_pred.values), target_names=target_names)
        print(report)
        f1Val =  f1_score(list(df_confVal.y_true.values), list(df_confVal.y_pred.values), pos_label='__label__0', average='macro')
        scoresVal.append(f1Val)
        f1Train =  f1_score(list(df_train.eligible.values), predictedTrain, pos_label='__label__0', average='macro')
        scoresTrain.append(f1Train)


    scoresTrain = np.array(scoresTrain)
    scoresVal = np.array(scoresVal)
    print("Accuracy " + str(y.size) +": %0.2f (+/- %0.2f)" % (scoresVal.mean(), scoresVal.std() * 2))
    return scoresTrain,scoresVal





#run_classifier(None)
train_sizes = [1000, 10000, 100000, 1000000]

train_scores_mean = []
train_scores_std = []
test_scores_mean = []
test_scores_std = []
for s in train_sizes:
    scoresT, scoresV = run_classifier(s)
    train_scores_mean.append(scoresT.mean())
    train_scores_std.append(scoresT.std())
    test_scores_mean.append(scoresV.mean())
    test_scores_std.append(scoresV.std())



import matplotlib
matplotlib.use('Agg')
import plot as pl
title =  "Learning_Curves_FastText_Classifier"

pl.plot_learning_curve(title, train_sizes = train_sizes, logX=True,
                       test_scores_mean = test_scores_mean,test_scores_std = test_scores_std,
                       train_scores_mean = train_scores_mean,train_scores_std = train_scores_std)
