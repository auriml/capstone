from __future__ import print_function
import os
import numpy as np
np.random.seed(1337)
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model

import sys

BASE_DIR = '.'
VE_DIR = BASE_DIR + '/wordEmbeddings/'
VE_FNAME = 'vectorsFastText.vec'
TEXT_DATA_DIR = BASE_DIR + '/textData/'
TEXT_DATA_FNAME = 'labeledEligibilityFastText.csv'
#TEXT_DATA_FNAME = 'labeledEligibilitySample'
RESULT_DIR = BASE_DIR + '/classifiers/'
RESULT_FNAME = 'model_custom_CNN'
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


#classifier = load_model(filepath= RESULT_DIR + RESULT_FNAME +'2100')

def sequences_generator(seq, lab, max_sequence_length):
    my_list_len = len(lab)
    print("Generator length:" , my_list_len )
    batch_size = 10
    num_batches = int(my_list_len/batch_size)
    from sklearn.utils import shuffle
    while True:
        seq, lab = shuffle(seq,lab)
        for i in range(num_batches):
            sup = (i+1)*batch_size
            if sup > my_list_len:
                sup =   my_list_len
            yield pad_sequences(seq[i*batch_size:sup], max_sequence_length), lab[i*batch_size:sup]


def run_classifier(size = None):
    # load text samples and their labels
    print('Loading text dataset')
    path = ""
    if size: #load existing size file
        path = os.path.join(TEXT_DATA_DIR, TEXT_DATA_FNAME + str(size) + '.csv')
    else: #load full sample
        path = os.path.join(TEXT_DATA_DIR, TEXT_DATA_FNAME )
    df = pd.read_csv(path, sep='\t', header=None, names = ["eligible", "eligibility"])
    print(df.describe())

    #balance labels
    # Apply the random under-sampling
    import util  as u
    rus = u.balanced_subsample(df['eligible'])
    df = df.iloc[rus]

    print("sample after under-sampling: " )
    print(df.describe())


    # dictionary mapping label name to numeric id
    labels_values = df['eligible'].unique()
    labels_index = {k: v for v, k in enumerate(labels_values)}
    # list of text samples
    texts = df['eligibility']
    print(texts[0:3])
    # list of label ids
    labels = [labels_index[x] for x in df['eligible']]
    print(labels[0:3])



    print('Found %s texts.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))




    #MAX_SEQUENCE_LENGTH = np.max(np.array([len(j) for j in sequences]))
    MAX_SEQUENCE_LENGTH = 1000

    # split the data into a training set and a validation set
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=VALIDATION_SPLIT, random_state=0)
    X = df['eligibility']
    y = df['eligible']
    i = 0
    scoresTrain = []
    scoresVal = []
    for train_index, test_index in sss.split(X, y):
        cat_labels = to_categorical(np.asarray(labels))
        series_sequences = pd.Series(sequences)
        train = sequences_generator(series_sequences[train_index],cat_labels[train_index],MAX_SEQUENCE_LENGTH)
        val = sequences_generator(series_sequences[test_index],cat_labels[test_index],MAX_SEQUENCE_LENGTH)

        i +=1
        # split the data into a training set and a validation set
        # indices = np.arange(data.shape[0])
        # np.random.shuffle(indices)
        # data = data[indices]
        # labels = labels[indices]
        # nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
        #
        # x_train = data[:-nb_validation_samples]
        # y_train = labels[:-nb_validation_samples]
        # x_val = data[-nb_validation_samples:]
        # y_val = labels[-nb_validation_samples:]

        print('Preparing embedding matrix.')

        # build index mapping words in the embeddings set
        # to their embedding vector

        print('Indexing word vectors.')

        embeddings_index = {}
        f = open(os.path.join(VE_DIR, VE_FNAME))
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))


        # prepare embedding matrix
        nb_words = min(MAX_NB_WORDS, len(word_index))
        embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i > MAX_NB_WORDS:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(nb_words + 1,
                                    EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH
                                    )




        # train a 1D convnet with global maxpooling
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(35)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(len(labels_index), activation='softmax')(x)

        model = Model(sequence_input, preds)
        model.layers[1].trainable=False
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc', 'matthews_correlation', 'precision', 'recall', 'fmeasure'])



        print('Training model.')
        #model.fit(x_train, y_train, validation_data=(x_val, y_val),
        #          nb_epoch=2, batch_size=128)
        print("Train index lenth:", len(train_index))
        print("Test index lenth:", len(test_index))
        model.fit_generator(train, validation_data=val, nb_val_samples= len(test_index),
                  nb_epoch=10, samples_per_epoch=len(train_index))
        #for x_train, y_train in train:
            #y_train = y_train.reshape((-1, 1))
        #    model.fit(x_train, y_train, nb_epoch=2, batch_size=128)

        model.save(os.path.join(RESULT_DIR, RESULT_FNAME + str(size)))
        results = model.evaluate_generator(val, val_samples= len(test_index))
        dicVal = dict(zip(model.metrics_names, results))
        results = model.evaluate_generator(train, val_samples= len(train_index))
        dicTrain = dict(zip(model.metrics_names, results))
        print(dicVal)
        print(dicTrain)
        scoresVal.append(dicVal['fmeasure'])
        scoresTrain.append(dicTrain['fmeasure'])



    print("Bye")
    return np.array(scoresTrain),np.array(scoresVal)




scoresT, scoresV = run_classifier()
#train_sizes = [1000, 10000, 100000, 1000000]
train_sizes = []
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

import plot as pl
title =  "Learning Curves CNN Classifier Model"

pl.plot_learning_curve(title, train_sizes = train_sizes, logX=True,
                       test_scores_mean = test_scores_mean,test_scores_std = test_scores_std,
                       train_scores_mean = train_scores_mean,train_scores_std = train_scores_std)
