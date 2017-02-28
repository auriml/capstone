__author__ = 'aureliabustos'


import numpy as np



data =  "./textData/labeledEligibilityFastText.csv"


def balanced_subsample(y, size=None):

    subsample = []

    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample


def generate_small_set(set_size =None , fname = None):
    import pandas as pd
    print('Loading text dataset')
    df = pd.read_csv(data, sep='\t', header=None, names = ["eligible", "eligibility"])
    print(df.describe())
    #balance labels
    # Apply the random under-sampling

    rus = balanced_subsample(df.eligible, size=set_size)
    df = df.iloc[rus,:]
    print("sample after under-sampling: " )
    print(df.describe() )
    if set_size != None:  #write subsample but not full sample
        fname = fname + str(set_size) + '.csv'
        df.to_csv(sep='\t', path_or_buf=fname, index=False, header = False)
    return df