import fasttext
import os
import pandas as pd
import numpy as np





VALIDATION_SPLIT = 0.2
TRAIN_MODEL = False

classifier_fname = "./classifiers/model_fasttext_sample"
data_test =  "./textData/testClassifier.csv"
df_test = pd.read_csv(data_test, sep='\t', header=None, names = ["y", "x"])

classifier = fasttext.load_model(classifier_fname +'.bin')
# predict with the probability
labels = classifier.predict_proba(df_test.x.apply(str))
result = classifier.test(data_test)
print ('P@1:', result.precision)
print ('R@1:', result.recall)
print ('Number of examples:', result.nexamples)

df_test= pd.concat([pd.Series(labels),df_test], axis = 1)
print(df_test)