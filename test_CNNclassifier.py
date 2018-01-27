from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd


BASE_DIR = '.'

TEXT_DATA_DIR = BASE_DIR + '/textData/'
RESULT_DIR = BASE_DIR + '/classifiers/'
RESULT_FNAME = 'model_custom_CNN'
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000

data_test =  "./textData/testClassifier.csv"
df_test = pd.read_csv(data_test, sep='\t', header=None, names = ["y", "x"])


classifier = load_model(filepath= RESULT_DIR + RESULT_FNAME +'2100')
# vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(df_test.x.apply(str))
sequences = tokenizer.texts_to_sequences(df_test.x.apply(str))
series_sequences = pd.Series(sequences)
tests =  pad_sequences(series_sequences, MAX_SEQUENCE_LENGTH)
df_result = pd.DataFrame(classifier.predict(tests), columns =["y_0", "y_1"] )
df_test = pd.concat([df_test, df_result], axis = 1)

print(df_test)