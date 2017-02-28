import pandas as pd
import os

BASE_DIR = '.'
VE_DIR = BASE_DIR + '/wordEmbeddings/'
VE_FNAME = 'vectorsFastText.vec'
TEXT_DATA_DIR = BASE_DIR + '/textData/'
TEXT_DATA_FNAME = 'data.csv'

path = os.path.join(TEXT_DATA_DIR, TEXT_DATA_FNAME)
df = pd.read_csv(path, sep='\t', header=0)
print(df.head() )
print(df.eligible[df['eligible']== True ].__len__())

print(df['intervention_name'].unique().__len__() )
print(df['eligibility'].unique().__len__() )
print(df['eligible'].describe())
s = df.intervention_name.value_counts()
print(df.eligibility.value_counts())
print(s)
