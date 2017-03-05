import pandas as pd
import os
import numpy as np

BASE_DIR = '.'

TEXT_DATA_DIR = BASE_DIR + '/textData/'
TEXT_DATA_FNAME = 'labeledEligibility.csv'

path = os.path.join(TEXT_DATA_DIR, TEXT_DATA_FNAME)
df = pd.read_csv(path, sep='\t', header=None, names = ["eligible", "eligibility"])
print(df.head() )
print(df['eligibility'].unique().__len__() )
print(df['eligible'].describe())

from scipy import stats, integrate
charCounts = np.array([len(s.split()) for s in df.eligibility])
print(stats.describe(charCounts))
print(np.sum(charCounts))

import matplotlib.pyplot as plt
#plt.hist(charCounts, bins=[0, 100, 200, 300, 400, 500])
#plt.savefig("eligibility_histogram.png")

plt.hist(charCounts, bins=[0, 10,20,30,40,50, 75, 100, 200, 300, 400, 500])
plt.xlabel('Words')
plt.ylabel('Clinical Statements')
plt.title('Histogram of statements word length')
plt.grid(True)
plt.savefig("eligibility_histogram_2.png")


mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(charCounts, 50, normed=1, facecolor='g', alpha=0.75)







