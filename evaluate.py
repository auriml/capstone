from __future__ import division
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy


class autovivify_list(dict):
    '''Pickleable class to replicate the functionality of collections.defaultdict'''
    def __missing__(self, key):
        value = self[key] = []
        return value

    def __add__(self, x):
        '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
        '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError



def build_word_vector_matrix(vector_file, n_words):
    '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
    numpy_arrays = []
    labels_array = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            if len(sr[1:])>1:
                labels_array.append(sr[0])
                numpy_arrays.append(numpy.array([float(i) for i in sr[1:]]))
            if c == n_words:
                return numpy.array(numpy_arrays), labels_array

    return numpy.array(numpy_arrays), labels_array


def find_word_clusters(labels_array, cluster_labels):
    '''Read the labels array and clusters label and return the set of words in each cluster'''
    cluster_to_words = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_words[i].append(labels_array[c])
    return cluster_to_words


# if __name__ == "__main__":
#     input_vector_file = sys.argv[1] # The Glove file to analyze (e.g. glove.6B.300d.txt)
#     n_words           = int(sys.argv[2]) # The number of lines to read from the input file
#     reduction_factor  = float(sys.argv[3]) # The desired amount of dimension reduction
#     clusters_to_make  = int( n_words * reduction_factor ) # The number of clusters to make
#     df, labels_array  = build_word_vector_matrix(input_vector_file, n_words)
#     kmeans_model      = KMeans(init='k-means++', n_clusters=clusters_to_make, n_init=10)
#     kmeans_model.fit(df)
#
#     cluster_labels    = kmeans_model.labels_
#     cluster_inertia   = kmeans_model.inertia_
#     cluster_to_words  = find_word_clusters(labels_array, cluster_labels)
#
#     for c in cluster_to_words:
#         print cluster_to_words[c]
#         print "\n"


input_vector_file = "wordEmbeddings/vectorsFastText.vec"
n_words =  10000 # The number of lines to read from the input file
reduction_factor = 0.1 # The desired amount of dimension reduction
clusters_to_make = int( n_words * reduction_factor ) # The number of clusters to make
df, labels_array = build_word_vector_matrix(input_vector_file, n_words)
kmeans_model = KMeans(init='k-means++', n_clusters=clusters_to_make, n_init=10)
kmeans_model.fit(df)
cluster_labels = kmeans_model.labels_
cluster_inertia = kmeans_model.inertia_
cluster_to_words = find_word_clusters(labels_array, cluster_labels)

for c in cluster_to_words:
    print(cluster_to_words[c])
    print("\n")

#compare  wordembeddings within each API:
    #modify parameters to generate differentword embedding within each technology
    #use or not bigrams/trigrams

#compare word embeddings between APIs:
    #generate clusters with all other word embeddings and compare them

#compare text classifying models within each API:
    #modifying classified data (data.csv and conditionData.csv)

#compare text classifying models between APIs:
    #classify with fast_text and compare with Keras



