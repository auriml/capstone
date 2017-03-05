import fasttext
import numpy as np




data =  "./textData/words_data.csv"
skipgram_fname = "./wordEmbeddings/vectorsFastText"
bow_fname = "./wordEmbeddings/vectorsFastText_bow"
plot_filename = "tsne_relations.png"


# Skipgram model
model = fasttext.skipgram(data, skipgram_fname, ws = 5, minn = 3 ,maxn = 6, silent = 0, epoch = 10, bucket = 2000000)



# CBOW model
#model = fasttext.cbow(data, bow_fname)


#default values:
#  -lr 0.025 -dim 100  -ws 5 -epoch 1 -minCount 5 -neg 5 -loss ns -bucket 2000000
# -minn 3 -maxn 6 -thread 4 -t 1e-4 -lrUpdateRate 100

model = fasttext.load_model('./wordEmbeddings/vectorsFastText.bin')
print("Number of word embeddings in the model: " + str(len(model.words)))
#print(model.words) # list of words in dictionary
print("get the vector of word 'patient': " + str(model['patient']))      # get the vector of a word
print("Model name: " +model.model_name)       # Model name
print("Size of word vector: " + str(model.dim))              # Size of word vector
print("Size of context window: " + str(model.ws) )              # Size of context window
print("Number of epochs: " + str(model.epoch) )           # Number of epochs
print("Minimal number of word occurences: " + str(model.min_count) )       # Minimal number of word occurences
print("Number of negative sampled: " + str(model.neg) )             # Number of negative sampled
print("Max length of word ngram: " + str(model.word_ngrams) )     # Max length of word ngram
print("Loss function name: " + str(model.loss_name))        # Loss function name
print("Number of buckets: " + str(model.bucket))           # Number of buckets
print("Min length of char ngram: " + str(model.minn))             # Min length of char ngram
print("Max length of char ngram: " + str(model.maxn))             # Max length of char ngram
print("Rate of updates for the learning rate: " + str(model.lr_update_rate))   # Rate of updates for the learning rate
print("Value of sampling threshold: " + str(model.t) )               # Value of sampling threshold
print("Encoding of the model: " + model.encoding)         # Encoding of the model


#word space visualization
#select a set of words  to analyze
targets = ['tamoxifen', 'antiestrogen', 'fulvestrant', 'palbociclib','abemaciclib','itraconazole',
           'clarithromycin', 'erythromycin', 'voriconazole',
           'fatigue', 'insomnia',  'headache', 'pain'
           'bepridil', 'chlorpromazine', 'thioridazine',
           'antibiotics', 'fungal', 'antibiotic', 'antiviral',
           'rectal', 'anal',  'sphincter',
           'hemoglobin', 'platelets',
           'bronchi', 'trachea', 'bronchus', 'breast']



#get the word vectors of these words
X_target=[]
for w in targets:
    X_target.append(model[w])
X_target = np.asarray(X_target)

#select a subset of the dataset
word_list = list(model.words)[:10000]
X_subset=[]
for w in word_list:
    X_subset.append(model[w])
X_subset = np.asarray(X_subset)

#add both datasets together
X_target = np.concatenate((X_subset, X_target))
print(X_target.shape)

# compute the t-SNE algorithm.
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, perplexity=40, init='pca', method='exact',
              random_state=0, n_iter=200, verbose=2).fit_transform(X_target)


print(X_tsne.shape)

# as we can not visualize all words, we
# are only to represent the target words
X_tsne_target = X_tsne[-len(targets):,:]
print(X_tsne_target.shape)
import matplotlib.pyplot as plt
def plot_words(X, labels, classes=None, xlimits=None, ylimits=None):
    fig = plt.Figure(figsize=(6, 6))
    if xlimits is not None:
        plt.xlim(xlimits)
    if ylimits is not None:
        plt.ylim(ylimits)
    plt.scatter(X[:, 0], X[:, 1])
    for i, txt in enumerate(labels):
        plt.annotate(txt, (X[i, 0], X[i, 1]))


#plot_words(X_tsne_target, targets)


# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename= plot_filename):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(12, 12))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #plot_only = 500
    #low_dim_embs = tsne.fit_transform(model[:plot_only, :])
    #labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(X_tsne_target, targets)

except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
