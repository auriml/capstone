# import modules & set up logging
import gensim, logging
from gensim.models import Doc2Vec ,Phrases

import os
from preprocessor import MySentences


from num2words import num2words
import re

fname = "./wordEmbeddings/vectorsGensim_cbow.bin"
dataDirectory  =  '/Users/aureliabustos/Downloads/search_result/'

import sys, getopt
# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"i:")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-i':
        print("Starting to generate word embeddings using Gensim and bigrams")
        dataDirectory = a
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("Loading bigrams")
        bigram = Phrases.load("bigrams")
        print("Starting to train word embeddings")
        sentences = MySentences(dataDirectory) # a memory-friendly iterator
        model = gensim.models.Word2Vec(sentences, min_count=5, sg=0)
        #model = gensim.models.Word2Vec(bigram[sentences], min_count=5, sg=1)
        model.save(fname)
        del model
        print("end")
        print("Loading embedding model")
        model = gensim.models.Word2Vec.load(fname)
        print("Some tests:")
        print("Similarity between uterus and ovary:")
        print(model.similarity('uterus', 'ovary'))
        print("Similarity between nivolumab and radiotherapy:")
        print(model.similarity('nivolumab', 'radiotherapy'))
        print("Which one is not matching? nivolumab ipilimumab pembrolizumab chemotherapy")
        print(model.doesnt_match("nivolumab ipilimumab pembrolizumab chemotherapy".split()) )
        print("Most similar words to 'day'? ")
        print(model.most_similar('day'))
        print("Tamoxifen is used to treat breast cancer as X is used to treat prostate cancer? ")
        print(model.wv.most_similar_cosmul(positive=['prostate', 'tamoxifen'], negative=['breast']))

    else:
        print("Usage: %s [-i] <dataDirectory> " % sys.argv[0])




