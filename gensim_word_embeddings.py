# import modules & set up logging
import gensim, logging
from gensim.models import Doc2Vec ,Phrases

import os
from preprocessor import MySentences


from num2words import num2words
import re

fname = "./wordEmbeddings/vectorsGensim.bin"
dataDirectory  =  '/Users/aureliabustos/Downloads/search_result/'

import sys, getopt
# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"w")

###############################
# o == option
# a == argument passed to the o
###############################
for o, a in myopts:
    if o == '-w':
        print("Starting to generate word embeddings using Gensim and bigrams")
        dataDirectory = a
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        print("Loading bigrams")
        bigram = Phrases.load("bigrams")
        print("Starting to train word embeddings")
        sentences = MySentences(dataDirectory) # a memory-friendly iterator
        model = gensim.models.Word2Vec(sentences, min_count=5)
        model.save(fname)
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

    else:
        print("Usage: %s [-w] <dataDirectory> " % sys.argv[0])




