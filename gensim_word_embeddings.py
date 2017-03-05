# import modules & set up logging
import gensim, logging
from gensim.models import Doc2Vec ,Phrases

import os
from preprocessor import MySentences


from num2words import num2words
import re

fname = "./wordEmbeddings/vectorsGensim.bin"
dataDirectory  =  '/Users/aureliabustos/Downloads/search_result/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#generate word vectors
#bigram = Phrases.load("bigrams")
print("Starting to generate word vectors using gensim w/o bigrams")
sentences = MySentences(dataDirectory) # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=5)
model.save(fname)
print("end")

model = gensim.models.Word2Vec.load(fname)
print(model.similarity('woman', 'man'))
print(model.similarity('nivolumab', 'radiotherapy'))
print(model.doesnt_match("nivolumab ipilimumab pembrolizumab chemotherapy".split()) )
print(model.most_similar('day'))
