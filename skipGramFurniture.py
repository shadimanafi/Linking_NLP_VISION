
import spacy.cli
nlp = spacy.load("en_core_web_sm")
from gensim.models import word2vec
import multiprocessing
import pandas as pd
import csv
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

review_file='.../amazon_reviews_us_Furniture_v1_00.tsv/amazon_reviews_us_Furniture_v1_00.tsv'
with open(review_file) as f:
    x=f.readlines()
reviews=pd.read_csv(review_file,sep='\t', error_bad_lines=False)
sentences=reviews['review_body']

senSplited=[]
for id,sen in enumerate(sentences):
    if (isinstance(sen, str)):

        senTeokenized=[]
        doc=nlp(sen)
        for token in doc:
            if token.is_alpha and token.head.lemma_ not in senTeokenized:
                # print(token.orth_, token.tag_, token.head.lemma_)
                senTeokenized.append(token.head.lemma_)
        senSplited.append(senTeokenized)
        print(id)

EMB_dim=300


w2v=word2vec.Word2Vec(senSplited,vector_size=EMB_dim, window=5, min_count=5,negative=15,epochs=10,workers=multiprocessing.cpu_count())

word_vectors=w2v.wv
print(word_vectors.similar_by_word('desk'))
print(word_vectors.similar_by_word('sofa'))
print(word_vectors.similar_by_word('couch'))
word_vectors.similar_by_vector(word_vectors.word_vec('sofa')-word_vectors.word_vec('couch')+word_vectors.word_vec('door'))

