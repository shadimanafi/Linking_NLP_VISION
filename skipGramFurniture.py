import sys
sys.path.append('/s/red/a/nobackup/cwc-ro/.conda/ScanEnviron4/lib/python3.8/site-packages')
import spacy.cli
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
from gensim.models import word2vec
import multiprocessing
import pandas as pd
import csv
import re
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
#
# grammar = r'''
#     # VP: {<VB.*><DT>?<JJ>*<NN><RB.?>?}
#     # VP: {<DT>?<JJ>*<NN><VB.*><RB.?>?}
#     # NP: {<DT>?<JJ>*<NN>}
#     '''
# chunkParser = nltk.RegexpParser(grammar)
#
# # tokenizer = nltk.RegexpTokenizer(r"\w+")
review_file='/s/red/a/nobackup/cwc-ro/shadim/BigFurniturePack4/furniture/' \
            'amazon_reviews_us_Furniture_v1_00.tsv/amazon_reviews_us_Furniture_v1_00.tsv'
with open(review_file) as f:
    x=f.readlines()
reviews=pd.read_csv(review_file,sep='\t', error_bad_lines=False)
sentences=reviews['review_body']
# punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
#
# # Removing punctuations in string
# # Using loop + punctuation string
# print(len(sentences))
# # for si,sen in enumerate(sentences):
# #     print(si)
# #     if(isinstance(sen, str)):
# #         for letter in sen:
# #             sen.replace(letter, "")
senSplited=[]
for id,sen in enumerate(sentences):
    if (isinstance(sen, str)):
        ##1
        # senSplited.append(sen.split(' '))
        ##2
        # tagged = nltk.pos_tag(nltk.word_tokenize(sen))
        # tree = chunkParser.parse(tagged)
        # senTeokenized = []
        # for subtree in tree:
        #     if (not isinstance(subtree, tuple) and subtree.label() == '# VP'):
        #         for l in subtree.leaves():
        #             senTeokenized.append(l[0])
        #
        #     elif (isinstance(subtree, tuple) and re.match('VB*', subtree[1])):
        #         senTeokenized.append(subtree[0])
        ##3
        senTeokenized=[]
        doc=nlp(sen)
        for token in doc:
            if token.is_alpha and token.head.lemma_ not in senTeokenized:
                # print(token.orth_, token.tag_, token.head.lemma_)
                senTeokenized.append(token.head.lemma_)
        senSplited.append(senTeokenized)
        print(id)
#
# # with open('reviews', 'w') as f:
# #     # using csv.writer method from CSV package
# #     write = csv.writer(f)
# #     write.writerows(sentences.values)
#
# # sentences.to_csv("reviews", index=False)
#
EMB_dim=300


w2v=word2vec.Word2Vec(senSplited,vector_size=EMB_dim, window=5, min_count=5,negative=15,epochs=10,workers=multiprocessing.cpu_count())

word_vectors=w2v.wv
print(word_vectors.similar_by_word('desk'))
print(word_vectors.similar_by_word('sofa'))
print(word_vectors.similar_by_word('couch'))
word_vectors.similar_by_vector(word_vectors.word_vec('sofa')-word_vectors.word_vec('couch')+word_vectors.word_vec('door'))

print(1)
#
# # l=[[1,2],[3,4]]
# # with open('sensNoPuncSplited', 'w') as f:
# #     # using csv.writer method from CSV package
# #     write = csv.writer(f)
# #     write.writerows(l)
# # x=[]
# # with open('sensNoPuncSplited', 'r') as f:
# #     # using csv.writer method from CSV package
# #     for l in f:
# #         x.append(l.split('\n')[0].split(','))
#
#
#
#
#

# from stat_parser import Parser
# parser = Parser()
# print (parser.parse("Great style, good quality.  I am trying to use it as a food table but I don't think it likes to be moved often.  I am going to try some Loctite...probably fix the issue."))



