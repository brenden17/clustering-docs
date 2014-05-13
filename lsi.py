from gensim import corpora, models, similarities
import numpy as np

def file2list(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        lines = rawdata.split('\n')
        l = []
        for line in lines:
            l.extend(line.split(' '))
    return l

def prepare_data():
    files = ['data/no1_noun.txt',
              'data/no2_noun.txt',
              'data/no3_noun.txt',
              'data/park1_noun.txt',
              'data/park2_noun.txt',
              'data/park3_noun.txt']

    texts = map(file2list, files)
    dictionary = corpora.Dictionary.from_documents(texts)
    dictionary.save('data/text.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.save_corpus('data/text.mm', corpus)
    corpora.BleiCorpus.save_corpus('data/text.lda-c', corpus)

 
#prepare_data()

files = ['data/no1_noun.txt',
              'data/no2_noun.txt',
              'data/no3_noun.txt',
              'data/park1_noun.txt',
              'data/park2_noun.txt',
              'data/park3_noun.txt']

texts = map(file2list, files)
dictionary = corpora.Dictionary.from_documents(texts)
#dictionary.save('data/text.dict')
corpus = [dictionary.doc2bow(text) for text in texts]
#corpora.MmCorpus.save_corpus('data/text.mm', corpus)
corpus = corpora.MmCorpus('data/text.mm')

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) 
for ll in lsi.print_topics(2):
    print('-----')
    print ll

for text in texts:
    vec_bow = dictionary.doc2bow(text)
    vec_lsi = lsi[vec_bow]
    print(vec_lsi)
