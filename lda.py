from gensim import corpora, models, similarities
import numpy as np

def file2list(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
        data = rawdata.split('\n')
    return data

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

corpus = corpora.BleiCorpus('data/text.lda-c')
#tfidf = models.TfidfModel(corpus)
#corpus_tfidf = tfidf[corpus]
model = models.ldamodel.LdaModel(corpus,
                                num_topics=2,
                                id2word=corpus.id2word)

print model.print_topics(5)
"""
topics = model.show_topics(topics=-1, topn=10, formatted=False)

for ti, topic in enumerate(topics):
    for t in topics:
        print t[1], t[0]
    #print 'topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in
    #     topic))
"""
