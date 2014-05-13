import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import decomposition
from sklearn.cluster import KMeans, MiniBatchKMeans

def file2list(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
    return rawdata

def init_data():
    files = ['./data/no1_noun.txt',
                './data/no2_noun.txt',
                './data/no3_noun.txt',
                './data/park1_noun.txt',
                './data/park2_noun.txt',
                './data/park3_noun.txt',
                ]
    return map(file2list, files)

def vectorize(data):
    vectorizer = CountVectorizer(max_df=10, min_df=2)
    counts = vectorizer.fit_transform(data)
    tfidf = TfidfTransformer().fit_transform(counts)
    print tfidf.shape
    return tfidf

if __name__ == '__main__':
    init_data = init_data()
    data = vectorize(init_data)
    km = KMeans(n_clusters=2, init='random')
    km.fit(data)
    print km.labels_
