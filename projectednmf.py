from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import ProjectedGradientNMF
from sklearn.metrics.pairwise import euclidean_distances

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

def get_tfidf(data):
    vectorizer = CountVectorizer(max_df=10, min_df=2)
    counts = vectorizer.fit_transform(data)
    return TfidfTransformer().fit_transform(counts)

def decompose_by_nnf(debug=True):
    initdata = init_data()

    vectorizer = CountVectorizer(max_df=10, min_df=2)
    counts = vectorizer.fit_transform(initdata)
    tfidf =  TfidfTransformer().fit_transform(counts)

    nmf = ProjectedGradientNMF(beta=1.3, eta=0.2, init='nndsvd', n_components=2, sparseness='components').fit(tfidf)
    feature_names = vectorizer.get_feature_names()

    if debug:
        for topic_idx, topic in enumerate(nmf.components_):
            print "Topic #%d:" % topic_idx
            #print " ".join([str(i) for i in topic.argsort()[:-100:-1]])
            print " ".join([feature_names[i] for i in topic.argsort()[:-100:-1]])

    print euclidean_distances(nmf.components_, tfidf[0,:])
    print euclidean_distances(nmf.components_, tfidf[1,:])
    print euclidean_distances(nmf.components_, tfidf[2,:])
    print euclidean_distances(nmf.components_, tfidf[3,:])
    print euclidean_distances(nmf.components_, tfidf[4,:])
    print euclidean_distances(nmf.components_, tfidf[5,:])

if __name__ == '__main__':
    decompose_by_nnf()
