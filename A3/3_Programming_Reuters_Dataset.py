import html
import math
import re
from sklearn.cluster import KMeans
from html.parser import HTMLParser
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import glob
from sklearn.cluster import dbscan
import time

class DatasetParser(HTMLParser):
    def __init__(self, encoding='latin-1'):
        html.parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def _reset(self):
        self.in_body = False
        self.in_topics = False
        self.in_topic_d = False
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_starttag(self, tag, attrs):
        if tag == "reuters":
            pass
        elif tag == "body":
            self.in_body = True
        elif tag == "topics":
            self.in_topics = True
        elif tag == "d":
            self.in_topic_d = True 

    def handle_endtag(self, tag):
        if tag == "reuters":
            self.body = re.sub(r'\s+', r' ', self.body)
            self.docs.append( (self.topics, self.body) )
            self._reset()
        elif tag == "body":
            self.in_body = False
        elif tag == "topics":
            self.in_topics = False
        elif tag == "d":
            self.in_topic_d = False
            self.topics.append(self.topic_d)
            self.topic_d = ""  

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_topic_d:
            self.topic_d += data

def obtain_topic_tags():
    topics = open(
        "./all-topics-strings1.lc", "r"
    ).readlines()
    topics = [t.strip() for t in topics]
    return topics

def filter_doc_list_through_topics(topics, docs):
    ref_docs = []
    for d in docs:
        if d[0] == [] or d[0] == "":
            continue
        for t in d[0]:
            if t in topics:
                d_tup = (t, d[1])
                ref_docs.append(d_tup)
                break
    return ref_docs

def create_tfidf_training_data(docs):
    # Create the training data class labels
    y = [d[0] for d in docs]
    # Create the document corpus list
    corpus = [d[1] for d in docs]
    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, y

if __name__ == "__main__":
    files = glob.glob('./*.sgm')
    parser = DatasetParser()
    docs = []
    for fn in files:
        for d in parser.parse(open(fn, 'rb')):
            docs.append(d)

    topics = obtain_topic_tags()
    ref_docs = filter_doc_list_through_topics(topics, docs)
    
    X, y = create_tfidf_training_data(ref_docs)
    print(X)
    print(y)
    
UNCLASSIFIED = False
NOISE = None

def regionset(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        p = m[:,point_id]
        q = m[:,i]
        if((math.sqrt(np.power(p-q,2).sum())) < eps):
            seeds.append(i)
    return seeds

def constructcluster(m, classifications, point_id, cluster_id, eps, min_points):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        p = m[:,point_id]
        q = m[:,i]
        if((math.sqrt(np.power(p-q,2).sum())) < eps):
            seeds.append(i)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = regionset(m, current_point, eps)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def customdbscan(m, eps, min_points):
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(0, n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if constructcluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications

def computedbscan_ds2():
    eps = .5
    min_points = 5
    inputData = np.array(X.todense())
    labels_points = customdbscan(inputData, eps, min_points)
    print('DBScan Results')
    print(labels_points)

start_time = time.time()
computedbscan_ds2()
print("---Custom Algorithm DBSCAN -  %s seconds ---" % (time.time() - start_time))

start_time = time.time()
kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
print(kmeans.labels_)
print("---K means - %s seconds ---" % (time.time() - start_time))

inputData = np.array(X.todense())
t = time.time()
db1 = dbscan(inputData, eps=0.85, min_samples=2)
print("Inbuilt DBSCAN - clustering took %.5f seconds" %(time.time()-t))

values = ['acquisitions','corn','crude','earn','grain','interest','money-fx','ship','trade','wheat']
true_y = []
for val in y:
    true_y.append(values.index(val))
    
#print(true_y)
y_pred= kmeans.labels_

max_values = []
#From Result compute purity / impurity
a = confusion_matrix(true_y, y_pred)

max_sum = 0.0
sum_total = 0.0
for i in range(0,a.shape[0]):
    values = a[:][i]
    print('row -',i)
    print(values)
    print(max(values))
    max_sum = max_sum + max(values)
    sum_total = sum_total + sum(values)

print('max is',sum_total)
print('max values sum is ',max_sum)
print('purity')
print(max_sum/sum_total)

print('F1 Score')
print(f1_score(true_y, y_pred, average='macro'))

print('Adjusted Rand Score')
print(adjusted_rand_score(true_y, y_pred))

print('normalized_mutual_info_score')
print(normalized_mutual_info_score(true_y, y_pred))

