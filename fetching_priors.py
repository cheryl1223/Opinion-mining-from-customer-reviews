
# coding: utf-8

from __future__ import print_function
from time import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



print("Loading dataset...")
t0 = time()
doc_len = [0 for x in range(0,52624)]
doc_id = 0
with open('test_rr.txt') as f:
    content = f.readlines()
dataset = []
for line in content:
    i, words = line.split('\t')
    i = int(i)
    if doc_id == i:
        doc_len[i] = doc_len[i]+1
    else:
        doc_id = i
        doc_len[i]=1
    dataset.append(words.strip())
data_samples = dataset
print("done in %0.3fs." % (time() - t0))




tf_vectorizer = CountVectorizer(max_df = 0.8,stop_words = 'english',max_features=10000)
tf = tf_vectorizer.fit_transform(data_samples)
tf = tf.toarray()
print(tf.shape)
vocab = tf_vectorizer.get_feature_names()



from nltk.corpus import reuters
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
 
stop_words = stopwords.words('english') + list(punctuation)
 
def tokenize(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words]
    return [w for w in words if w not in stop_words and not w.isdigit()]

# build the vocabulary in one pass
vocabulary = set()
for file_id in reuters.fileids():
    words = tokenize(reuters.raw(file_id))
    vocabulary.update(words)
 
vocabulary = list(vocabulary)
word_index = {w: idx for idx, w in enumerate(vocabulary)}
 
VOCABULARY_SIZE = len(vocabulary)
DOCUMENTS_COUNT = len(reuters.fileids())
 
print (VOCABULARY_SIZE, DOCUMENTS_COUNT)     # 10788, 51581
 




word_idf = np.zeros(VOCABULARY_SIZE)
for file_id in reuters.fileids():
    words = set(tokenize(reuters.raw(file_id)))
    indexes = [word_index[word] for word in words]
    word_idf[indexes] += 1.0
 
word_idf = np.log(DOCUMENTS_COUNT / (1 + word_idf).astype(float))
print (word_idf[word_index['deliberations']])     # 7.49443021503
print (word_idf[word_index['committee']])        # 3.61286641709




mean_idf = np.mean(word_idf)
print (mean_idf)



import nltk


num_visible = tf.shape[1]
num_hidden = 10
num_sentences = tf.shape[0]

p_A_vk = np.zeros((num_visible,num_hidden))
nouns = [token for token, pos in nltk.pos_tag(vocab) if pos.startswith('N')]

tf_sum = np.sum(tf,axis = 0)

w = []
for noun in nouns:
    index = vocab.index(noun)

    if noun in word_index:
        tfidf = tf_sum[index]*word_idf[word_index[noun]]/mean_idf
    else:

        tfidf = tf_sum[index]
    w.append((noun,tfidf))
    for i in range(0,6):
        p_A_vk[index][i] = tfidf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
p_A_vk  = scaler.fit_transform( p_A_vk)
index = vocab.index('food')
print(p_A_vk[index])


index = vocab.index('chicken')
print(p_A_vk[index])


from nltk.corpus import sentiwordnet as swn
p_s_vk = np.zeros((num_visible,num_hidden))
adjs = [token for token, pos in nltk.pos_tag(vocab) if pos.startswith('J')]

w = []
for adj in adjs:
    index = vocab.index(adj)
    try:
        score = swn.senti_synset('%s.a.04'%adj)
        w.append((adj ,score.pos_score()-score.neg_score()))
        p_s_vk[index][6] = score.pos_score()
        p_s_vk[index][7] = score.neg_score()        
    except:
        continue 
index = vocab.index('good')
print(p_s_vk[index])


# In[6]:

p_sj_vk = np.zeros((num_visible,num_hidden))
w = []
for adj in adjs:
    index = vocab.index(adj)
    try:
        score = swn.senti_synset('%s.a.04'%adj)
        w.append((adj ,score.pos_score()-score.neg_score()))
        p_sj_vk[index][6] = 1- score.obj_score()
        p_sj_vk[index][7] = 1-score.obj_score()        
    except:
        continue 
index = vocab.index('great')
print(p_sj_vk[index])
print(index)


# In[7]:

from sklearn.decomposition import LatentDirichletAllocation

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        x = [(feature_names[i],model.components_[topic_idx][i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(x)


lda = LatentDirichletAllocation(n_topics=6, max_iter=20,
                                learning_method='online',
                                learning_offset=60.,
                                random_state=10)
t0 = time()
lda.fit(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names,30)


n_top_words = 50
w = []
for topic_idx, topic in enumerate(lda.components_):
    x = [(tf_feature_names[i],lda.components_[topic_idx][i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
    w.append(x)

lda_result = np.dot(tf,lda.components_.T)


for i in range(lda_result.shape[0]):
    m= max(lda_result[i])
    for j in range(lda_result.shape[1]):
        if lda_result[i][j] == m:
            lda_result[i][j] = 1
        else:
            lda_result[i][j] =0

lda_test = np.transpose(np.vstack((lda_result[:,0],lda_result[:,4],lda_result[:,5])))
label = np.loadtxt('labels')[0:15000]

from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(label[:,0],lda_test[:,0]))
print(recall_score(label[:,0],lda_test[:,0]))
print(f1_score(label[:,0],lda_test[:,0]))

print(precision_score(label[:,1],lda_test[:,1]))
print(recall_score(label[:,1],lda_test[:,1]))
print(f1_score(label[:,1],lda_test[:,1]))

print(precision_score(label[:,2],lda_test[:,2]))
print(recall_score(label[:,2],lda_test[:,2]))
print(f1_score(label[:,2],lda_test[:,2]))


# In[8]:

n_top_words = 50
w = []
for topic_idx, topic in enumerate(lda.components_):
    x = [(tf_feature_names[i],lda.components_[topic_idx][i]) for i in topic.argsort()[:-n_top_words - 1:-1]]
    w.append(x)

p_Aj_vk=np.zeros((num_visible,num_hidden))
count = 0

for (x,y) in w[0]:
    try:
        index = vocab.index(x)
        p_Aj_vk[index][0] = y
    except:
        continue
for (x,y) in w[4]:
    try:
        index = vocab.index(x)
        p_Aj_vk[index][1] = y
    except:
        continue
for (x,y) in w[5]:
    try:
        index = vocab.index(x)
        p_Aj_vk[index][2] = y
    except:
        continue

p_Aj_vk = scaler.fit_transform(p_Aj_vk)



# In[47]:

test = vocab.index('salad')
print(p_Aj_vk[test])


# In[10]:

dist = []

a = 0
l = 0
for i in range(0,52624):
    l = l+doc_len[i]
    if l < 10000:
        if i == 0:
            dist.append(tf[:doc_len[i]])
            a = doc_len[i]
        else:
            b = a + doc_len[i]
            #print(a,b)
            dist.append(tf[a:b])
            a = b
training_data = np.array(dist)
tf_sum = []
for d in dist:
    d = np.sum(d,axis=0)
    tf_sum.append(d)
tf_sum = np.array(tf_sum)
print(tf_sum.shape)
print(tf.shape)
print(training_data[0].shape)


tf_vectorizer = CountVectorizer(max_df = 0.8,stop_words = 'english',max_features=8000)
tf = tf_vectorizer.fit_transform(data_samples[0:50000])
tf = tf.toarray()
print(tf.shape)
vocab = tf_vectorizer.get_feature_names()


np.savetxt('p_Aj_vk',p_Aj_vk)
np.savetxt('p_A_vk' ,p_A_vk)
np.savetxt('p_s_vk' , p_s_vk)
np.savetxt('p_sj_vk',p_sj_vk)

