
# coding: utf-8


import numpy as np
import cPickle
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def sigmoid(x):
    x = np.clip( x, -500, 500)
    return 1. / (1 + np.exp(-x))

class RBM(object):

    def __init__(self, n_visible, n_hidden, mbtsz, epochs, eta, mrate, np_rng, weightinit=0.001,               lambda1 = 1, lambda2 = 1, lambda3 = 1,lambda4 = 1,                p_A_vk = None,p_s_vk = None,p_Aj_vk= None,p_sj_vk = None):
        """
        CD-k training of RBM with SGD + Momentum.
        @param n_visible:   num of lexicon
        @param n_hidden:    num of latent topics
        @param epochs:      training epochs
        @param eta:         learning rate
        @param mrate:       momentum rate
        @param mbtsz:       mini-batch size
        @param np_rng:      instances of RandomState
        @param weightinit:  scaling of random weight initialization
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.mbtsz = mbtsz
        self.epochs = epochs
        self.eta = eta
        self.mrate = mrate
        self.np_rng = np_rng
        self.W = weightinit * np_rng.randn(n_visible, n_hidden)
        self.vbias = weightinit * np_rng.randn(n_visible)
        self.hbias = np.zeros((n_hidden))
        # for momentum
        self.mW = np.zeros((n_visible, n_hidden))
        self.mvbias = np.zeros((n_visible))
        self.mhbias = np.zeros((n_hidden))

        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.p_A_vk = p_A_vk
        self.p_s_vk = p_s_vk
        self.p_Aj_vk = p_Aj_vk
        self.p_sj_vk = p_sj_vk
    def train(self, data):
        for epoch in range(self.epochs):
            print(epoch)
            #self.np_rng.shuffle(data)
            for i in range(0, data.shape[0], self.mbtsz):
                mData = data[i:i + self.mbtsz]
                ph_mean, nv_samples, nh_means = self.cd_k(mData)

                self.mW = self.mW * self.mrate + (np.dot(mData.T, ph_mean) - np.dot(nv_samples.T, nh_means))
                self.mvbias = self.mvbias * self.mrate + np.mean(mData - nv_samples, axis=0)
                self.mhbias = self.mhbias * self.mrate + np.mean(ph_mean - nh_means, axis=0)
                
                prior1 = np.zeros((self.n_visible,self.n_hidden))
                prior2 = np.zeros((self.n_visible,self.n_hidden))
                prior3 = np.zeros((self.n_visible,self.n_hidden))
                prior4 = np.zeros((self.n_visible,self.n_hidden))
                for doc in range(mData.shape[0]):

                    vk = mData[doc]
                    Gj = np.zeros((self.n_visible,self.n_hidden))

                    for i in range(0,self.n_visible):
                        if vk[i] != 0:                 
                            for j in range(0,self.n_hidden):
                                Gj[i][j] = self._logistic(self.W[i][j]*vk[i]+self.hbias[j])
                                if 1/(1+Gj[i][j]) == self.p_Aj_vk[i][j] or 1/(1+Gj[i][j]) == self.p_A_vk[i][j] or 1/(1+Gj[i][j]) == self.p_s_vk[i][j] or 1/(1+Gj[i][j]) == self.p_sj_vk[i][j]:
                                    continue
                                if self.p_Aj_vk[i][j] != 0:
                                    prior1[i][j] += 2*Gj[i][j]*vk[i]/((1+Gj[i][j])**2*(1/(1+Gj[i][j])-self.p_Aj_vk[i][j]))

                                #if 1/(1+Gj[i][j])-self.p_A_vk[i][j] != 0:
                                if self.p_A_vk[i][j] != 0:
                                    prior2[i][j] += 2*Gj[i][j]*vk[i]/((1+Gj[i][j])**2*(1/(1+Gj[i][j])-self.p_A_vk[i][j]))

                                if self.p_s_vk[i][j] != 0:
                                    prior3[i][j] += 2*Gj[i][j]*vk[i]/((1+Gj[i][j])**2*(1/(1+Gj[i][j])-self.p_s_vk[i][j]))

                                if self.p_sj_vk[i][j] != 0:
                                    prior4[i][j] += 2*Gj[i][j]*vk[i]/((1+Gj[i][j])**2*(1/(1+Gj[i][j])-self.p_sj_vk[i][j]))
                
                self.W += self.eta * self.mW - self.lambda1*prior1 - self.lambda2*prior2-self.lambda3*prior3 - self.lambda4*prior4
                #self.W += self.eta * self.mW 
                self.vbias += self.eta * self.mvbias
                self.hbias += self.eta * self.mhbias

    def cd_k(self, data, k=1):
        D = data.sum(axis=1)
        ph_mean, ph_sample = self.sample_h(data, D)
        chain_start = ph_sample

        for step in range(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start, D) 
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples, D)
        return ph_mean, nv_samples, nh_means

    def sample_h(self, v0_sample, D):
        h1_mean = sigmoid(np.dot(v0_sample, self.W) + np.outer(D, self.hbias))
        h1_sample = self.np_rng.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return [h1_mean, h1_sample]

    def sample_v(self, h0_sample, D):
        x = np.dot(h0_sample, self.W.T)
        x = np.clip( x, -500, 500 )
        pre_soft = np.exp( x+ self.vbias)
        pre_soft_sum = pre_soft.sum(axis=1).reshape((self.mbtsz, 1))
        v1_mean = pre_soft/pre_soft_sum
        v1_sample = np.zeros((self.mbtsz, v1_mean.shape[1]))
        for i in range(self.mbtsz):
            v1_sample[i] = self.np_rng.multinomial(size=1, n=D[i], pvals=v1_mean[i])
        return [v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample, D):
        v1_mean, v1_sample = self.sample_v(h0_sample, D)
        h1_mean, h1_sample = self.sample_h(v1_sample, D)
        return [v1_mean, v1_sample, h1_mean, h1_sample]

    def wordPredict(self, topic, voc):
        vecTopics = np.zeros((topic, topic))
        for i in range(len(vecTopics)):
            vecTopics[i][i] = 1
        for i, vecTopic in enumerate(vecTopics):
            pre_soft = np.exp(np.dot(vecTopic, self.W.T) + self.vbias)
            pre_soft_sum = pre_soft.sum().reshape((1, 1))
            word_distribution = (pre_soft/pre_soft_sum).flatten()
            tmpDict = {}
            for j in range(len(voc)):
                tmpDict[voc[j]] = word_distribution[j]
            print 'topic', str(i), ':', vecTopic
            k = 0
            for word, prob in sorted(tmpDict.items(), key=lambda x:x[1], reverse=True):
                if (k < 30):
                    print word, str(prob)
                    k = k+1
            print '-'
    def run_visible(self, data,t):
        num_examples = data.shape[0]
        hidden_states = np.ones((num_examples, self.n_hidden))
        
        hidden_activations = np.dot(data, self.W)+self.hbias        
        hidden_probs = self._logistic(hidden_activations)
        '''
        for i in range(hidden_probs.shape[0]):
            m= max(hidden_probs[i])
            for j in range(hidden_probs.shape[1]):
                if hidden_probs[i][j] == m:
                    hidden_states[i][j] = 1
                else:
                    hidden_states[i][j] =0 
        '''
        threshold = t* np.ones((num_examples, self.n_hidden))
        hidden_states[:,:] = hidden_probs > threshold
        return hidden_states    

    def _logistic(self, x):
        x = np.clip( x, -500, 500 )
        return 1.0 / (1 + np.exp(-x))  

    def saveParams(self, filePath):
        cPickle.dump({'W': self.W,
                      'vbias': self.vbias,
                      'hbias': self.hbias},
                      open(filePath, 'w'))

def inputData(filePath):
    docs = []
    voc = defaultdict(lambda: len(voc))
    file = open(filePath, "r")
    for line in file:
        doc = line.rstrip().split()
        for word in doc:
            voc[word]
        cnt = Counter(doc)
        docs.append(cnt)
    file.close()
    docSize, vocSize = len(docs), len(voc)
    v = np.zeros((docSize, vocSize))
    for i in range(docSize):
        for word, freq in docs[i].most_common():
            wID = voc[word]
            v[i][wID] = freq
    return v, {v:k for k, v in voc.items()}


# In[9]:

print("Loading dataset...")
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
print 'Total number of sentences:' ,len(dataset)
tf_vectorizer = CountVectorizer(max_df = 0.85,stop_words = 'english',max_features = 10000)
tf = tf_vectorizer.fit_transform(dataset[0:15000])
tf = tf.toarray()
print 'Number of training objects: ', tf.shape[0]
print 'Number of vocabulary dictionary: ', tf.shape[1]

#vocab = tf_vectorizer.get_feature_names()
vocab = tf_vectorizer.vocabulary_
voc = defaultdict(lambda: len(voc))

for k,v in vocab.items():
    voc[v]=k
docs = tf
p_Aj_vk = np.loadtxt('p_Aj_vk')
p_A_vk = np.loadtxt('p_A_vk' )
p_s_vk = np.loadtxt('p_s_vk')
p_sj_vk = np.loadtxt('p_sj_vk')


# In[3]:

topic = 10
rbm = RBM( n_visible=len(docs[0]), 
           n_hidden=topic, 
           mbtsz=50,
           epochs=10,
           eta=0.1,
           mrate=0.8,
           np_rng=np.random.RandomState(1234),
           lambda1 = 0.04, lambda2 = 0.01, lambda3 = 0.01,lambda4 = 0.01, 
           p_A_vk = p_A_vk,p_s_vk = p_s_vk,p_Aj_vk= p_Aj_vk,p_sj_vk = p_sj_vk)

rbm.train(docs[0:15000])
rbm.wordPredict(topic, voc)



# In[12]:

label = np.loadtxt('labels')[0:15000]
result =rbm.run_visible(docs[0:15000],0.99)
from sklearn.metrics import precision_score,recall_score,f1_score

print(precision_score(label[:,0],result[:,2]))
print(recall_score(label[:,0],result[:,2]))
print(f1_score(label[:,0],result[:,2]))

print(precision_score(label[:,1],result[:,6]))
print(recall_score(label[:,1],result[:,6]))
print(f1_score(label[:,1],result[:,6]))

print(precision_score(label[:,2],result[:,8]))
print(recall_score(label[:,2],result[:,8]))
print(f1_score(label[:,2],result[:,8]))


# In[29]:

np.savetxt('rbm.txt',result,fmt = '%d')


# In[16]:

#try different parameters for better accuracy

rbm = RBM( n_visible=len(docs[0]), 
           n_hidden=topic, 
           mbtsz=50,
           epochs=10,
           eta=0.1,
           mrate=0.8,
           np_rng=np.random.RandomState(1234),
           lambda1 = 0.01, lambda2 = 0.05, lambda3 = 0.01,lambda4 = 0.01, 
           p_A_vk = p_A_vk,p_s_vk = p_s_vk,p_Aj_vk= p_Aj_vk,p_sj_vk = p_sj_vk)

rbm.train(docs[0:15000])
rbm.wordPredict(topic, voc)


# In[44]:

label = np.loadtxt('labels')[0:15000]
result =rbm.run_visible(docs[0:15000],0)

from sklearn.metrics import precision_score,recall_score,f1_score
print(precision_score(label[:,0],result[:,6]))
print(recall_score(label[:,0],result[:,6]))
print(f1_score(label[:,0],result[:,6]))

print(precision_score(label[:,1],result[:,7]))
print(recall_score(label[:,1],result[:,7]))
print(f1_score(label[:,1],result[:,7]))
print(precision_score(label[:,2],result[:,8]))
print(recall_score(label[:,2],result[:,8]))
print(f1_score(label[:,2],result[:,8]))


