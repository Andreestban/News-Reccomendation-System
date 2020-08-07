import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords,wordnet
from nltk import pos_tag,WordNetLemmatizer
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from sklearn.decomposition import PCA

df = pd.read_csv("news_data.csv")
df = df.drop(["Unnamed: 0"],axis=1)
df.isnull().sum()

sns.countplot(x = 'category',data = df)

text= []
for i in tqdm(range(len(df))):
    text.append(df["headline"][i])


cat_map = {}
for i in tqdm(range(len(df))):
    if df["category"][i] not in cat_map:
        cat_map[df["category"][i]] = [i]
    else:
        cat_map[df["category"][i]].append(i)

def text_preproccesing(text):
    hm = defaultdict(lambda: wordnet.NOUN)
    hm['J'] = wordnet.ADJ
    hm['V'] = wordnet.VERB
    hm['R'] = wordnet.ADV
    for i in tqdm(range(len(text))):
        text[i] = re.sub('[^a-zA-Z]',' ',text[i])
        text[i] = text[i].split()
        text[i] = [word for word in text[i] if not word in set(stopwords.words("english"))]
        lemm = WordNetLemmatizer()
        sent = ""
        for token,tag in pos_tag(text[i]):
            word = lemm.lemmatize(token,hm[tag[0]])
            sent += word+ " "
        sent = sent[:-1]
        text[i] = sent.lower()
    return text

text = text_preproccesing(text)
    
def Vectorization(corpus):
    vector = TfidfVectorizer(max_features = 3500)
    X = vector.fit_transform(corpus).toarray()
    return X,vector

X,vector = Vectorization(text)

#only execute when required
def elbow_curve(X):
    wcss = []
    for i in tqdm(range(1,81)):
        kmeans = KMeans(n_clusters = i, init = 'k-means++',max_iter = 300, n_init = 80,random_state = 21 ,n_jobs = -1)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,81),wcss)
    plt.xlabel("Iteration")
    plt.ylabel("WCSS")

elbow_curve(X)

        
kmeans = KMeans(n_clusters = 7, init  = "k-means++", max_iter = 300, n_init = 10, random_state = 21)
kmeans.fit(X)
y = kmeans.predict(X)

def pcaPlot(X,y):
    pca = PCA(n_components = 3)
    comp = pca.fit_transform(X)
    comp = pd.DataFrame(comp)
    comp['label'] = pd.DataFrame(y)
    sns.pairplot(data = comp,hue = 'label')    
    
pcaPlot(X,y)

def Mapping(kmeans):
    label = kmeans.labels_
    map = {}
    for l in tqdm(range(len(label))):
        if  label[l] not in map:
            map[label[l]] = [l]
        else:
            map[label[l]].append(l)
    return map

map = Mapping(kmeans)
   
global X
global y
global map
global cat_map

def getReccomendation(index,df):
    #recc by Model
    prediction = y[index]
    vector_arr= []
    for i in map[prediction]:
        if i!= index:
            vector_arr.append(X[i])
    vector_arr = np.array(vector_arr)
    sim_array = cosine_similarity([X[index]],vector_arr)
    sim_array = list(sim_array[0])
    
    #general recc. from category
    cat = df["category"][index]
    new_vect_arr =[]
    for i in cat_map[cat]:
        if i!= index:
            new_vect_arr.append(X[i])
    new_vect_arr = np.array(new_vect_arr)
    cat_sim = cosine_similarity([X[index]],new_vect_arr)
    cat_sim = list(cat_sim[0])
    K=10
    gen_rec= []
    heap =[]
    cat_sim.extend(sim_array)
    new_arr = list(set(cat_sim))
    for i in range(0,len(new_arr)):
        heapq.heappush(heap,(new_arr[i],i))
        if len(heap)>K:
            heapq.heappop(heap)
    score=[]
    while(len(heap)):
        t = heapq.heappop(heap)
        score.append(t[0])
        gen_rec.append(t[1])
    return gen_rec,score


def reccomendations(index, df):
    t,score = getReccomendation(index, df)
    t,score = t[::-1],score[::-1]
    X = df.iloc[:,:].values
    arr =[]
    for i in t:
        arr.append(X[i][:])
    arr = pd.DataFrame(arr)
    arr['score'] =score
    return arr
        
rec_df = reccomendations(7,df)