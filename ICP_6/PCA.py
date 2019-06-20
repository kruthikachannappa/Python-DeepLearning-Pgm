from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score

sns.set(style="white", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

dataset = pd.read_csv('CC.csv')
x = dataset.iloc[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]
y =  dataset.iloc[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

nulls = pd.DataFrame(x.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
data = x.select_dtypes(include=[np.number]).interpolate().dropna()
print(sum(data.isnull().sum() != 0))
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(data)
X_scaled_array = scaler.transform(data)
X_scaled = pd.DataFrame(X_scaled_array, columns = data.columns)

pca = PCA(2)
train_img = pca.fit_transform(X_scaled_array)
test_img = pca.fit_transform(X_scaled_array)
df2 = pd.DataFrame(data=train_img)

from sklearn.cluster import KMeans
nclusters = 3 # this is the k in kmeans
seed = 0
km = KMeans(n_clusters=nclusters, random_state=seed)
km.fit(df2)

y_cluster_kmeans = km.predict(df2)
from sklearn import metrics
score = metrics.silhouette_score(df2, y_cluster_kmeans)
print("score",score)
scores = metrics.silhouette_samples(df2, y_cluster_kmeans)
sns.distplot(scores)
print("scores",score)
# # can we add the species info to that plot?
# # well, can plot them separately using pandas -
df_scores = pd.DataFrame()
df_scores['SilhouetteScore'] = scores


wcss = []
##elbow method to know the number of clusters
for i in range(2,9):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)



plt.plot(range(2,9),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()


