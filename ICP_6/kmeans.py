import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score

dataset = pd.read_csv('CC.csv') # read the cc data

x_train = dataset.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]] # Normalization for 17 rows using index by row function
df = x_train

nulls = pd.DataFrame(x_train.isnull().sum().sort_values(ascending=False)[:])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

x_train = x_train.select_dtypes(include=[np.number]).interpolate().dropna()
#Preprocessing the data
scaler = preprocessing.StandardScaler()
scaler.fit(x_train) # plotting the graph
X_scaled_array = scaler.transform(x_train)
X_scaled = pd.DataFrame(X_scaled_array, columns=x_train.columns)

wcss = []
##elbow method to know the number of clusters
for i in range(2,9):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,random_state=0)
    kmeans.fit(X_scaled_array)
    wcss.append(kmeans.inertia_)
    y_cluster_kmeans = kmeans.predict(X_scaled_array)
    score = silhouette_score(X_scaled_array, y_cluster_kmeans, metric='euclidean')
    print("For clusters = {}, silhouette score is {})".format(i, score))

plt.plot(range(2,9),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()