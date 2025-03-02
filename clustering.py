# CLUSTERING USING KMEANS AND AGGOLOMERATIVE CLUSTERING
# The data is first cleaned and preprocessed. 
# The data is then vectorized using TF-IDF Vectorizer. 
# The data is then reduced using PCA. 
# The optimal number of components is found using the cumulative explained variance. 
# The optimal number of clusters is found using the elbow method. (Dynamically from kneed library)
# The data is then clustered using KMeans and Agglomerative Clustering. 
# The number of movies and TV shows in each cluster is then plotted using a countplot.


from sklearn.cluster import KMeans, AgglomerativeClustering
from kneed import KneeLocator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_transfomations import cleaning_data

def trial():
  path = "/Users/devaang/Documents/Netflix_movies_and_tv_shows_clustering.csv"

  df_raw = pd.read_csv(path, index_col = 'show_id')
  df = df_raw.copy()

  df = cleaning_data(df)

  return df


'''clustered_features = df['preprocessed_clustered_features']
reduced_tfidf_matrix = tfidf(clustered_features)  

inertia = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, max_iter=300, n_init=10)
    kmeans.fit(reduced_tfidf_matrix)
    inertia.append(kmeans.inertia_)
   
kl = KneeLocator(range(1,20), inertia, curve='convex', direction='decreasing')
optimal_k_value = kl.elbow

kmeans = KMeans(n_clusters=optimal_k_value, init='k-means++', random_state=42, max_iter=300, n_init=10)
kmeans.fit(reduced_tfidf_matrix)
df['KMeans Clusters'] = kmeans.labels_

hier_clusters = AgglomerativeClustering(n_clusters=12, metric='euclidean', linkage='ward')
hier_clusters.fit_predict(reduced_tfidf_matrix)
df['Hierarchial Clusters'] = hier_clusters.labels_

plt.figure(figsize=(10,5))
q = sns.countplot(x='KMeans Clusters',data=df, hue='type')
plt.title('Number of movies and TV shows in each cluster - Kmeans Clustering')
for i in q.patches:
  q.annotate(format(i.get_height(), '.0f'), (i.get_x() + i.get_width() / 2., i.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()

plt.figure(figsize=(10,5))
q = sns.countplot(x='Hierarchial Clusters',data=df, hue='type')
plt.title('Number of movies and TV shows in each cluster - Hier Clustering')
for i in q.patches:
  q.annotate(format(i.get_height(), '.0f'), (i.get_x() + i.get_width() / 2., i.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()'''



