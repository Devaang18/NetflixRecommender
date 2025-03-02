import numpy as np
from sklearn.decomposition import PCA

def transform(tfidf_matrix):
    pca = PCA(random_state=42)
    pca.fit(tfidf_matrix.toarray()) 

    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    desired_variance = 0.90
    optimal_n_components = np.argmax(cum_explained_variance >= desired_variance) + 1

    pca = PCA(n_components=optimal_n_components, random_state=42)
    reduced_tfidf_matrix = pca.fit_transform(tfidf_matrix.toarray())
    return reduced_tfidf_matrix