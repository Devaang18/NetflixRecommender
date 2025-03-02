from sklearn.feature_extraction.text import TfidfVectorizer
import pca

def tfidf(clustered_features):
    tfidf_vectoriser = TfidfVectorizer(stop_words='english', lowercase=False,max_features = 2000)
    tfidf_matrix = tfidf_vectoriser.fit_transform(clustered_features)
    reduced_tfidf_matrix = pca.transform(tfidf_matrix.toarray())

    return reduced_tfidf_matrix