import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataLoader:
    def __init__(self, df):
        self.df = df
        self.recommender_df = self.prepare_data()
        self.vectorizer = CountVectorizer()
        self.similarity_matrix = self.compute_similarity()
        self.titles = self.recommender_df.index.tolist()
        self.indices = pd.Series(range(len(self.titles)), index=self.titles)

    def prepare_data(self):
        df = self.df.copy()
        df = df[df['preprocessed_clustered_features'].str.strip() != ""].copy()
        df.set_index('title', inplace=True)
        return df

    def compute_similarity(self):
        feature_matrix = self.vectorizer.fit_transform(self.recommender_df['preprocessed_clustered_features'])
        return cosine_similarity(feature_matrix)
