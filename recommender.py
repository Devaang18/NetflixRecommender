class Recommender:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def get_recommendations(self, title, num_recommendations=10):
        if title not in self.data_loader.indices:
            return ['Invalid Entry: Title not found.']

        idx = self.data_loader.indices[title]
        similarity_scores = list(enumerate(self.data_loader.similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in similarity_scores[1:num_recommendations + 1]]
        
        return [self.data_loader.titles[i] for i in top_indices]
