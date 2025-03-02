import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def cleaning_data(data):
    data.fillna('', inplace=True)
    feat_for_clusters = ['director', 'cast', 'country', 'listed_in', 'description', 'rating']
    rating_map = {'TV-MA':'Adults', 'R':'Adults', 'PG-13':'Teens', 'TV-14':'Young Adults', 'TV-PG':'Older Kids',
                  'NR':'Adults', 'TV-G':'Kids', 'TV-Y':'Kids', 'TV-Y7':'Older Kids', 'PG':'Older Kids', 'G':'Kids', 'NC-17':'Adults',
                  'TV-Y7-FV':'Older Kids', 'UR':'Adults'}
   
    data['country'] = data['country'].apply(lambda x: x.split(',')[0])
    data['listed_in'] = data['listed_in'].apply(lambda x: x.split(',')[0])
    data['duration'] = data['duration'].apply(lambda x: int(x.split()[0]))
    data.replace({'rating': rating_map}, inplace=True)
    data['clustered_features']= data[feat_for_clusters].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    data['preprocessed_clustered_features'] = data['clustered_features'].apply(preprocess_data)
    return data

def preprocess_data(text):
    text = re.sub(r'[^\x00-\x7F]+', '', text) # remove non-ASCII
    text = text.lower() # to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text) # remove punctuations
    tokens = word_tokenize(text) # tokenising the text
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # removing stopwords and lemmatise the tokens
   
    return ' '.join(cleaned_tokens) # joining the tokens back into a string
