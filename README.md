# Netflix Recommender

This project is a simple Netflix Recommender System that clusters movies and TV shows using Kmeans and Agglomerative Clustering. Also, generated top 10 recommendations using cosine similarity between vectorised features.

It's a project still in progress, for learning the concepts therefore, the end goal is not set. Will be trying various different techniques on it as I learn further.

##
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation
1. clone the repo:
   '''
   git clone <url>
   cd NetflixRecommender
  '''
2. Create a virtual env:
   '''
   python3 -m venv venv
   source venv/bin/activate
   '''
3. Install the required packages:
   '''
   pip install -r requirements.txt
   '''
4. Download NLTK data:
   '''
   import NLTK
   nltk.download('punkt') #Issues on MAC, worked smoothly on JupyterNotebook on windows.
   '''
## Usage
1. Ensure you have the data file available from Kaggle. (Netflix movies and TV show data.)
2. run the main.py #issues with cleaning for me, so had to clean and load the data file separately.

## Features
 - Load and Preprocess Netflix Data (NLTK issues.)
 - Perform TF-Idf and PCA
 - Cluster using Kmeans and Agglomerative Clustering.
 - Visualise using Seaborn and Matplotlib
 - Provide Recommendations using Cosine Similarity.

## Comments:
Old project had to be deleted. 
Newer push, same code, however, issues with NLTK. 
New branch contains more dynamic input for search.
