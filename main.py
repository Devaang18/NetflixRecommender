import pandas as pd
from clustering import trial
from data_loader import DataLoader
from recommender import Recommender
from ui import launch_app

df = trial()
#df = pd.read_csv('/Users/devaang/Documents/cluster_data.csv')

data_loader = DataLoader(df)
recommender = Recommender(data_loader)

launch_app(recommender)

