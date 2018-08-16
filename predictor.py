import pandas
import numpy as np
import re

chunksize = 10 ** 5
train_set = pandas.read_csv("./dataset/train.csv",nrows=10 ** 6)
test_set = pandas.read_csv("./dataset/test.csv")

print(train_set.info())
print(train_set.describe())
from sklearn.base import BaseEstimator, TransformerMixin

class CreateDistance(BaseEstimator,TransformerMixin):

    def __init__(self):pass

    def fit(self,X, y= None):
        return self

    def transform(self,X,y=None):
        from math import sin, cos, sqrt, atan2, radians

        # approximate radius of earth in km
        R = 6373.0


        dlon = X["dropoff_longitude"].values - X["pickup_longitude"].values
        dlat = X["dropoff_latitude"].values - X["pickup_latitude"].values

        a = np.sin(dlat / 2) ** 2 + np.multiply(np.cos(X["pickup_latitude"].values), np.cos(X["dropoff_latitude"].values)) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        X["distance"] = R * c
        X = X.drop(["pickup_latitude","pickup_longitude","dropoff_longitude","dropoff_latitude"],axis=1)
        return X

class CategorizePickupTime(BaseEstimator,TransformerMixin):
    def __init__(self):pass

    def fit(self, X, y= None): return self

    def transform(self, X, y= None):
        X["pickup_datetime"] = (X["pickup_datetime"].str.extract('( ..)', expand=True).astype(int))
        by_hour = X["pickup_datetime"].value_counts()

        mean = np.mean(by_hour)
        std = np.std(by_hour)

        by_hour = X["pickup_datetime"].value_counts().__str__()
        by_hour = by_hour.split("\n")
        i = 0
        for string in by_hour:
            by_hour[i] = (re.findall(r'\S+', by_hour[i]))
            i += 1

        by_hour = np.transpose(by_hour)

        for entry in by_hour:
            if len(entry) != 2: continue
            if int(entry[1]) < mean - std:
                X.loc[X["pickup_datetime"] == int(entry[0]), "pickup_datetime"] = -1
            elif int(entry[1]) > mean + std:
                X.loc[X["pickup_datetime"] == int(entry[0]), "pickup_datetime"] = -3
            else:
                X.loc[X["pickup_datetime"] == int(entry[0]), "pickup_datetime"] = -2

        X["pickup_datetime"] = X["pickup_datetime"] * -1
        return X

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
pipeline_num = Pipeline([
    ("imputer", Imputer(strategy="mean")),
    ("distance_creator",CreateDistance()),
    ("scaler", StandardScaler())
])

pipeline_cat = Pipeline([
    ("time_category_creator",CategorizePickupTime()),
    ("HotEncoder",OneHotEncoder())
])

final_pipeline = FeatureUnion([
    ("num_pipeline", pipeline_num),
    ("cat_pipelone",pipeline_cat)
])



'''
train_set["pickup_datetime"] = (train_set["pickup_datetime"].str.extract('( ..)', expand=True).astype(int))
by_hour = train_set["pickup_datetime"].value_counts()

mean = np.mean(by_hour)
std = np.std(by_hour)

by_hour = train_set["pickup_datetime"].value_counts().__str__()
by_hour = by_hour.split("\n")
i = 0
for string in by_hour:
    by_hour[i]= (re.findall(r'\S+', by_hour[i]))
    i +=1

by_hour = np.transpose(by_hour)

for entry in by_hour:
    if len(entry)!=2: continue
    if int(entry[1]) < mean - std:
        train_set.loc[train_set["pickup_datetime"] == int(entry[0]),"pickup_datetime"] = -1
    elif int(entry[1]) > mean + std:
        train_set.loc[train_set["pickup_datetime"] == int(entry[0]),"pickup_datetime"] = -3
    else: train_set.loc[train_set["pickup_datetime"] == int(entry[0]),"pickup_datetime"] = -2


train_set["pickup_datetime"] = train_set["pickup_datetime"] * -1
'''
# TODO move to class
# TODO Pipeline
