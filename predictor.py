import pandas
import numpy as np
import re

chunksize = 10 ** 5
train_set = pandas.read_csv("./dataset/train.csv",nrows=10 ** 6)
test_set = pandas.read_csv("./dataset/test.csv")

print(train_set.info())
print(train_set.describe())
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

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
        return np.c_[X]

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
        return np.c_[X]

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
pipeline_num = Pipeline([
    ("selector",DataFrameSelector(["pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count"])),
    ("distance_creator",CreateDistance()),
    ("imputer", Imputer(strategy="mean")),
    ("scaler", StandardScaler())
])

pipeline_cat = Pipeline([
    ("selector", DataFrameSelector(["pickup_datetime"])),
    ("time_category_creator",CategorizePickupTime()),
    ("HotEncoder",OneHotEncoder(sparse=False))
])

final_pipeline = FeatureUnion([
("cat_pipelone",pipeline_cat),
    ("num_pipeline", pipeline_num)

])

y_train = train_set.drop(["key","pickup_datetime","pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude","passenger_count"],axis=1)
X_train = train_set.drop(["key","fare_amount"],axis=1)
X_train = train_set.append(test_set.drop("key",axis=1))

X_train = final_pipeline.fit_transform(X_train)


X_test = X_train[chunksize:]
X_train = X_train[:chunksize]

