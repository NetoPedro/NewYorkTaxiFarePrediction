import pandas
import numpy as np
import re
import multiprocessing

chunksize = (10 ** 6) * 2
train_set = pandas.read_csv("./dataset/train.csv",nrows=chunksize)
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
        X.loc[X["passenger_count"] < 5, "passenger_count"] = 0
        X.loc[X["passenger_count"] >= 5, "passenger_count"] = 1

        return np.c_[X]

class CategorizePickupTime(BaseEstimator,TransformerMixin):
    def __init__(self):pass

    def fit(self, X, y= None): return self

    def transform(self, X, y= None):
        X["weekday"] = pandas.to_datetime(X['pickup_datetime'].str.replace(" UTC", ""), format='%Y-%m-%d %H:%M:%S').dt.weekday
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

from sklearn.svm  import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

'''grid_params = [{"C":[0.2,0.4,0.6,0.8,1,1.2,1.4],"epsilon":[0.01,0.03,0.05,0.07,0.1,0.12,0.14],"kernel":["linear","poly","rbf","sigmoid"]}]
svm = SVR()
grid_svm = GridSearchCV(svm,param_grid=grid_params,scoring="neg_mean_squared_error",verbose=3,cv=3)

grid_svm.fit(X_train,y_train)

print(grid_svm.best_estimator_)
print(grid_svm.best_score_)

y_pred = grid_svm.predict(X_test)
'''
grid_params = {'solver': ['lbfgs'], 'max_iter': [100000], 'alpha': [0.001],"learning_rate":["constant"],"activation":["relu"], 'hidden_layer_sizes':np.arange(23, 25), 'random_state':[42]}
nn = MLPRegressor()
grid_nn = GridSearchCV(nn, param_grid=grid_params, scoring="neg_mean_squared_error", verbose=3, cv=3, n_jobs=-1)

grid_nn.fit(X_train, y_train.values.ravel())

print(grid_nn.best_estimator_)
print(grid_nn.best_score_)

y_pred = grid_nn.predict(X_test)


submissions = pandas.DataFrame(y_pred, index=test_set.key,columns=["fare_amount"])
submissions.to_csv('./submission.csv', index=True)