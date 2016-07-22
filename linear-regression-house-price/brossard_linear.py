import numpy
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv('houseprice.csv')
dataset = dataframe.values

X = dataset[:, 1:].astype(float)
Y = dataset[:, 0]

estimators = (
    ('standardize', StandardScaler()),
    ('linear', LinearRegression()),
)

pipeline = Pipeline(estimators)

seed = 7
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("%.2f (%.2f) MSE" % (results.mean(), results.std()))

pipeline.fit(X, Y)
print pipeline.predict([[1966, 9, 3, 1, 1, 1]])

