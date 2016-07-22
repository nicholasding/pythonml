import numpy
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_model():
    model = Sequential()
    model.add(Dense(12, input_dim=6, init='normal', activation='relu'))
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(Dense(50, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

dataframe = pd.read_csv('houseprice.csv')
dataset = dataframe.values

X = dataset[:, 1:].astype(float)
Y = dataset[:, 0]

estimators = (
    ('standardize', StandardScaler()),
    ('mlp', KerasRegressor(build_fn=create_model, nb_epoch=100, batch_size=20, verbose=0)),
)

pipeline = Pipeline(estimators)

seed = 7
kfold = KFold(n=len(X), n_folds=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)

print("%.2f (%.2f) MSE" % (results.mean(), results.std()))

pipeline.fit(X, Y)
print pipeline.predict([[1966, 9, 3, 1, 1, 1]])

