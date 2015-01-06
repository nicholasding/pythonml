import numpy as np
import mnist

from helper import normalize_features, random_sampling
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib

training = mnist.DataReader('train')
testing = mnist.DataReader('t10k')

data, target = random_sampling(training.data, training.target, 3000)

# Gaussian Kernel, so far the best
classifier = svm.SVC(gamma=0.03)
classifier.fit(normalize_features(data), target.ravel())

joblib.dump(classifier, 'model.pkl')

print 'Training done, modek.pkl saved.'

predicted = classifier.predict(normalize_features(testing.data))
expected = testing.target.ravel()

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
