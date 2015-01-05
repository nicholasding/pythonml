import numpy as np
import mnist

from helper import normalize_features, random_sampling
from sklearn import datasets, svm, metrics
from sklearn.externals import joblib

training = mnist.DataReader('train')
testing = mnist.DataReader('t10k')

data, target = random_sampling(training.data, training.target, 100)

# Linear SVC
classifier = svm.LinearSVC()

# Gaussian Kernel
# classifier = svm.SVC(gamma=0.03)

# Polynomial Kernel, d = 4
# classifier = svm.SVC(kernel='poly', degree=4)
classifier.fit(normalize_features(data), target.ravel())

joblib.dump(classifier, 'model.pkl')

print 'Training done, modek.pkl saved.'

predicted = classifier.predict(testing.data)
expected = testing.target

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
