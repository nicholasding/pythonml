import numpy as np
import mnist

from sklearn import datasets, svm, metrics
from sklearn.externals import joblib

training = mnist.DataReader('train')
testing = mnist.DataReader('t10k')

classifier = svm.LinearSVC()
classifier.fit(training.data, training.target)

joblib.dump(classifier, 'model.pkl')

print 'Training done, modek.pkl saved.'

predicted = classifier.predict(testing.data)
expected = testing.target

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
