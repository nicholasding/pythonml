from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

from optparse import OptionParser

import sys
import json

class DataSource(object):
    """
    Attributes:
    - data
    - target
    - target_names
    """
    def __init__(self, json_file):
        self.target_index = 0
        self.target_mapping = {}
        self.target_names = []

        data = []
        target = []

        with open(json_file, 'r') as fp:
            for line in fp:
                i = json.loads(line)
                data.append(i['X'])
                target.append(self.mapping_target(i['y']))
        
        self.data = data
        self.target = target

    def mapping_target(self, label):
        if label not in self.target_mapping:
            self.target_names.append(label)
            self.target_mapping[label] = self.target_index
            self.target_index += 1
        return self.target_mapping[label]


CLASSIFIERS = {
    'svm': LinearSVC(),
    'sgd': SGDClassifier(alpha=.0001, n_iter=50, penalty='l2'),
    'nb': MultinomialNB(alpha=.01),
}


def classifier(name):
    if name in CLASSIFIERS: return CLASSIFIERS[name]
    return CLASSIFIERS['sgd']


def train_and_test(opts):
    data_train = DataSource(opts.training_file)
    data_test = DataSource(opts.testing_file)

    # Train
    if opts.use_hashing:
        vectorizer = HashingVectorizer(stop_words='english', non_negative=True, n_features=2 ** 16)
        X_train = vectorizer.transform(data_train.data)
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        X_train = vectorizer.fit_transform(data_train.data)
    y_train = data_train.target

    print("Training: n_samples: %d, n_features: %d" % X_train.shape)

    clf = classifier(opts.classifier)
    clf.fit(X_train, y_train)

    # Test
    X_test = vectorizer.transform(data_test.data)
    y_test = data_test.target

    print("Testing: n_samples: %d, n_features: %d" % X_test.shape)

    # Metrics
    pred = clf.predict(X_test)
    score = metrics.f1_score(y_test, pred)
    print('F1 score: %.3f' % score)
    print(metrics.classification_report(y_test, pred, target_names=data_train.target_names))


def main():
    op = OptionParser()
    op.add_option('-i', '--training_file', dest='training_file', action='store', help='JSON file for training')
    op.add_option('-t', '--testing_file', dest='testing_file', action='store', help='JSON file for testing')
    op.add_option('-c', '--classifier', dest='classifier', action='store', help='Classifier, options (svm, sgd, nb)')
    op.add_option('--use_hashing', dest='use_hashing', action='store_true', help='Use feature hashing')
    (opts, args) = op.parse_args()

    if not (opts.training_file and opts.testing_file):
        op.print_help()
    else:
        train_and_test(opts)

if __name__ == '__main__':
    main()
