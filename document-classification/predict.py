from optparse import OptionParser
from sklearn.externals import joblib

import sys
import json


def predict(opts):
    vectorizer, clf, labels = joblib.load(opts.model_file)
    fp = open(opts.input_file, 'r')
    data = []
    for line in fp:
        i = json.loads(line)
        data.append(i['X'])
    fp.close()
    X = vectorizer.transform(data)
    for X, y in zip(data, clf.predict(X)):
        print json.dumps({'X': X, 'y': labels[y]})


def main():
    op = OptionParser()
    op.add_option('-m', '--model_file', dest='model_file', action='store', help='Load trained model')
    op.add_option('-i', '--input_file', dest='input_file', action='store', help='JSON file for predicting')
    (opts, args) = op.parse_args()

    if not (opts.model_file and opts.input_file):
        op.print_help()
    else:
        predict(opts)


if __name__ == '__main__':
    main()
