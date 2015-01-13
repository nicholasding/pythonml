Machine Learning Projects
=========================

Personal machine learning projects for fun.

document-classification
-----------------------

<pre>
% python test.py -i training.json -t testing.json -c svm

Training: n_samples: 4500, n_features: 23061
Testing: n_samples: 1500, n_features: 23061
F1 score: 0.936
                                 precision    recall  f1-score   support

                         Gaming       0.98      0.97      0.97       100
            Computers & Tablets       0.91      0.85      0.88       100
                     Appliances       0.95      0.98      0.97       100
                 Movies & Music       0.95      0.98      0.97       100
      Toys, Sports & Recreation       0.92      0.90      0.91       100
                    Baby & Kids       0.95      0.96      0.96       100
      Cell Phones & Accessories       0.91      0.93      0.92       100
      Car, GPS & Satelite Radio       0.96      0.96      0.96       100
              TV & Home Theatre       0.87      0.92      0.89       100
Musical Instruments & Equipment       0.96      0.97      0.97       100
    iPod, Headphones & Speakers       0.88      0.86      0.87       100
        Health, Beauty & Travel       0.91      0.93      0.92       100
           Cameras & Camcorders       0.97      0.99      0.98       100
          Ink & Office Supplies       1.00      0.99      0.99       100
           Home, Tools & Garden       0.91      0.85      0.88       100

                    avg / total       0.94      0.94      0.94      1500
</pre>

handwritten-digists-recognition
-------------------------------

<pre>
Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.03,
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False):
             precision    recall  f1-score   support

        0.0       0.97      0.99      0.98       980
        1.0       0.98      0.99      0.99      1135
        2.0       0.95      0.97      0.96      1032
        3.0       0.96      0.96      0.96      1010
        4.0       0.96      0.97      0.97       982
        5.0       0.97      0.96      0.97       892
        6.0       0.98      0.98      0.98       958
        7.0       0.97      0.95      0.96      1028
        8.0       0.95      0.96      0.96       974
        9.0       0.97      0.94      0.95      1009

avg / total       0.97      0.97      0.97     10000


Confusion matrix:
[[ 968    0    3    0    0    2    4    1    2    0]
 [   0 1122    3    2    0    1    3    1    3    0]
 [   6    0 1000    3    1    0    4   10    8    0]
 [   0    0    9  974    0    8    0    9    9    1]
 [   1    0    3    0  954    0    7    1    2   14]
 [   3    1    3   11    2  857    6    1    7    1]
 [   7    3    1    0    4    3  938    0    2    0]
 [   0   10   24    3    7    0    0  972    2   10]
 [   5    0    2   10    5    6    0    3  938    5]
 [   6    5    2    8   20    3    0    8   10  947]]
</pre>