ML Examples done by scikit-learn
================================

Some ML examples based on scikit-learn library.

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
