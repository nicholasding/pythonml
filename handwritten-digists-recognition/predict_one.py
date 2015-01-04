from sklearn.externals import joblib
from find_boundary import load_pic

import numpy as np

im = load_pic()
X = np.asarray(im).flatten()

classifier = joblib.load('model.pkl')
label = classifier.predict(X)

print label