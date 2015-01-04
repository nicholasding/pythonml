from sklearn.externals import joblib
from helper import resize_image

import numpy as np

im = resize_image('number5.png')
X = np.asarray(im).flatten()

classifier = joblib.load('model.pkl')
label = classifier.predict(X)

print label