import web
import re
import StringIO
import numpy as np
import base64

from helper import resize_image, normalize_features
from PIL import Image
from sklearn.externals import joblib

classifier = joblib.load('model.pkl')

urls = (
    '/image', 'ImageClassification'
)

class ImageClassification(object):

    def POST(self):
        i = web.input()
        uri = i['img']
        data_to_64 = re.search(r'base64,(.*)', uri).group(1)
        data = StringIO.StringIO()
        # data.write(base64.decodestring(data_to_64))
        data.write(data_to_64.decode('base64'))
        data.seek(0)
        im = resize_image(data)
        X = 255 - np.asarray(im).flatten()
        print X
        label = classifier.predict(normalize_features(X))
        print label
        return label[0]

if __name__ == '__main__':
    app = web.application(urls, globals())
    app.run()
