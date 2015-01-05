import numpy as np

from PIL import Image
from collections import defaultdict
from sklearn.decomposition import PCA

def resize_image(image_file):
    """
    Convert image to grey scale then find the bounding box.
    Resize it to 20 x 20, center it on 28 x 28 image.
    """
    im = Image.open(image_file).convert('L')
    pixels = im.load()
    width, height = im.size

    left, top, right, bottom = 0, 0, 0, 0
    for x in xrange(width):
        col = [pixels[x, y] for y in xrange(height)]
        if len(set(col)) > 1:
            left = x
            break

    for x in reversed(xrange(width)):
        col = [pixels[x, y] for y in xrange(height)]
        if len(set(col)) > 1:
            right = x
            break

    for y in xrange(height):
        row = [pixels[x, y] for x in xrange(width)]
        if len(set(row)) > 1:
            top = y
            break

    for y in reversed(xrange(height)):
        row = [pixels[x, y] for x in xrange(width)]
        if len(set(row)) > 1:
            bottom = y
            break

    # print left, top, right, bottom
    # max_w = max((left, top, right, bottom))
    new_width, new_height = (right - left), (bottom - top)
    if new_width > new_height:
        # reset height
        top -= (new_width - new_height) / 2
        bottom += (new_width - new_height) / 2
    if new_width < new_height:
        # reset width
        left -= (new_height - new_width) / 2
        right += (new_height - new_width) / 2

    cropped_im = im.crop((left, top, right, bottom)).resize((20, 20), Image.ANTIALIAS)
    new_im = Image.new('L', (28, 28), 'white')
    new_im.paste(cropped_im, (4, 4))
    return new_im

def normalize_features(data):
    return data / 255

def random_sampling(data, target, limit_per_sample):
    new_data = np.zeros((10 * limit_per_sample, data.shape[1]))
    new_target = np.zeros((10 * limit_per_sample, 1))
    num_feature = data.shape[1]
    count = defaultdict(int)

    for row in np.hstack((data, target)):
        label = row[-1]
        if count[label] > (limit_per_sample - 1):
            continue
        else:
            idx = label * limit_per_sample + count[label]
            new_data[idx] = row[:-1]
            new_target[idx] = row[-1]
            count[label] += 1

    return new_data, new_target

def reduce_features(data, num_features):
    pca = PCA(n_components=num_features)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca
