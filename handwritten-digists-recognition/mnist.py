import os
import gzip
import struct
import numpy as np

data_folder = os.path.join(os.path.dirname(__file__), 'data')

class DataReader(object):

    def __init__(self, filetype='train'):
        self.image_file = '%s-images-idx3-ubyte.gz' % filetype
        self.label_file = '%s-labels-idx1-ubyte.gz' % filetype

        self.load()

    def load(self):
        # Load images
        f = gzip.open(os.path.join(data_folder, self.image_file), 'rb')
        magic_number, num_images, num_rows, num_columns = struct.unpack('>iiii', f.read(16))
        block_size = num_rows * num_columns
        data = np.zeros((num_images, block_size))
        for i in xrange(num_images):
            data[i] = np.array(struct.unpack('B' * block_size, f.read(block_size)), dtype=np.uint8)
        f.close()

        # print magic_number, num_images, num_rows, num_columns
        # print data
        self.data = data

        # Load labels
        f = gzip.open(os.path.join(data_folder, self.label_file), 'rb')
        magic_number, num_items = struct.unpack('>ii', f.read(8))
        target = np.zeros((num_items, 1))
        for i in xrange(num_items):
            target[i] = struct.unpack('B', f.read(1))
        f.close()

        self.target = np.ravel(target)

# training = DataReader()
testing = DataReader('t10k')

# print testing.target