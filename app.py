import base64
import re
import io
import pickle

import matplotlib.image as mpimg
import scipy.misc
import numpy as np
from scipy import ndimage
from skimage import img_as_bool, img_as_uint
from skimage.util import invert
from skimage.color import rgb2gray
from sklearn.neighbors import KNeighborsClassifier

from flask import Flask
from flask import render_template
from flask import request

from extract_objects import get_object_boudaries
from mnist_loader import load_data


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.1.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    # get image data from POST request
    datauri = request.form['imgBase64']
    # remove base64 header
    imgstr = re.search(r'base64,(.*)', datauri).group(1)
    # decode to get binary data
    image_data = imgstr.decode('base64')

    # read binary data as image
    img = io.BytesIO(image_data)
    img = mpimg.imread(img, format='PNG')

    """        
    # convert image to grayscale
    img = img_as_uint(~img_as_bool(rgb2gray(img)))

    # find bounding box of the number in the image
    bound = get_object_boudaries(img)
    # slice out the number from the image
    number_object = img[bound[0]]
    """
    img = invert(rgb2gray(img))
    # rescale image to correct size and unfold the 2D array
    # to get data of the correct format (787 long 1D array)
    data = scipy.misc.imresize(img, (28, 28)).ravel()
    
    # make prediction with model
    prediction = knn.predict([data])
    
    return str(prediction[0]), 201


if __name__ == '__main__':
    knn = KNeighborsClassifier()
    training_data, validation_data, test_data = load_data('data/mnist.pkl.gz')

    knn.fit(test_data[0], test_data[1])

    app.run(host='0.0.0.0', debug=True)

