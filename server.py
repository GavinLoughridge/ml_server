from flask import Flask
from flask import render_template
from flask import request
from flask import Response
import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image
from io import BytesIO
import base64
import json
import math

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
saver = tf.train.Saver()
directory = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(directory, 'model/digit_model.ckpt')
saver.restore(sess, filename)

app = Flask(__name__, template_folder='./static')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def classify():
    size = 28, 28
    image = Image.open(BytesIO(base64.b64decode(json.loads(request.data)['data']))).convert('L')

    (width, height) = image.size

    print("width:", width)
    print("height:", height)

    if width > height:
        print('cropping width')
        left = math.floor(0 + (width - height) / 2)
        right = math.floor(width - (width - height) / 2)
        upper = 0
        lower = height
        box = (left, upper, right, lower)
        image = image.crop(box)

    if width < height:
        print('cropping height')
        left = 0
        right = width
        upper = math.floor(0 + (height - width) / 2)
        lower = math.floor(height - (height - width) / 2)
        box = (left, upper, right, lower)
        image = image.crop(box)

    image.save('cropped.jpg', "JPEG")

    image.thumbnail(size, Image.ANTIALIAS)

    image.save('resized.jpg', "JPEG")
    array = np.array(image, dtype=np.float)

    average = np.mean(array)
    def format(pixle):
        return ((pixle * -1) + 255) / 255 if pixle < (average / 2) else 0
    format = np.vectorize(format, otypes=[np.float])
    formatted_array = format(array)

    guess = tf.argmax(y,1)
    myGuess = guess.eval(feed_dict={x: formatted_array.reshape(1, 784)})[0]

    return Response(json.dumps({
        'guess': myGuess
    }), mimetype='application/json')
