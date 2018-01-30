from flask import Flask
from flask import render_template
from flask import request
from flask import Response
import tensorflow as tf
import numpy as np
import os, sys
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from io import BytesIO
import base64
import json
import math
from sklearn import cluster

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
    image_data = json.loads(request.data)['data']
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '0-original.jpg')
    image.save(outfile, "JPEG")

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(2.0)

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '1-bright.jpg')
    image.save(outfile, "JPEG")

    r, g, b = image.split()
    r = r.point(lambda p: p > 220 and 255)
    g = g.point(lambda p: p > 220 and 255)
    b = b.point(lambda p: p > 200 and 255)
    image = Image.merge("RGB", (r, g, b))

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '2-clipped.jpg')
    image.save(outfile, "JPEG")

    image = image.filter(ImageFilter.GaussianBlur(5))

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '3-blur.jpg')
    image.save(outfile, "JPEG")

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(4.0)

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '4-contrast.jpg')
    image.save(outfile, "JPEG")

    (width, height) = image.size
    if width > height:
        mage = image.rotate(270)

    image.thumbnail((28, 28), Image.ANTIALIAS)
    image = image.convert('L')
    array = np.array(image, dtype=np.float)

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '5-final.jpg')
    image.save(outfile, "JPEG")

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(4.0)

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '6-contrast.jpg')
    image.save(outfile, "JPEG")

    average = np.mean(array)
    amax = np.amax(array)
    amin = np.amin(array)
    def format_max(pixle, threshold):
        return 1 if pixle < threshold else 0
    format_max = np.vectorize(format_max, otypes=[np.float])
    formatted_array = format_max(array, average)

    (rows, cols) = array.shape
    shadows = []
    for i in range (0, cols):
        if formatted_array.item((0, i)) > 0:
            shadows.append((0, i))
        if formatted_array.item((rows - 1, i)) > 0:
            shadows.append((rows - 1, i))

    for i in range (0, rows):
        if formatted_array.item((i, 0)) > 0:
            shadows.append((i, 0))
        if formatted_array.item((i, cols - 1)) > 0:
            shadows.append((i, cols - 1))

    loopGuard = 0
    while shadows:
        (i, j) = shadows.pop(0)
        formatted_array.itemset((i,j), 0)
        if i > 0 and array.item(i-1,j) > 0:
            shadows.append((i-1, j))
        if i < rows-1 and array.item(i+1,j) > 0:
            shadows.append((i+1,j))
        if j > 0 and array.item((i,j-1)) > 0:
            shadows.append((i,j-1))
        if j < cols-1 and array.item((i,j+1)) > 0:
            shadows.append((i,j+1))
        loopGuard += 1
        if loopGuard > 800:
            print('in a loop')
            break

    (nonzero_row,nonzero_col) = np.nonzero(formatted_array)
    min_row = np.amin(nonzero_row)
    max_row = np.amax(nonzero_row)
    min_col = np.amin(nonzero_col)
    max_col = np.amax(nonzero_col)

    nonzero_array = np.zeros((max_row+1-min_row,max_col+1-min_col))

    nonzero_pixles = nonzero_row.shape
    for i in range(0,nonzero_pixles[0]):
        row = int(nonzero_row[i])
        col = int(nonzero_col[i])
        value = formatted_array.item((row,col))
        nonzero_array.itemset((row-min_row,col-min_col), value)

    image_inner = Image.fromarray(np.uint8(nonzero_array * 255), 'L')
    (width, height) = image_inner.size
    if width > height:
        ratio = float(20)/width
    else:
        ratio = float(20)/height
    image_inner = image_inner.resize((int(width*ratio), int(height*ratio), Image.LANCZOS))
    image_inner = image_inner.convert('L')
    inner_array = np.array(image_inner, dtype=np.float)

    average = np.mean(inner_array)
    amax = np.amax(inner_array)
    amin = np.amin(inner_array)
    def format_min(pixle, threshold):
        return 1 if pixle > threshold else 0
    format_min = np.vectorize(format_min, otypes=[np.float])
    inner_array = format_min(inner_array, average)

    (inner_row,inner_col) = np.nonzero(inner_array)
    max_row = np.amax(inner_row)
    max_col = np.amax(inner_col)
    min_row = np.amin(inner_row)
    min_col = np.amin(inner_col)
    shift_row = int(math.floor(28-(max_row-min_row))/2)
    shift_col = int(math.floor(28-(max_col-min_col))/2)

    centered_array = np.zeros((28,28))
    inner_pixles = inner_row.shape
    for i in range(0,inner_pixles[0]):
        row = int(inner_row[i])
        col = int(inner_col[i])
        value = inner_array.item((row,col))
        centered_array.itemset((row+shift_row,col+shift_col), value)

    guess = tf.argmax(y,1)
    myGuess = guess.eval(feed_dict={x: centered_array.reshape(1, 784)})[0]

    img = Image.fromarray(np.uint8(centered_array * 255), 'L')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    content = base64.b64encode(buffered.getvalue())

    return Response(json.dumps({
        'guess': myGuess,
        'image': content
    }), mimetype='application/json')
