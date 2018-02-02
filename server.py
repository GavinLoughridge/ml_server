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
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import hdbscan
from sklearn.datasets import make_blobs

# set run mode
network = 'original'
preprocessing = '5dhdb'
evaluate = False

# set up tensorflow variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# load model
saver = tf.train.Saver()
directory = os.path.dirname(os.path.abspath(__file__))
if network == 'original':
    rel_path = 'model/digit_model.ckpt'
elif network == 'convolution':
    rel_path = 'model/digit_model_conv.ckpt'
filename = os.path.join(directory, rel_path)
saver.restore(sess, filename)

# start flask app
app = Flask(__name__, template_folder='./static')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def classify():
    # load image
    image_data = json.loads(request.data)['data']
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')

    directory = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(directory, '0-origin.jpg')
    image.save(outfile, "JPEG")

    # reduce image resolution
    (width, height) = image.size
    image = image.resize((int(math.floor(width/10)), int(math.floor(height/10))))

    # rotate image if nessecary
    if width > height:
        image = image.rotate(270, expand=True)

        directory = os.path.dirname(os.path.abspath(__file__))
        outfile = os.path.join(directory, '2-rotated.jpg')
        image.save(outfile, "JPEG")

    # set up and run DBSCAN on 5d pixles
    if preprocessing == '5ddb':
        array = np.array(image, dtype=np.float)
        (rows_orig, cols_orig, three) = array.shape
        cord_array = np.zeros((rows_orig, cols_orig, 5))
        for i in range(0,rows_orig):
            for j in range(0,cols_orig):
                cord_array[i][j][0] = array[i][j][0]
                cord_array[i][j][1] = array[i][j][1]
                cord_array[i][j][2] = array[i][j][2]
                cord_array[i][j][3] = i
                cord_array[i][j][4] = j
        cord_array = cord_array.reshape(-1, cord_array.shape[-1])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
        cluster_labels = clusterer.fit_predict(cord_array)

        print('max label is:', np.amax(cluster_labels))
        cluster_labels = cluster_labels.reshape((rows_orig, cols_orig))

        final_array = cluster_labels

    # set up and run HDBSCAN on 5d pixles
    if preprocessing == '5dhdb':
        array = np.array(image, dtype=np.float)
        (rows_orig, cols_orig, three) = array.shape
        cord_array = np.zeros((rows_orig, cols_orig, 5))
        for i in range(0,rows_orig):
            for j in range(0,cols_orig):
                cord_array[i][j][0] = array[i][j][0]
                cord_array[i][j][1] = array[i][j][1]
                cord_array[i][j][2] = array[i][j][2]
                cord_array[i][j][3] = i
                cord_array[i][j][4] = j
        cord_array = cord_array.reshape(-1, cord_array.shape[-1])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
        cluster_labels = clusterer.fit_predict(cord_array)

        print('max label is:', np.amax(cluster_labels))
        cluster_labels = cluster_labels.reshape((rows_orig, cols_orig))

        final_array = cluster_labels

    if preprocessing == '3dhdb':
        array = np.array(image, dtype=np.float)
        (rows_orig, cols_orig, three) = array.shape
        array = array.reshape(-1, array.shape[-1])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=100)
        cluster_labels = clusterer.fit_predict(array)

        print('max label is:', np.amax(cluster_labels))
        cluster_labels = cluster_labels.reshape((rows_orig, cols_orig))

        final_array = cluster_labels

    if preprocessing == 'kmeans':
        array = np.array(image, dtype=np.float)
        (rows_orig, cols_orig, three) = array.shape
        array = array.reshape(-1, array.shape[-1])

        k = 3
        clt = KMeans(n_clusters=3)
        clt.fit(array)

        print('labels:', clt.labels_)
        print('centroids:', clt.cluster_centers_)

        centroids = clt.cluster_centers_
        minimums = [np.amin(centroids[0]), np.amin(centroids[1]), np.amin(centroids[2])]
        minimum = np.amin(minimums)
        label = minimums.index(minimum)
        print('number is centroid:', label)

        def filter_labels(pixel):
            return 1 if pixel == label else 0
        filter_labels = np.vectorize(filter_labels, otypes=[np.float])
        filtered_labels = filter_labels(clt.labels_)

        filtered_array = filtered_labels.reshape((rows_orig, cols_orig))

        def contrast(pixel):
            return 1 if pixel > 0 else 0
        contrast = np.vectorize(contrast, otypes=[np.float])
        contrast_array = contrast(filtered_array)

        final_array = contrast_array

    # color0_rgb = sRGBColor(centroids[0][0],centroids[0][1],centroids[0][2])
    # color0_lab = convert_color(color0_rgb, LabColor)
    # color1_rgb = sRGBColor(centroids[0][1],centroids[1][1],centroids[1][2])
    # color1_lab = convert_color(color1_rgb, LabColor)
    # color2_rgb = sRGBColor(centroids[0][2],centroids[2][1],centroids[2][2])
    # color2_lab = convert_color(color2_rgb, LabColor)
    # delta_01 = delta_e_cie2000(color0_lab, color1_lab)
    # delta_02 = delta_e_cie2000(color0_lab, color2_lab)
    # delta_12 = delta_e_cie2000(color1_lab, color2_lab)

    if preprocessing == 'manual':
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(2.0)

        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '1-bright.jpg')
        # image.save(outfile, "JPEG")

        r, g, b = image.split()
        r = r.point(lambda p: p > 220 and 255)
        g = g.point(lambda p: p > 220 and 255)
        b = b.point(lambda p: p > 200 and 255)
        image = Image.merge("RGB", (r, g, b))

        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '2-clipped.jpg')
        # image.save(outfile, "JPEG")

        image = image.filter(ImageFilter.GaussianBlur(5))

        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '3-blur.jpg')
        # image.save(outfile, "JPEG")

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(4.0)

        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '4-contrast.jpg')
        # image.save(outfile, "JPEG")

        (width, height) = image.size
        if width > height:
            image = image.rotate(270, expand=True)

            # directory = os.path.dirname(os.path.abspath(__file__))
            # outfile = os.path.join(directory, '4-rotated.jpg')
            # image.save(outfile, "JPEG")

        image.thumbnail((28, 28), Image.ANTIALIAS)
        image = image.convert('L')

        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '5-thumb.jpg')
        # image.save(outfile, "JPEG")

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(4.0)

        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '6-contrast2.jpg')
        # image.save(outfile, "JPEG")

        array = np.array(image, dtype=np.float)

        average = np.mean(array)
        amax = np.amax(array)
        amin = np.amin(array)
        def format_max(pixle, threshold):
            return 1 if pixle < threshold else 0
        format_max = np.vectorize(format_max, otypes=[np.float])
        formatted_array = format_max(array, average)

        # image = Image.fromarray(np.uint8(formatted_array * 255), 'L')
        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '7-vect_max.jpg')
        # image.save(outfile, "JPEG")

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

        # image = Image.fromarray(np.uint8(formatted_array * 255), 'L')
        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '8-shadowless.jpg')
        # image.save(outfile, "JPEG")

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
        image_inner = image_inner.resize((int(width*ratio), int(height*ratio)))
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

        # image = Image.fromarray(np.uint8(centered_array * 255), 'L')
        # directory = os.path.dirname(os.path.abspath(__file__))
        # outfile = os.path.join(directory, '9-centered.jpg')
        # image.save(outfile, "JPEG")

        final_array = centered_array

    myGuess = 'N'
    if evaluate == True:
        guess = tf.argmax(y,1)
        myGuess = guess.eval(feed_dict={x: centered_array.reshape(1, 784)})[0]

    img = Image.fromarray(np.uint8(final_array * 255), 'L')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    content = base64.b64encode(buffered.getvalue())

    return Response(json.dumps({
        'guess': myGuess,
        'image': content
    }), mimetype='application/json')
