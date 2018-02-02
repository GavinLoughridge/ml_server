# ml_server

this is a program that accepts a picture of a handwritten digit and returns a classification for the digit.

in it's current form it recognizes handwritten digits from pictures, which you can try out on heroku here:
digitreader.herokuapp.com
it grew out some experimental code available here: https://github.com/GavinLoughridge/ml_project

the neural network was made using tensorflow and the MNIST dataset.

I used Vue to render the single page app, which made it easy to update the view as the app receives data from users and
the server.

from the web app a user can upload a picture which is then sent to the server. the sever is written in python and uses flask to serve up the web app and also to respond to images that get sent to it.

images received by the server goes through a preprocessing pipeline to enhance, center, and resize the image so that it better matches the training data from MNIST. I use the python image library PILLOW to manipulate the image directly and I used numpy to manipulate the image as an array.

the processed array is classified by the neural network
and the networks 'guess' along with a copy of the processed image is sent back to the web app, which updates the view to display both.

I'm in the process of improving the neural network by switching to a convolutional neural network, which should preform better with this kind of image recognition task.

I'm also changing the preprocessing pipeline to utilize clustering algorithms so that it is more flexible when confronted with unexpected light or color variance in the original image.

KMEANS showed some promising results at first but requires an estimate of how many color clusters are expected. Now I'm using HDBSCAN which is designed to be more flexible when the number of clusters is unknown.
