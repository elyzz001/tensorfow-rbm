#Hi there, Can you see this
import numpy as np
import tensorflow as tf
from numpy import load
import operator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) #this line shold always stay above the next line below
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.train.images
mnist_images1= np.where(mnist_images > 0, 1, 0)


def pad_with(vector, pad_width, iaxis, kwargs): #https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def show_digit(x):
    plt.imshow(x)#,cmap = plt.cm.binary)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title("Original Image" )
    #plt.locator_params(axis="x", nbins=30)
    plt.locator_params(axis="y", nbins=30)
    plt.show()
# convert your labels in one-hot

#pad image all around one time with zeros

#for loop to add one-hot labels to the images as pixels
mnist_labeled = np.zeros(18000) #create empty array for the new images
mnist_labeled = mnist_labeled.reshape(20,900) #reshape it (2D)
for i in range(19):
    b = np.pad(mnist_images1[i].reshape(28, 28), 1, pad_with, padder=0)#add contour of zeros around the image
    x = np.where(mnist.train.labels[i] == 1)
    #show_digit(mnist_images1[2].reshape(28, 28))
    x1 = list(x) # change to list to add 20 offset
    x1[0] = x1[0] +20
    x =tuple(x1) #revert back to tuple
   # print("x is", x[0])
   # x[0] = x[0]+[20]
    b[x,0] = 1 # add the label coressponding to the image on the right most clolumn [0..10] ,+20 means that we want to have the labels on pixels [20...29]
    mnist_labeled[i] = b.flatten() #convert the image to 1D and store it 
#test to see if corretly added labels
show_digit(mnist_labeled[5].reshape(30,30))#new_image.reshape(29,29)

for m in range(19):
    show_digit(mnist_labeled[20].reshape(30, 30))


#create the BM
bbrbm = BBRBM(n_visible=900, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
err = bbrbm.fit(mnist_labeled, n_epoches=100000, batch_size=10)
#save the weights
filename = 'weights_class'
name = 'bbrbm_class'
bbrbm.save_weights(filename,name)





