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
    #plt.locator_params(axis="y", nbins=30)
    plt.show()
# convert your labels in one-hot
t1 = np.zeros(10)
print("mnist size",mnist_images1.size)
#t1[3]=5;
#t5 = np.concatenate((t1, t2), axis=0)
#t6 = np.concatenate((t5, t3), axis=0)
#t7 = np.concatenate((t6, t4), axis=0)
#pad image all around one time with zeros
#show_digit(t1.reshape(1,-1))
#for loop to add one-hot labels to the images as pixels
mnist_labeled = np.zeros(43670000) #create empty array for the new images
mnist_labeled = mnist_labeled.reshape(55000,794) #reshape it (2D)
n_data = mnist_images1.shape[0]
for i in range(n_data-1):
    t1 = np.zeros(10)
    b = mnist_images1[i]#.reshape(28, 28)
    x = np.where(mnist.train.labels[i] == 1)
    #show_digit(mnist_images1[2].reshape(28, 28))

    x1 = list(x) # change to list to add 20 offset
    t1[x1[0]] = 1
    #print("t1 is", t1)
    b = np.concatenate((t1 , mnist_images1[i]), axis=0)
    #x1[0] = x1[0] +18
    #x =tuple(x1) #revert back to tuple
    #print("b size", b.size)
    #print("b dim", b.dimension)
    #x[0] = x[0]+[20]
    #b[x,0] = 1 # add the label coressponding to the image on the right most clolumn [0..10] ,+20 means that we want to have the labels on pixels [20...29]
    mnist_labeled[i] = b#.flatten() #convert the image to 1D and store it
#test to see if corretly added labels
# create figure
fig = plt.figure(figsize=(10, 10))

# setting values to rows and column variables
rows = 2
columns = 1
fig.add_subplot(rows, columns, 2)
plt.imshow(mnist_labeled[5][10:794].reshape(28,28))#new_image.reshape(29,29)
fig.add_subplot(rows, columns, 1)
plt.imshow(mnist_labeled[5][0:10].reshape(1,-1))
plt.locator_params(axis="x", nbins=9)
plt.show()
#for m in range(19):
   # show_digit(mnist_labeled[m].reshape(28, 28))


#create the BM
bbrbm = BBRBM(n_visible=794, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
err = bbrbm.fit(mnist_labeled, n_epoches=100, batch_size=10)
#save the weights
filename = 'weights_class3'
name = 'bbrbm_class3'
bbrbm.save_weights(filename,name)





