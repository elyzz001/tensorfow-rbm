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
fname = ["weights_class3kep","weights_class4kep","weights_class5kep","weights_class6kep","weights_class7kep"
    ,"weights_class8kep","weights_class9kep","weights_class10kep","weights_class11kep","weights_class12kep"
    ,"weights_class13kep","weights_class14kep","weights_class15kep","weights_class16kep",
         "weights_class17kep","weights_class18kep","weights_class19kep","weights_class20kep"]#,"5-3","6-3","7-3","8-3","9-3","10-3","11-3","12-3","13-3","14-3","15-3","16-3","17-3","18-3","19-3","20-3"]
name = ["bbrbm_class3kep","bbrbm_class4kep","bbrbm_class5kep","bbrbm_class6kep","bbrbm_class7kep"
    ,"bbrbm_class8kep","bbrbm_class9kep","bbrbm_class10kep","bbrbm_class11kep","bbrbm_class12kep"
    ,"bbrbm_class13kep","bbrbm_class14kep","bbrbm_class15kep","bbrbm_class16kep","bbrbm_class17kep"
        ,"bbrbm_class18kep","bbrbm_class19kep","bbrbm_class20kep"]
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
    b = np.concatenate((t1, mnist_images1[i]), axis=0)
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
plt.imshow(mnist_labeled[6][10:794].reshape(28,28))#new_image.reshape(29,29)
fig.add_subplot(rows, columns, 1)
plt.imshow(mnist_labeled[6][0:10].reshape(1,-1))
plt.locator_params(axis="x", nbins=9)
plt.show()
#for m in range(19):
   # show_digit(mnist_labeled[m].reshape(28, 28))
ep = np.array([3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,15000,16000,17000,18000,19000,20000])#700, 800, 900, 1000, 1100, 1200,1300,1400, 1500,1600,1700]


bbrbm = BBRBM(n_visible=794, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)
#for i in range(ep.size):
err = bbrbm.fit(mnist_labeled, n_epoches=20001, batch_size=10)
        #save the weights
    #filename = fname[i] #'weights_class600ep'
    #name1 = name[i] #'bbrbm_class600ep'
    #bbrbm.save_weights(filename,name1)
