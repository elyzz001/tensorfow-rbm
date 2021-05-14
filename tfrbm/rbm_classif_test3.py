#Hi there, Can you see this
import numpy as np
import tensorflow as tf
from numpy import load
import operator
import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) #this line shold always stay above the next line below
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data

#load the mnist data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.test.images
mnist_images1= np.where(mnist_images > 0, 1, 0)

#helper fcts
def pad_with(vector, pad_width, iaxis, kwargs): #https://numpy.org/doc/stable/reference/generated/numpy.pad.html
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def show_digit(x,y):
    plt.imshow(x)#,cmap = plt.cm.binary)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title(y)
    #plt.locator_params(axis="x", nbins=30)
    plt.locator_params(axis="y", nbins=30)
    plt.show()

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

#training accuracy variable
accu = [0]
n_data = 1000#mnist_images1.shape[0]
print("accuracy",accu)

#labels pixels
labels_px = np.zeros(10)

#masks
mask_0aa = np.zeros(206)
mask_1aa = np.ones(588)#-7
mask_0bb = np.zeros(588)
mask_1bb = np.ones(206)

#create empty array for the new images
mnist_labeled = np.zeros(794000)
mnist_labeled = mnist_labeled.reshape(n_data,794) #reshape it (2D)
mask_1a = np.zeros(794000)
mask_1a = mask_1a.reshape(n_data,794)
mask_1b = np.zeros(794000)
mask_1b = mask_1b.reshape(n_data,794)

#for loop to add one-hot labels to the images as pixels
for i in range(n_data-1):
    b = np.concatenate((mnist_images1[i],labels_px), axis=0)
    mask_1a[i] = np.concatenate((mask_1aa, mask_0aa), axis=0)
    mask_1b[i] = np.concatenate((mask_0bb, mask_1bb), axis=0)
    mnist_labeled[i] = b#.flatten() #convert the image to 1D and store it

#test to see if corretly added labels
#show_digit(mnist_labeled[5].reshape(28,28),"test for new dim")#new_image.reshape(29,29)
mnist_labeled_backup =  mnist_labeled

#create the BM
bbrbm = BBRBM(n_visible=794, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)

#load the saved weights during training
filename = 'weights_class4'
name = 'bbrbm_class4'
bbrbm.load_weights(filename,name)

#names in case saving reconstruction is desired
fname = ["1-3","2-3","3-3","4-3","5-3","6-3","7-3","8-3","9-3","10-3","11-3","12-3","13-3","14-3","15-3","16-3","17-3","18-3","19-3","20-3"]

#802936312
for j in range(1):

    #reconstruct image for N-MC
    mnist_labeled = mnist_labeled_backup*mask_1a
    show_digit(mask_1a[0][0:784].reshape(28, 28), "mask_1a")
    show_digit(mask_1a[0][784:794].reshape(1, -1), "mask_1a labels")
    show_digit(mask_1b[0][0:784].reshape(28, 28), "mask_1b")
    show_digit(mnist_labeled[0][0:784].reshape(28, 28), "croped input image")
    show_digit(mask_1b[0][784:794].reshape(1, -1), "mask_1b labels")
    show_digit(mnist_labeled[0][784:794].reshape(1, -1), "copped labels")
    for i in range(100):
        image_rec1 = bbrbm.reconstruct(mnist_labeled)

        #print("shape of of rec1",image_rec1.shape)
        #mnist_labeled = image_rec1  # image_rec1.reshape(794,)

        mnist_labeled = mnist_labeled_backup*mask_1a + image_rec1*mask_1b
        #show_digit(image_rec1[0][78:7094].reshape(1, -1), "returned labels")
        #show_digit(image_rec1[0][0:784].reshape(28, 28), "recreated image")
        #show_digit(mnist_labeled[0][0:784].reshape(28, 28), "to feed back image")
        #show_digit(mnist_labeled[0][784:794].reshape(1, -1), "to feed back labels")

#plot the reconstruction results
for n in range(20):
    fig = plt.figure(figsize=(10, 10))

    # setting values to rows and column variables for the figure
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mnist_labeled[n][0:784].reshape(28, 28))  # new_image.reshape(29,29)
    plt.title("Reconstructed Image after %i Iterations" % i)

    fig.add_subplot(rows, columns, 1)
    plt.imshow(image_rec1[n][784:794].reshape(1, -1))
    plt.title("Reconstructed Labels")

    fig.add_subplot(rows, columns, 3)
    plt.imshow(mnist.test.labels[n].reshape(1, -1))  # new_image.reshape(29,29)
    plt.title("Original labels")

    fig.add_subplot(rows, columns, 4)
    plt.imshow(mnist_labeled_backup[n][0:784].reshape(28, 28))  # new_image.reshape(29,29)
    plt.title("Original Image" )

    plt.locator_params(axis="x", nbins=10)
    #plt.show()
    #save figure

    plt.savefig(fname[n], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    # plt.show()
    #plt.close()
    #plt.show()
    
#print("accu  ", accu)
#accuracy = accu[0]/n_data
#print("accuracy  ",accuracy )

