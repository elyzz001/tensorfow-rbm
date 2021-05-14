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
mnist_images = mnist.train.images
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


#for loop to add one-hot labels to the images as pixels
mnist_labeled = np.zeros(18000) #create empty array for the new images
mnist_labeled = mnist_labeled.reshape(20,900) #reshape it (2D)
for i in range(19):
    b = np.pad(mnist_images1[i].reshape(28, 28), 1, pad_with, padder=0)#add contour of zeros around the image
    #x = np.where(mnist.train.labels[i] == 1)
    #b[x,0] = 1 # add the label coressponding to the image on the right most clolumn [0..10]
    mnist_labeled[i] = b.flatten() #convert the image to 1D and store it
#test to see if corretly added labels
show_digit(mnist_labeled[5].reshape(30,30),"test for new dim")#new_image.reshape(29,29)


#print(mnist_images[1])
#create the BM
bbrbm = BBRBM(n_visible=900, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)

#load the saved weights
filename = 'weights_class'
name = 'bbrbm_class'
bbrbm.load_weights(filename,name)



def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

#Test the Reconstruction of the RBM
IMAGE = 20#26, 31 works well (which is a 6)
#image = mnist_images1[IMAGE]

mask_a_or =np.ones(900)
mask_c_or =np.zeros(900)
#prepare first mask
mask_bb = mask_a_or
mask_bb = mask_bb.reshape(30,30)
mask_bb = mask_bb[0:20,0:30] #change 16 to lower number to clamp smaller area
mask_b = np.pad(mask_bb, [(0,10), (0,0)], mode='constant')
show_digit(mask_b.reshape(30, 30), "Mask B")
#prepare second mask
mask_cc = mask_c_or
mask_cc = mask_cc.reshape(30,30)
mask_cc = mask_cc[0:20,0:30]
mask_c = np.pad(mask_cc, [(0, 10), (0, 0)], mode='constant', constant_values=1)
show_digit(mask_c.reshape(30, 30), "Mask C")
#crop the imag
#crop dimentions
print('size of mask b',mask_b.size)

print('size of mask c', mask_c.size)




#reconstruct

#first run
fname = ["1-3","2-3","3-3","4-3","5-3","6-3","7-3","8-3","9-3","10-3","11-3","12-3","13-3","14-3","15-3","16-3","17-3","18-3","19-3","20-3"]
n_data = mnist_labeled.shape[0]
print("number of images", n_data)
#random image for testing
#random_image = np.random.uniform(0,1,784)
#image_rec_bin = np.greater(random_image, np.random.uniform(0,1,784))
#random_image = image_rec_bin.astype( int)
for j in range(n_data-1):
    #random_image = np.random.uniform(0, 1, 784)
    #image_rec_bin = np.greater(random_image, np.random.uniform(0, 1, 784))
    #random_image = image_rec_bin.astype(int)

    #print(j)
    image = mnist_labeled[j]
    #show_digit(image.reshape(30, 30), "Original image")
    a = image.reshape(30, 30)
    c = image.reshape(30, 30)
    #show_digit(image.reshape(28,28))
    # img = a[0:16,0:28] #crop the image
    img = a * mask_b
    img_org = img
    #imga = random_image#imga = img
    #show_digit(img_org.reshape(30,30),"croped input")
    #reconstruct image for N-MC
    for i in range(2):
        image_rec1 = bbrbm.reconstruct(img.reshape(1,-1))
        img= img_org + image_rec1.reshape(30, 30) * mask_c
        #show_digit(image_rec1.reshape(30, 30), "returned image")
        #show_digit(img.reshape(30, 30), "image to be fed")


    #print the result of construction
    plt.imshow(img_org.reshape(30, 30))#,cmap = plt.cm.binary)
   # plt.colorbar(mappable=None, cax=None, ax=None)
    plt.locator_params(axis="y", nbins=30)
    plt.title("Clamped Image")#Reconstruction results for iteration : %i  " % i)
    plt.savefig(fname[j], dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
    #plt.show()
    plt.close()









