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



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images = mnist.test.images
mnist_images1= np.where(mnist_images > 0.5, 1, 0)

print(mnist_images[1])
#create the BM
bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)

#load the saved weights
filename = 'weights'
name = 'bbrbm'
bbrbm.load_weights(filename,name)

# fct to plot the images
def show_digit(x):
    plt.imshow(x,cmap = plt.cm.binary)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title("Original Image" )
    plt.show()
   # plt.title("epoch 5", fontdict=None, loc='center', pad=None)

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

#Test the Reconstruction of the RBM
IMAGE = 20#26, 31 works well (which is a 6)
#image = mnist_images1[IMAGE]

mask_a_or =np.ones(784)
mask_c_or =np.zeros(784)
#prepare first mask
mask_bb = mask_a_or
mask_bb = mask_bb.reshape(28,28)
mask_bb = mask_bb[0:12,0:28] #change 16 to lower number to clamp smaller area
mask_b = np.pad(mask_bb, [(0,16), (0,0)], mode='constant')
#prepare second mask
mask_cc = mask_c_or
mask_cc = mask_cc.reshape(28,28)
mask_cc = mask_cc[0:12,0:28]
mask_c = np.pad(mask_cc, [(0, 16), (0, 0)], mode='constant', constant_values=1)
#crop the imag
#crop dimentions
print('size of mask b')
np.size(mask_b)
print('size of mask c')
np.size(mask_b)
#print(image)
#a = image.reshape(28,28)
#c = image.reshape(28,28)

#img = a[0:16,0:28] #crop the image
#img = a*mask_b
#img_org = img
#imga = img
#img = cropND(a,(x,y))
#show cropped image
#rint(img)
#show_digit(a)
#show_digit(img)
#pad the image to make it 780 before feeding it to the BM
#imge = np.pad(img, [(0,12), (0,0)], mode='constant')
#print(imge)
#show_digit(imge)

#reconstruct

#first run
iter_num = 1
i= 1
j = 1
n_data = mnist_images1.shape[0]
for j in range(n_data):
    print(j)
    image = mnist_images1[j]
    print(image)
    a = image.reshape(28, 28)
    c = image.reshape(28, 28)
    show_digit(image.reshape(28,28))
    # img = a[0:16,0:28] #crop the image
    img = a * mask_b
    img_org = img
    imga = img
    show_digit(img.reshape(28,28))
    for i in range(10001):
        image_rec1 = bbrbm.reconstruct(imga.reshape(1,-1))
    #plot reconstructed image
    #print(image_rec1)

    #iter_num = iter_num + 1
    #i = i + 1
    #img = image_rec1
        image = image_rec1
       # h_st3 = np.greater(image_rec1, np.random.uniform(0, 1, 784))
        #imga1 = h_st3.astype(int)
        imga = img_org + image_rec1.reshape(28, 28) * mask_c

    #close(plt)

#print the result of construction

    #h_st2 = np.greater(image_rec1, np.random.uniform(0,1,784))
    #imga = h_st2.astype(int)
    #imga = img_org + imga.reshape(28, 28) * mask_c

    image_rec_bin = np.greater(image_rec1, np.random.uniform(0,1,784))
    image_bin = image_rec_bin.astype( int)
    print(image_bin)
    #imga = img_org + image_bin.reshape(28, 28) * mask_c

    plt.imshow(imga.reshape(28, 28),cmap = plt.cm.binary)
    plt.colorbar(mappable=None, cax=None, ax=None)
    plt.title("Reconstruction results for iteration : %i  " % i)
    plt.show()

"""
#second run
a = image_rec1.reshape(28,28)

image_rec1 = a[0:16,0:28]
image_rec1  = np.pad(image_rec1 , [(0,12), (0,0)], mode='constant')
image_rec2 = bbrbm.reconstruct(image_rec1.reshape(1,-1))
#plot reconstructed image
print(image_rec2)
plt.imshow(image_rec2.reshape(28,28))
plt.show()

#third run
a

#plot reconstructed image
print(image_rec3)
plt.imshow(image_rec3.reshape(28,28))
plt.show()

IMAGE = 3 #26, 31 works well (which is a 6)

image = mnist_images1[IMAGE]
print(image)
plt.imshow(image.reshape(28, 28))
plt.show()

j = 1
for j in range(100):
    a = image.reshape(28,28)
    image_rec1 = a[0:16,0:28]
    image_rec2  = np.pad(image_rec1 , [(0,12), (0,0)], mode='constant')
    image = bbrbm.reconstruct(image_rec2.reshape(1,-1))
    #print(image)
    #plt.imshow(image.reshape(28, 28))
   # plt.show()
    j = j +1
    print(j)

print(image)
plt.imshow(image.reshape(28,28))
plt.show()
"""







