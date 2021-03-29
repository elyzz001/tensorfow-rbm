#Hi there, Can you see this
import numpy as np
from numpy import load
import operator
import matplotlib.pyplot as plt
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..')) #this line shold always stay above the next line below
from tfrbm import BBRBM, GBRBM
from tensorflow.examples.tutorials.mnist import input_data



mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
mnist_images1 = mnist.test.images
#mnist_images= np.where(mnist_images1 > 0, 1, 0)
#create the BM
bbrbm = BBRBM(n_visible=784, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)

#load the saved weights
filename = 'weights'
name = 'bbrbm'
bbrbm.load_weights(filename,name)

# fct to plot the images
def show_digit(x):
    plt.imshow(x)
    plt.show()

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

#Test the Reconstruction of the RBM
IMAGE = 50 #26, 31 works well (which is a 6)
image = mnist_images1[IMAGE]

#crop the imag
#crop dimentions
x = 6
y = 6
print(image)
a = image.reshape(28,28)
c = image.reshape(28,28)

img = a[0:16,0:28] #crop the image
#img = cropND(a,(x,y))
#show cropped image
#rint(img)
show_digit(a)
show_digit(img)
#pad the image to make it 780 before feeding it to the BM
imge = np.pad(img, [(0,12), (0,0)], mode='constant')
#print(imge)
show_digit(imge)

#reconstruct

#first run
image_rec1 = bbrbm.reconstruct(imge.reshape(1,-1))
#plot reconstructed image
print(image_rec1)
plt.imshow(image_rec1.reshape(28,28))
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







