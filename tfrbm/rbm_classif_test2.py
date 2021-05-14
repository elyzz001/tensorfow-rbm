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

#labels pixels
accu = [0]
n_data = 1000#mnist_images1.shape[0]
print("accuracy",accu)
t1 = np.zeros(10)
#for loop to add one-hot labels to the images as pixels
mnist_labeled = np.zeros(794000) #create empty array for the new images
mnist_labeled = mnist_labeled.reshape(1000,794) #reshape it (2D)
for i in range(n_data-1):
    #b =mnist_images1[i]#.reshape(28, 28)#add contour of zeros around the image
    #x = np.where(mnist.train.labels[i] == 1)
    b = np.concatenate((t1, mnist_images1[i]), axis=0)
    #b[x,0] = 1 # add the label coressponding to the image on the right most clolumn [0..10]
    mnist_labeled[i] = b#.flatten() #convert the image to 1D and store it
#test to see if corretly added labels
#show_digit(mnist_labeled[5].reshape(28,28),"test for new dim")#new_image.reshape(29,29)


print("minist test size",mnist_images1.shape)
#create the BM
bbrbm = BBRBM(n_visible=794, n_hidden=64, learning_rate=0.01, momentum=0.95, use_tqdm=True)

#load the saved weights
filename = 'weights_class3'
name = 'bbrbm_class3'
bbrbm.load_weights(filename,name)



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
mask_bb = mask_bb[0:18,0:28] #change 16 to lower number to clamp smaller area
mask_b = np.pad(mask_bb, [(0,10), (0,0)], mode='constant')
#show_digit(mask_b.reshape(28, 28), "Mask B")
#prepare second mask
mask_cc = mask_c_or
mask_cc = mask_cc.reshape(28,28)
mask_cc = mask_cc[0:18,0:28]
mask_c = np.pad(mask_cc, [(0, 10), (0, 0)], mode='constant', constant_values=1)
#show_digit(mask_c.reshape(28, 28), "Mask C")
#crop the imag
#crop dimentions
print('size of mask b',mask_b.size)

print('size of mask c', mask_c.size)


#reconstruct

#first run
fname = ["1-3","2-3","3-3","4-3","5-3","6-3","7-3","8-3","9-3","10-3","11-3","12-3","13-3","14-3","15-3","16-3","17-3","18-3","19-3","20-3"]
#n_data = mnist_images1.shape[0]
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
    image = mnist_images1[j]
    #show_digit(image.reshape(28, 28), "Original image")
    #print("image label",mnist.test.labels[j])
    a = image.reshape(28,28)
    c = image.reshape(28, 28)
    #show_digit(image.reshape(28,28))
    # img = a[0:16,0:28] #crop the image
    img = a * mask_b

    img = np.concatenate((t1, img.flatten()), axis=0)
    img_org = img
    #print("shape of of org img", img_org.shape)
    #imga = random_image#imga = img
    #show_digit(img_org[10:794].reshape(28,28),"croped input")
    #reconstruct image for N-MC
    for i in range(100):
        image_rec1 = bbrbm.reconstruct(img.reshape(1,-1))
        #print("shape of of rec1",image_rec1.shape)
        image_rec1 = image_rec1.reshape(794,)
        #print("new shape of of rec1", image_rec1.shape)
        rec_backup = image_rec1
        image_rec1 = image_rec1[10:794].reshape(28,28 )
        #print("size ofa", a.size)
        img= img_org + np.concatenate((t1, (image_rec1 * mask_c).flatten()), axis=0)
        #show_digit(image_rec1.reshape(30, 30), "returned image")
        #show_digit(img.reshape(30, 30), "image to be fed")


    #print the result of construction
    # create figure
    # check if the reconstructed label matches the correct label
    #compare  rec_backup[0:10] to mnist.test.labels[j]
    a1 = rec_backup[0:10]
    a2 = mnist.test.labels[j]
    a3 = np.where(a2 == True)
    a4 = list(a3)
   #print("org label position" , a3[0])
   # print("recondtructed label at orig position", a1[a3[0]])
    if(a1[a3[0]] == 1):
        accu[0] = accu[0] + 1
    #a = np.array_equal(rec_backup[0:10], mnist.test.labels[j])
    #print("org image label", mnist.train.labels[j])
    #print("reconstructed image label", rec_backup[0:10])


    #print("accu  ", accu)
    #print("a of labels result ", a)
    #fig = plt.figure(figsize=(10, 10))

    # setting values to rows and column variables
    """
    rows = 2
    columns = 1
    fig.add_subplot(rows, columns, 2)
    plt.imshow(img[10:794].reshape(28, 28))  # new_image.reshape(29,29)
    plt.title("Reconstructed Image after %i Iterations" %i)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(rec_backup[0:10].reshape(1, -1))
    plt.title("Labels")
    plt.locator_params(axis="x", nbins=10)

    #save figure
    plt.savefig(fname[j], dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    # plt.show()
    plt.close()
    #plt.show()
    """
print("accu  ", accu)
accuracy = accu[0]/n_data
print("accuracy  ",accuracy )












